"""Train the video-only DCVC-RT VCM codec.

For every training GOP, frame 0 is an external decoded reference seed.
Frames 1..7 are P-frames coded by the trainable DMC model. Training samples
one base QP in [0, 63] per iteration and applies the DCVC-RT hierarchical
QP schedule [0, 8, 0, 4, 0, 4, 0, 4]. The only trainable parameters are DMC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vcm_loss import VCMLoss
from src.models.video_model import DMC
from src.models.yolov5_extractor import DEFAULT_FEATURE_LAYER_INDICES
from src.utils.dataset import VideoSequenceDataset


QP_OFFSETS = (0, 8, 0, 4, 0, 4, 0, 4)
HIERARCHICAL_DISTORTION_WEIGHTS = (0.5, 1.2, 0.5, 0.9)
METRICS = (
    "total_loss",
    "estimated_bpp",
    "feature_mse",
    "weighted_feature_mse",
    "lambda_rd",
    "distortion_weight",
    "base_qp",
    "coding_qp",
)


def interpolate_lambda(base_qp: int, lambda_min: float, lambda_max: float) -> float:
    """Log-linearly interpolate lambda exactly as DCVC-FM."""
    if not 0 <= base_qp <= 63:
        raise ValueError("base_qp must be in [0, 63]")
    if not 0 < lambda_min <= lambda_max:
        raise ValueError("lambda range must satisfy 0 < lambda_min <= lambda_max")
    position = base_qp / 63.0
    return math.exp(
        math.log(lambda_min)
        + position * (math.log(lambda_max) - math.log(lambda_min))
    )


def load_dmc_weights(model: DMC, path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "dmc_state_dict",
            checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)),
        )
    else:
        state = checkpoint
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})


def make_loader(
    args: argparse.Namespace,
    root_dir: str,
    list_file: str | Path | None,
    shuffle: bool,
) -> DataLoader:
    dataset = VideoSequenceDataset(
        root_dir,
        list_file=list_file,
        crop_size=args.crop_size,
        num_frames=args.gop_size,
        training=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def run_gop(
    dmc: DMC,
    criterion: VCMLoss,
    frames: torch.Tensor,
    base_qp: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Code P-frames in one GOP and return their mean VCM loss.

    ``frames[:, 0]`` is deliberately not sent through an image codec. It is an
    externally supplied decoded reference, which is the required starting
    condition for a P-frame-only DMC codec.
    """
    if frames.shape[1] < 2:
        raise ValueError("gop_size must be at least 2: one seed frame and one coded P-frame")
    if frames.shape[1] != len(QP_OFFSETS):
        raise ValueError(f"DCVC-RT hierarchical training requires exactly {len(QP_OFFSETS)} pictures")

    dmc.clear_dpb()
    dmc.set_curr_poc(0)
    dmc.add_ref_frame(feature=None, frame=frames[:, 0])
    lambda_rd = interpolate_lambda(base_qp, args.lambda_min, args.lambda_max)
    per_frame = []
    for frame_index in range(1, frames.shape[1]):
        coding_qp = base_qp + QP_OFFSETS[frame_index]
        distortion_weight = HIERARCHICAL_DISTORTION_WEIGHTS[
            frame_index % len(HIERARCHICAL_DISTORTION_WEIGHTS)
        ]
        original = frames[:, frame_index]
        reconstructed, estimated_bpp = dmc.forward_train(original, coding_qp)
        details = criterion(
            original,
            reconstructed,
            estimated_bpp,
            lambda_rd,
            distortion_weight=distortion_weight,
            return_details=True,
        )
        details["base_qp"] = details["total_loss"].new_tensor(float(base_qp))
        details["coding_qp"] = details["total_loss"].new_tensor(float(coding_qp))
        per_frame.append(details)

    return (
        torch.stack([entry["total_loss"] for entry in per_frame]).mean(),
        {
            key: torch.stack([entry[key] for entry in per_frame]).mean()
            for key in (*METRICS, *criterion.layer_metric_names)
        },
    )


def aggregate(entries: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    if not entries:
        raise RuntimeError("No batches were processed")
    return {
        key: sum(float(entry[key].detach()) for entry in entries) / len(entries)
        for key in entries[0]
    }


@torch.no_grad()
def validate(
    dmc: DMC,
    criterion: VCMLoss,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    dmc.eval()
    criterion.eval()
    entries = []
    for batch_index, frames in enumerate(loader):
        if args.max_validation_batches is not None and batch_index >= args.max_validation_batches:
            break
        _, details = run_gop(dmc, criterion, frames.to(device), args.validation_qp, args)
        entries.append(details)
    return aggregate(entries)


class TrainingLogger:
    def __init__(
        self,
        directory: Path,
        args: argparse.Namespace,
        metric_names: tuple[str, ...],
    ):
        directory.mkdir(parents=True, exist_ok=True)
        run_name = f"video_vcm_{datetime.now():%Y%m%d_%H%M%S}"
        self.metric_names = metric_names
        self.path = directory / f"{run_name}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "epoch",
                "learning_rate",
                *[f"train_{key}" for key in self.metric_names],
                *[f"val_{key}" for key in self.metric_names],
            ],
        )
        self.writer.writeheader()
        (directory / f"{run_name}.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    def log(
        self,
        epoch: int,
        learning_rate: float,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ):
        row = {"epoch": epoch, "learning_rate": learning_rate}
        row.update(
            {f"train_{key}": train_metrics[key] for key in self.metric_names}
        )
        if val_metrics is not None:
            row.update(
                {f"val_{key}": val_metrics[key] for key in self.metric_names}
            )
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def restore(
    path: str | None,
    dmc: DMC,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    if path is None:
        return 1
    checkpoint = torch.load(path, map_location=device)
    dmc.load_state_dict(checkpoint["dmc_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"]) + 1


def save_checkpoint(
    path: Path,
    epoch: int,
    dmc: DMC,
    criterion: VCMLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    metrics: dict[str, float],
):
    torch.save(
        {
            "schema_version": 3,
            "epoch": epoch,
            "dmc_state_dict": dmc.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "training_schedule": {
                "pictures_per_group": len(QP_OFFSETS),
                "qp_offsets": QP_OFFSETS,
                "distortion_weights": HIERARCHICAL_DISTORTION_WEIGHTS,
            },
            "feature_objective": {
                "layer_indices": criterion.feature_layer_indices,
                "normalized_layer_weights": criterion.layer_weights.detach()
                .cpu()
                .tolist(),
            },
        },
        path,
    )


def train(args: argparse.Namespace):
    if args.crop_size % 16:
        raise ValueError("crop_size must be divisible by 16 for DCVC-RT")
    if args.gop_size != len(QP_OFFSETS):
        raise ValueError(
            f"gop_size must be {len(QP_OFFSETS)} for QP offsets {list(QP_OFFSETS)}"
        )
    interpolate_lambda(0, args.lambda_min, args.lambda_max)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(args, args.data_dir, args.train_list, shuffle=True)
    validation_loader = (
        make_loader(args, args.val_dir, args.val_list, shuffle=False)
        if args.val_dir
        else None
    )

    dmc = DMC().to(device)
    if args.video_init:
        load_dmc_weights(dmc, args.video_init, device)
    criterion = VCMLoss(
        args.task_model,
        args.feature_layer_indices,
        args.feature_layer_weights,
    ).to(device)
    metric_names = (*METRICS, *criterion.layer_metric_names)
    optimizer = optim.Adam(dmc.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
    )
    start_epoch = restore(args.resume, dmc, optimizer, scheduler, device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(checkpoint_dir / "logs", args, metric_names)
    best_loss = float("inf")
    stale_epochs = 0
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            dmc.train()
            criterion.train()
            epoch_entries = []
            progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
            for batch_index, frames in enumerate(progress):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                base_qp = random.randint(0, 63)
                loss, details = run_gop(dmc, criterion, frames.to(device), base_qp, args)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dmc.parameters(), args.grad_clip)
                optimizer.step()
                epoch_entries.append(details)
                progress.set_postfix(
                    loss=f"{float(loss.detach()):.5f}",
                    bpp=f"{float(details['estimated_bpp'].detach()):.5f}",
                )

            train_metrics = aggregate(epoch_entries)
            val_metrics = validate(dmc, criterion, validation_loader, args, device) if validation_loader else None
            scheduler.step()
            logger.log(epoch, scheduler.get_last_lr()[0], train_metrics, val_metrics)
            selected = val_metrics or train_metrics
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch}.pt",
                epoch,
                dmc,
                criterion,
                optimizer,
                scheduler,
                selected,
            )

            print(
                f"epoch {epoch}: loss={train_metrics['total_loss']:.6f}, "
                f"bpp={train_metrics['estimated_bpp']:.6f}, "
                f"feature_mse={train_metrics['feature_mse']:.8f}, "
                f"mean_base_qp={train_metrics['base_qp']:.2f}"
            )
            if selected["total_loss"] < best_loss:
                best_loss = selected["total_loss"]
                stale_epochs = 0
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    epoch,
                    dmc,
                    criterion,
                    optimizer,
                    scheduler,
                    selected,
                )
            else:
                stale_epochs += 1
                if args.patience and stale_epochs >= args.patience:
                    print(f"Early stopping after {args.patience} non-improving epochs.")
                    break
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing long frame sequences")
    parser.add_argument("--val-dir", help="Optional validation frame-sequence directory")
    parser.add_argument("--train-list", help="Sequence list relative to data-dir (or an absolute path)")
    parser.add_argument("--val-list", help="Sequence list relative to val-dir (or an absolute path)")
    parser.add_argument("--checkpoint-dir", default="checkpoints/vcm_video")
    parser.add_argument("--video-init", help="Optional pretrained DMC checkpoint; no image checkpoint is used")
    parser.add_argument("--resume", help="Resume a checkpoint produced by this script")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--gop-size",
        type=int,
        default=8,
        help="Fixed at 8: one external seed plus seven coded P-frames",
    )
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=(60, 80))
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--lambda-min", type=float, default=1.0)
    parser.add_argument("--lambda-max", type=float, default=768.0)
    parser.add_argument("--validation-qp", type=int, default=32, choices=range(64))
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--task-model", default="yolov5s")
    parser.add_argument(
        "--feature-layer-indices",
        type=int,
        nargs="+",
        default=DEFAULT_FEATURE_LAYER_INDICES,
        help="Ascending YOLOv5 backbone layers used for multi-level Feature MSE",
    )
    parser.add_argument(
        "--feature-layer-weights",
        type=float,
        nargs="+",
        help="Positive per-layer weights; defaults to equal and is normalized to sum to one",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
