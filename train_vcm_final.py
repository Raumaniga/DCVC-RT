"""Train the video-only DCVC-RT VCM codec with the paper's video protocol.

Stage ``vimeo7`` trains on Vimeo-90K septuplets. Stage ``long8`` fine-tunes
that model on processed original Vimeo videos using eight-picture clips.
Frame 0 is an external reference seed and all later frames are coded by DMC.
The only trainable parameters are DMC; distortion is machine Feature MSE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
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
HIERARCHICAL_DISTORTION_WEIGHTS = (0.5, 1.2, 0.5, 0.9, 0.5, 1.2, 0.5, 0.9)
DEFAULT_VIMEO_CURRICULUM_FRAMES = (2, 3, 5, 7)
DEFAULT_VIMEO_CURRICULUM_START_EPOCHS = (1, 6, 11, 21)
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


@dataclass(frozen=True)
class TrainingSchedule:
    name: str
    num_frames: int
    qp_offsets: tuple[int, ...]
    distortion_weights: tuple[float, ...]


@dataclass(frozen=True)
class RestoredTrainingState:
    start_epoch: int = 1
    best_loss: float = float("inf")
    stale_validations: int = 0
    active_num_frames: int | None = None


def get_training_schedule(stage: str) -> TrainingSchedule:
    """Return the published hierarchy truncated to the active clip length."""
    frame_counts = {"vimeo7": 7, "long8": 8}
    if stage not in frame_counts:
        raise ValueError(f"Unknown training stage: {stage}")
    num_frames = frame_counts[stage]
    return TrainingSchedule(
        name=stage,
        num_frames=num_frames,
        qp_offsets=QP_OFFSETS[:num_frames],
        distortion_weights=HIERARCHICAL_DISTORTION_WEIGHTS[:num_frames],
    )


def get_epoch_num_frames(args: argparse.Namespace, epoch: int) -> int:
    """Select the Vimeo temporal crop length for the current epoch."""
    schedule = get_training_schedule(args.training_stage)
    if args.training_stage != "vimeo7":
        return schedule.num_frames

    frame_counts = tuple(args.vimeo_curriculum_frames)
    start_epochs = tuple(args.vimeo_curriculum_start_epochs)
    if len(frame_counts) != len(start_epochs):
        raise ValueError(
            "--vimeo-curriculum-frames and --vimeo-curriculum-start-epochs "
            "must have the same length"
        )
    if not frame_counts or start_epochs[0] != 1:
        raise ValueError("The Vimeo curriculum must start at epoch 1")
    if any(
        current >= following
        for current, following in zip(start_epochs, start_epochs[1:])
    ):
        raise ValueError("Vimeo curriculum start epochs must be strictly increasing")
    if any(
        current >= following
        for current, following in zip(frame_counts, frame_counts[1:])
    ):
        raise ValueError("Vimeo curriculum frame counts must be strictly increasing")
    if frame_counts[0] < 2 or frame_counts[-1] != schedule.num_frames:
        raise ValueError(
            f"Vimeo curriculum must start at 2 or more frames and end at "
            f"{schedule.num_frames} frames"
        )

    active_num_frames = frame_counts[0]
    for start_epoch, num_frames in zip(start_epochs, frame_counts, strict=True):
        if epoch < start_epoch:
            break
        active_num_frames = num_frames
    return active_num_frames


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
    checkpoint = torch.load(path, map_location=device, weights_only=True)
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
        num_frames=get_training_schedule(args.training_stage).num_frames,
        training=shuffle,
        samples_per_sequence=args.samples_per_sequence if shuffle else 1,
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
    schedule = get_training_schedule(args.training_stage)
    if frames.shape[1] < 2:
        raise ValueError("A clip needs one seed frame and at least one coded P-frame")
    if args.training_stage == "long8" and frames.shape[1] != schedule.num_frames:
        raise ValueError(
            f"Stage {schedule.name} requires {schedule.num_frames} pictures, "
            f"but received {frames.shape[1]}"
        )
    if args.training_stage == "vimeo7" and frames.shape[1] > schedule.num_frames:
        raise ValueError(
            f"Stage {schedule.name} accepts at most {schedule.num_frames} pictures, "
            f"but received {frames.shape[1]}"
        )

    dmc.clear_dpb()
    dmc.set_curr_poc(0)
    dmc.add_ref_frame(feature=None, frame=frames[:, 0])
    lambda_rd = interpolate_lambda(base_qp, args.lambda_min, args.lambda_max)
    per_frame = []
    for frame_index in range(1, frames.shape[1]):
        coding_qp = base_qp + schedule.qp_offsets[frame_index]
        distortion_weight = schedule.distortion_weights[frame_index]
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


def make_optimizer(
    dmc: DMC,
    learning_rate: float,
    weight_decay: float,
) -> optim.AdamW:
    """Build AdamW without decaying quantization controls, biases or 1-D scales."""
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in dmc.named_parameters():
        if not parameter.requires_grad:
            continue
        leaf_name = name.rsplit(".", 1)[-1]
        no_decay = (
            parameter.ndim <= 1
            or leaf_name == "bias"
            or leaf_name.startswith("q_")
        )
        target = no_decay_parameters if no_decay else decay_parameters
        target.append(parameter)

    return optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


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
        entries.append({key: value.detach() for key, value in details.items()})
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
    training_stage: str,
) -> RestoredTrainingState:
    if path is None:
        return RestoredTrainingState()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    optimizer_config = checkpoint.get("optimizer_config")
    if optimizer_config is None or optimizer_config.get("name") != "AdamW":
        raise ValueError(
            "This resume checkpoint does not contain a compatible AdamW state. "
            "Load its DMC weights with --video-init instead."
        )
    saved_stage = checkpoint.get("training_stage")
    if saved_stage is not None and saved_stage != training_stage:
        raise ValueError(
            f"Checkpoint stage is {saved_stage}, but --training-stage is "
            f"{training_stage}. Use --video-init for a stage transition."
        )
    dmc.load_state_dict(checkpoint["dmc_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    training_state = checkpoint.get("training_state", {})
    return RestoredTrainingState(
        start_epoch=int(checkpoint["epoch"]) + 1,
        best_loss=float(training_state.get("best_loss", float("inf"))),
        stale_validations=int(training_state.get("stale_validations", 0)),
        active_num_frames=training_state.get("active_num_frames"),
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    dmc: DMC,
    criterion: VCMLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    metrics: dict[str, float],
    schedule: TrainingSchedule,
    active_num_frames: int,
    best_loss: float,
    stale_validations: int,
    args: argparse.Namespace,
):
    payload = {
        "schema_version": 6,
        "epoch": epoch,
        "training_stage": schedule.name,
        "dmc_state_dict": dmc.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "training_schedule": {
            "pictures_per_group": schedule.num_frames,
            "qp_offsets": schedule.qp_offsets,
            "distortion_weights": schedule.distortion_weights,
        },
        "training_curriculum": {
            "active_num_frames": active_num_frames,
            "vimeo_frame_counts": tuple(args.vimeo_curriculum_frames),
            "vimeo_start_epochs": tuple(args.vimeo_curriculum_start_epochs),
        },
        "training_state": {
            "best_loss": best_loss,
            "stale_validations": stale_validations,
            "active_num_frames": active_num_frames,
        },
        "optimizer_config": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "quantization_and_bias_weight_decay": 0.0,
        },
        "feature_objective": {
            "layer_indices": criterion.feature_layer_indices,
            "normalized_layer_weights": criterion.layer_weights.detach()
            .cpu()
            .tolist(),
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def copy_checkpoint(source: Path, destination: Path) -> None:
    """Atomically copy an already serialized checkpoint."""
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary_path)
    temporary_path.replace(destination)


def prune_periodic_checkpoints(directory: Path, keep: int) -> None:
    """Keep only the newest ``keep`` files named ``epoch_<number>.pt``."""
    if keep < 0:
        raise ValueError("keep_periodic_checkpoints must be non-negative")
    checkpoints = []
    for path in directory.glob("epoch_*.pt"):
        match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    for _, path in checkpoints[:-keep] if keep else checkpoints:
        path.unlink()


def train(args: argparse.Namespace):
    if args.crop_size % 16:
        raise ValueError("crop_size must be divisible by 16 for DCVC-RT")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.validate_every < 1:
        raise ValueError("validate_every must be at least 1")
    if args.save_every < 0:
        raise ValueError("save_every must be non-negative")
    if args.keep_periodic_checkpoints < 0:
        raise ValueError("keep_periodic_checkpoints must be non-negative")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max_batches must be at least 1")
    if (
        args.val_dir
        and args.max_validation_batches is not None
        and args.max_validation_batches < 1
    ):
        raise ValueError("max_validation_batches must be at least 1")
    if args.grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    schedule = get_training_schedule(args.training_stage)
    if args.resume and args.video_init:
        raise ValueError("Use --resume or --video-init, not both")
    if args.training_stage == "long8" and not (args.video_init or args.resume):
        raise ValueError(
            "Stage long8 is fine-tuning and requires the Vimeo7 checkpoint via "
            "--video-init (or --resume for an interrupted long8 run)"
        )
    interpolate_lambda(0, args.lambda_min, args.lambda_max)
    get_epoch_num_frames(args, 1)

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
    print(
        f"stage={schedule.name}, frames={schedule.num_frames}, "
        f"train_samples={len(train_loader.dataset)}, "
        f"qp_offsets={list(schedule.qp_offsets)}"
    )

    dmc = DMC().to(device)
    if args.video_init:
        load_dmc_weights(dmc, args.video_init, device)
    criterion = VCMLoss(
        args.task_model,
        args.feature_layer_indices,
        args.feature_layer_weights,
        yolov5_repository=args.yolov5_repo,
        yolov5_weights=args.yolov5_weights,
    ).to(device)
    metric_names = (*METRICS, *criterion.layer_metric_names)
    optimizer = make_optimizer(dmc, args.learning_rate, args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
    )
    restored_state = restore(
        args.resume,
        dmc,
        optimizer,
        scheduler,
        device,
        args.training_stage,
    )

    checkpoint_dir = Path(
        args.checkpoint_dir or f"checkpoints/vcm_{args.training_stage}"
    )
    args.checkpoint_dir = str(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(checkpoint_dir / "logs", args, metric_names)
    best_loss = restored_state.best_loss
    stale_validations = restored_state.stale_validations
    previous_active_num_frames = restored_state.active_num_frames
    latest_checkpoint = checkpoint_dir / "latest.pt"
    try:
        for epoch in range(restored_state.start_epoch, args.epochs + 1):
            active_num_frames = get_epoch_num_frames(args, epoch)
            phase_changed = active_num_frames != previous_active_num_frames
            if phase_changed:
                best_loss = float("inf")
                stale_validations = 0
                previous_active_num_frames = active_num_frames
            train_loader.dataset.set_num_frames(active_num_frames)
            dmc.train()
            criterion.train()
            epoch_entries = []
            skipped_batches = 0
            progress = tqdm(
                train_loader,
                desc=(
                    f"epoch {epoch}/{args.epochs} "
                    f"({active_num_frames} frames)"
                ),
            )
            for batch_index, frames in enumerate(progress):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                base_qp = random.randint(0, 63)
                loss, details = run_gop(
                    dmc,
                    criterion,
                    frames.to(device, non_blocking=True),
                    base_qp,
                    args,
                )
                if not torch.isfinite(loss):
                    skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(status="skip non-finite loss")
                    continue
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    dmc.parameters(),
                    args.grad_clip,
                )
                if not torch.isfinite(grad_norm):
                    skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(status="skip non-finite gradient")
                    continue
                optimizer.step()
                detached_details = {
                    key: value.detach() for key, value in details.items()
                }
                epoch_entries.append(detached_details)
                progress.set_postfix(
                    loss=f"{float(loss.detach()):.5f}",
                    bpp=f"{float(details['estimated_bpp'].detach()):.5f}",
                )

            train_metrics = aggregate(epoch_entries)
            should_validate = validation_loader is not None and (
                phase_changed
                or epoch % args.validate_every == 0
                or epoch == args.epochs
            )
            val_metrics = (
                validate(dmc, criterion, validation_loader, args, device)
                if should_validate
                else None
            )
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step()
            logger.log(epoch, epoch_learning_rate, train_metrics, val_metrics)

            print(
                f"epoch {epoch}: loss={train_metrics['total_loss']:.6f}, "
                f"bpp={train_metrics['estimated_bpp']:.6f}, "
                f"feature_mse={train_metrics['feature_mse']:.8f}, "
                f"mean_base_qp={train_metrics['base_qp']:.2f}, "
                f"skipped_batches={skipped_batches}"
            )
            if val_metrics is not None:
                print(
                    f"validation: loss={val_metrics['total_loss']:.6f}, "
                    f"bpp={val_metrics['estimated_bpp']:.6f}, "
                    f"feature_mse={val_metrics['feature_mse']:.8f}"
                )
            selection_metrics = (
                val_metrics
                if validation_loader is not None
                else train_metrics
            )
            improved = (
                selection_metrics is not None
                and selection_metrics["total_loss"] < best_loss
            )
            should_stop = False
            if improved:
                best_loss = selection_metrics["total_loss"]
                stale_validations = 0
            elif selection_metrics is not None:
                stale_validations += 1
                curriculum_complete = (
                    args.training_stage != "vimeo7"
                    or active_num_frames == schedule.num_frames
                )
                if (
                    curriculum_complete
                    and args.patience
                    and stale_validations >= args.patience
                ):
                    should_stop = True

            checkpoint_metrics = selection_metrics or train_metrics
            save_checkpoint(
                latest_checkpoint,
                epoch,
                dmc,
                criterion,
                optimizer,
                scheduler,
                checkpoint_metrics,
                schedule,
                active_num_frames,
                best_loss,
                stale_validations,
                args,
            )
            if improved:
                copy_checkpoint(latest_checkpoint, checkpoint_dir / "best.pt")
            if args.save_every and epoch % args.save_every == 0:
                copy_checkpoint(
                    latest_checkpoint,
                    checkpoint_dir / f"epoch_{epoch}.pt",
                )
                prune_periodic_checkpoints(
                    checkpoint_dir,
                    args.keep_periodic_checkpoints,
                )
            if should_stop:
                print(
                    "Early stopping after "
                    f"{args.patience} non-improving validation checks."
                )
                break
    finally:
        logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-stage",
        choices=("vimeo7", "long8"),
        default="vimeo7",
        help="vimeo7: initial Vimeo-90K training; long8: long-Vimeo fine-tuning",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Vimeo-90K sequences directory or processed long-Vimeo frame root",
    )
    parser.add_argument("--val-dir", help="Optional validation frame-sequence directory")
    parser.add_argument("--train-list", help="Sequence list relative to data-dir (or an absolute path)")
    parser.add_argument("--val-list", help="Sequence list relative to val-dir (or an absolute path)")
    parser.add_argument(
        "--checkpoint-dir",
        help="Defaults to checkpoints/vcm_<training-stage>",
    )
    parser.add_argument("--video-init", help="Optional pretrained DMC checkpoint; no image checkpoint is used")
    parser.add_argument("--resume", help="Resume a checkpoint produced by this script")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--samples-per-sequence",
        type=int,
        default=1,
        help="Random temporal crops drawn per sequence per training epoch",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW decay; quantization controls, biases and 1-D scales use zero",
    )
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=(60, 80))
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--lambda-min", type=float, default=1.0)
    parser.add_argument("--lambda-max", type=float, default=768.0)
    parser.add_argument(
        "--vimeo-curriculum-frames",
        type=int,
        nargs="+",
        default=DEFAULT_VIMEO_CURRICULUM_FRAMES,
        help="Increasing Vimeo7 temporal crop lengths; must end at 7",
    )
    parser.add_argument(
        "--vimeo-curriculum-start-epochs",
        type=int,
        nargs="+",
        default=DEFAULT_VIMEO_CURRICULUM_START_EPOCHS,
        help="Epoch where each Vimeo7 temporal crop length becomes active",
    )
    parser.add_argument("--validation-qp", type=int, default=32, choices=range(64))
    parser.add_argument(
        "--validate-every",
        type=int,
        default=5,
        help="Run validation every N epochs and at each curriculum transition",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=100,
        help="Number of deterministic validation batches per check",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save epoch_N.pt every N epochs; 0 disables periodic snapshots",
    )
    parser.add_argument(
        "--keep-periodic-checkpoints",
        type=int,
        default=2,
        help="Number of newest epoch_N.pt snapshots to retain",
    )
    parser.add_argument("--task-model", default="yolov5s")
    parser.add_argument(
        "--yolov5-repo",
        help="Optional local YOLOv5 v7 repository for Kaggle offline training",
    )
    parser.add_argument(
        "--yolov5-weights",
        help="Optional local YOLOv5 .pt weights; pairs with --yolov5-repo offline",
    )
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
