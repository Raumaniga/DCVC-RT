"""WG2-style machine-task evaluation for the P-frame-only DCVC-RT VCM codec.

Training remains differentiable and uses ``DMC.forward_train``. This script is
evaluation-only: it uses ``DMC.compress`` and ``DMC.decompress`` to measure
actual sequence-container bytes, then measures detector mAP against real
ground-truth labels at four rate points.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as transforms
from tqdm import tqdm

from src.models.video_model import DMC
from src.models.yolov5_extractor import load_yolov5
from src.utils.bd_rate import compute_bd_metric, compute_bd_rate, pareto_front
from src.utils.detection_map import DetectionMAP
from src.utils.vcm_bitstream import VCMSequenceReader, VCMSequenceWriter
from src.utils.vcm_eval_dataset import AnnotatedVideoDataset, VideoSequence


QP_OFFSETS = (0, 8, 0, 4, 0, 4, 0, 4)
RATE_POINT_COUNT = 4


def load_dmc_weights(model: DMC, path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "dmc_state_dict",
            checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)),
        )
    else:
        state = checkpoint
    model.load_state_dict(
        {key.removeprefix("module."): value for key, value in state.items()}
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "method"


def pad_frame(frame: torch.Tensor, model: DMC, device: torch.device) -> torch.Tensor:
    frame = frame.unsqueeze(0).to(device, non_blocking=True)
    padding_right, padding_bottom = model.get_padding_size(
        frame.shape[-2],
        frame.shape[-1],
        64,
    )
    return F.pad(frame, (0, padding_right, 0, padding_bottom), mode="replicate")


def as_yolo_image(frame: torch.Tensor) -> np.ndarray:
    return (
        frame.detach()
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .clip(0, 1)
        * 255
    ).astype(np.uint8)


def coding_qp(base_qp: int, frame_index: int) -> int:
    return base_qp + QP_OFFSETS[frame_index % len(QP_OFFSETS)]


@torch.inference_mode()
def encode_sequence(
    model: DMC,
    dataset: AnnotatedVideoDataset,
    sequence: VideoSequence,
    base_qp: int,
    output_path: Path,
    device: torch.device,
) -> None:
    """Encode P-frames to a real entropy-coded sequence container."""
    model.clear_dpb()
    model.set_curr_poc(0)
    seed = pad_frame(dataset.load_frame(sequence.frame_paths[0]), model, device)
    model.add_ref_frame(feature=None, frame=seed)

    with VCMSequenceWriter(
        output_path,
        width=sequence.width,
        height=sequence.height,
        fps=sequence.fps,
        coded_frames=sequence.frame_count - 1,
        external_seed=True,
        two_entropy_coders=False,
    ) as writer:
        for frame_index in range(1, sequence.frame_count):
            frame = pad_frame(
                dataset.load_frame(sequence.frame_paths[frame_index]),
                model,
                device,
            )
            qp = coding_qp(base_qp, frame_index)
            encoded = model.compress(frame, qp)
            writer.write_frame(qp, encoded["bit_stream"])
    torch.cuda.synchronize(device)


@torch.inference_mode()
def decode_and_evaluate_sequence(
    model: DMC,
    detector,
    evaluator: DetectionMAP,
    dataset: AnnotatedVideoDataset,
    sequence: VideoSequence,
    bitstream_path: Path,
    device: torch.device,
    detector_size: int,
    first_image_id: int,
    reconstruction_dir: Path | None,
) -> int:
    """Decode a sequence and add ground-truth detection results to mAP."""
    model.clear_dpb()
    model.set_curr_poc(0)
    seed = pad_frame(dataset.load_frame(sequence.frame_paths[0]), model, device)
    model.add_ref_frame(feature=None, frame=seed)

    with VCMSequenceReader(bitstream_path) as reader:
        header = reader.header
        if (header.width, header.height) != (sequence.width, sequence.height):
            raise ValueError(f"Bitstream resolution mismatch for {sequence.name}")
        if header.coded_frames != sequence.frame_count - 1:
            raise ValueError(f"Bitstream frame-count mismatch for {sequence.name}")
        if not header.external_seed:
            raise ValueError("This project requires the external-seed P-frame protocol")

        sps = {
            "height": sequence.height,
            "width": sequence.width,
            "ec_part": int(header.two_entropy_coders),
            "use_ada_i": 0,
        }
        image_id = first_image_id
        for frame_index, packet in enumerate(reader.frames(), start=1):
            decoded = model.decompress(packet.bitstream, sps, packet.qp)
            reconstructed = decoded["x_hat"][
                :,
                :,
                : sequence.height,
                : sequence.width,
            ]
            detections = detector(
                [as_yolo_image(reconstructed)],
                size=detector_size,
            ).xyxy[0].detach().cpu()

            target_boxes, target_classes = dataset.load_ground_truth(
                sequence.label_paths[frame_index],
                sequence.width,
                sequence.height,
            )
            if len(target_classes) and int(target_classes.max()) >= len(detector.names):
                raise ValueError(
                    f"{sequence.label_paths[frame_index]} contains class "
                    f"{int(target_classes.max())}, but the task model only "
                    f"defines {len(detector.names)} classes"
                )
            evaluator.add(
                image_id=image_id,
                predicted_boxes=detections[:, :4],
                predicted_scores=detections[:, 4],
                predicted_classes=detections[:, 5].long(),
                target_boxes=target_boxes,
                target_classes=target_classes,
            )

            if reconstruction_dir is not None:
                output_path = (
                    reconstruction_dir
                    / safe_name(sequence.name)
                    / sequence.frame_paths[frame_index].with_suffix(".png").name
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                transforms.to_pil_image(reconstructed[0].float().cpu()).save(output_path)
            image_id += 1

    torch.cuda.synchronize(device)
    return image_id


def sequence_rate_record(
    sequence: VideoSequence,
    bitstream_path: Path,
) -> dict[str, float | int | str]:
    coded_frames = sequence.frame_count - 1
    actual_bits = bitstream_path.stat().st_size * 8
    total_pixels = coded_frames * sequence.width * sequence.height
    return {
        "name": sequence.name,
        "bitstream_file": str(bitstream_path),
        "actual_bits": actual_bits,
        "actual_bpp": actual_bits / total_pixels,
        "kbps": actual_bits * sequence.fps / (1000.0 * coded_frames),
        "fps": sequence.fps,
        "width": sequence.width,
        "height": sequence.height,
        "coded_frames": coded_frames,
    }


def aggregate_rate(sequence_records: list[dict]) -> dict[str, float | int]:
    total_bits = sum(record["actual_bits"] for record in sequence_records)
    total_pixels = sum(
        record["coded_frames"] * record["width"] * record["height"]
        for record in sequence_records
    )
    total_duration = sum(
        record["coded_frames"] / record["fps"] for record in sequence_records
    )
    return {
        "actual_bits": int(total_bits),
        "actual_bpp": float(total_bits / total_pixels),
        "kbps": float(total_bits / total_duration / 1000.0),
        "coded_frames": int(
            sum(record["coded_frames"] for record in sequence_records)
        ),
    }


def evaluate_codec(args: argparse.Namespace) -> None:
    if len(args.qps) != RATE_POINT_COUNT or len(set(args.qps)) != RATE_POINT_COUNT:
        raise ValueError("Exactly four distinct base QPs are required")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Actual DMC bitstream coding requires CUDA. Training with "
            "forward_train remains separate and differentiable."
        )

    device = torch.device(f"cuda:{args.cuda_index}")
    dataset = AnnotatedVideoDataset(args.data_dir, args.dataset_manifest)
    sequences = list(dataset)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError("No evaluation sequences were selected")

    model = DMC().to(device).eval()
    load_dmc_weights(model, args.video_ckpt, device)
    try:
        model.update(force_zero_thres=args.force_zero_thres)
    except ImportError as error:
        raise RuntimeError(
            "Actual bitstream evaluation requires the MLCodec_extensions_cpp "
            "entropy-coder extension. Build src/cpp first."
        ) from error
    model.set_use_two_entropy_coders(False)

    detector = load_yolov5(args.task_model).to(device).eval()
    detector.conf = args.confidence_threshold
    detector.iou = args.nms_iou_threshold
    detector.max_det = args.max_detections

    method_name = safe_name(args.method_name)
    bitstream_root = Path(args.bitstream_dir) / method_name
    reconstruction_root = (
        Path(args.reconstruction_dir) / method_name
        if args.save_reconstructions
        else None
    )

    points = []
    for base_qp in args.qps:
        evaluator = DetectionMAP()
        sequence_records = []
        next_image_id = 0
        progress = tqdm(sequences, desc=f"{method_name}: base QP {base_qp}")
        for sequence in progress:
            bitstream_path = (
                bitstream_root
                / f"qp_{base_qp:02d}"
                / f"{safe_name(sequence.name)}.bin"
            )
            encode_sequence(
                model,
                dataset,
                sequence,
                base_qp,
                bitstream_path,
                device,
            )
            next_image_id = decode_and_evaluate_sequence(
                model,
                detector,
                evaluator,
                dataset,
                sequence,
                bitstream_path,
                device,
                args.detector_size,
                next_image_id,
                (
                    reconstruction_root / f"qp_{base_qp:02d}"
                    if reconstruction_root is not None
                    else None
                ),
            )
            sequence_records.append(
                sequence_rate_record(sequence, bitstream_path)
            )

        point = {
            "base_qp": base_qp,
            **aggregate_rate(sequence_records),
            **evaluator.compute(),
            "sequences": sequence_records,
        }
        points.append(point)

    output = {
        "schema_version": 4,
        "method": args.method_name,
        "codec": "DMC-only P-frame VCM",
        "protocol": "external seed excluded; all coded P-frames included",
        "rate_source": "actual sequence-container bytes including headers",
        "rate_points": RATE_POINT_COUNT,
        "task": "object_detection",
        "task_model": args.task_model,
        "ground_truth": "normalized YOLO labels from evaluation manifest",
        "points": points,
    }
    output_path = Path(args.output_dir) / f"{method_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved actual-bitstream mAP results to {output_path}")


def load_results(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 4:
        raise ValueError(f"{path} does not use actual-bitstream evaluation schema v4")
    if len(data.get("points", [])) != RATE_POINT_COUNT:
        raise ValueError(f"{path} must contain exactly four rate points")
    return data


def curve_arrays(data: dict, rate_key: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    rates = np.asarray([point[rate_key] for point in data["points"]], dtype=np.float64)
    quality = np.asarray([point[metric] for point in data["points"]], dtype=np.float64)
    return pareto_front(rates, quality)


def save_curve_csv(
    anchor: dict,
    candidate: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "base_qp",
                "actual_bpp",
                "kbps",
                "map50",
                "map5095",
            ),
        )
        writer.writeheader()
        for data in (anchor, candidate):
            for point in data["points"]:
                writer.writerow(
                    {
                        key: value
                        for key, value in {
                            "method": data["method"],
                            "base_qp": point["base_qp"],
                            "actual_bpp": point["actual_bpp"],
                            "kbps": point["kbps"],
                            "map50": point["map50"],
                            "map5095": point["map5095"],
                        }.items()
                    }
                )


def plot_rd_curve(
    anchor: dict,
    candidate: dict,
    rate_key: str,
    metric: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("RD-curve plotting requires matplotlib") from error

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    for data, marker in ((anchor, "o"), (candidate, "s")):
        points = sorted(data["points"], key=lambda point: point[rate_key])
        rates = [point[rate_key] for point in points]
        quality = [point[metric] for point in points]
        axis.plot(rates, quality, marker=marker, linewidth=2, label=data["method"])
        for point in points:
            axis.annotate(
                f"q={point['base_qp']}",
                (point[rate_key], point[metric]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_xlabel("Actual BPP" if rate_key == "actual_bpp" else "Actual bitrate (kbps)")
    axis.set_ylabel("mAP@0.5" if metric == "map50" else "mAP@[0.5:0.95]")
    axis.set_title("VCM Rate–Accuracy Curve")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def compare_bd_rate(args: argparse.Namespace) -> None:
    anchor = load_results(args.anchor_results)
    candidate = load_results(args.candidate_results)
    anchor_rate, anchor_quality = curve_arrays(anchor, args.rate, args.metric)
    candidate_rate, candidate_quality = curve_arrays(candidate, args.rate, args.metric)

    result = {
        "anchor": anchor["method"],
        "candidate": candidate["method"],
        "rate": args.rate,
        "metric": args.metric,
        "bd_rate_percent": compute_bd_rate(
            anchor_rate,
            anchor_quality,
            candidate_rate,
            candidate_quality,
        ),
        "bd_metric": compute_bd_metric(
            anchor_rate,
            anchor_quality,
            candidate_rate,
            candidate_quality,
        ),
        "interpretation": "negative BD-rate means bitrate saving at equal mAP",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bd_rate_map.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    save_curve_csv(anchor, candidate, output_dir / "rd_points.csv")
    plot_rd_curve(
        anchor,
        candidate,
        args.rate,
        "map50",
        output_dir / f"rd_curve_{args.rate}_map50.png",
    )
    plot_rd_curve(
        anchor,
        candidate,
        args.rate,
        "map5095",
        output_dir / f"rd_curve_{args.rate}_map5095.png",
    )
    print(json.dumps(result, indent=2))


def summarize_training(args: argparse.Namespace) -> None:
    summary = []
    for path in sorted(Path(args.log_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            summary.append({"run": path.stem, "epochs": len(rows), "final": rows[-1]})
    output_path = Path(args.output_dir) / "training_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved training summary to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("codec", "bdrate", "training"))
    parser.add_argument("--data-dir", help="Root containing evaluation frames and labels")
    parser.add_argument("--dataset-manifest", help="Full-resolution evaluation manifest JSON")
    parser.add_argument("--video-ckpt", help="DMC-only checkpoint")
    parser.add_argument("--method-name", default="dcvc_rt_vcm")
    parser.add_argument(
        "--qps",
        type=int,
        nargs=RATE_POINT_COUNT,
        default=(0, 21, 42, 63),
        choices=range(64),
        metavar=("QP1", "QP2", "QP3", "QP4"),
    )
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--force-zero-thres", type=float)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--task-model", default="yolov5s")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--bitstream-dir", default="output/bitstreams")
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument("--reconstruction-dir", default="output/reconstructions")
    parser.add_argument("--anchor-results")
    parser.add_argument("--candidate-results")
    parser.add_argument("--rate", default="actual_bpp", choices=("actual_bpp", "kbps"))
    parser.add_argument("--metric", default="map5095", choices=("map50", "map5095"))
    parser.add_argument("--log-dir", default="checkpoints/vcm_video/logs")
    parser.add_argument("--output-dir", default="output/evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "codec":
        if not arguments.data_dir or not arguments.dataset_manifest or not arguments.video_ckpt:
            raise ValueError(
                "codec mode requires --data-dir, --dataset-manifest and --video-ckpt"
            )
        evaluate_codec(arguments)
    elif arguments.mode == "bdrate":
        if not arguments.anchor_results or not arguments.candidate_results:
            raise ValueError(
                "bdrate mode requires --anchor-results and --candidate-results"
            )
        compare_bd_rate(arguments)
    else:
        summarize_training(arguments)
