"""Evaluate an HEVC HM anchor with actual bitrate and object-detection mAP.

This script is evaluation-only. It does not import or modify the VCM training
loop. For each of four QPs it performs:

RGB frames -> FFmpeg YUV 4:2:0 -> HM encode/reconstruct -> FFmpeg RGB
-> frozen YOLOv5 -> mAP.

The default conditional-P-frame protocol matches this project's DMC evaluator:
frame 0 initializes prediction, its HM-reported picture bits and its mAP are
excluded, while sequence-level HEVC headers remain counted. This is a
conditional comparison, not a complete independently decodable HEVC rate.
Use ``--protocol all-frames`` to count the complete HEVC bitstream instead.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models.yolov5_extractor import load_yolov5
from src.utils.detection_map import DetectionMAP
from src.utils.evaluation_protocol import (
    ALL_FRAMES_PROTOCOL,
    EXTERNAL_SEED_PROTOCOL,
    dataset_summary,
    detector_config,
    evaluation_id,
)
from src.utils.vcm_eval_dataset import AnnotatedVideoDataset, VideoSequence


RATE_POINT_COUNT = 4
PROTOCOLS = {
    "conditional-pframes": EXTERNAL_SEED_PROTOCOL,
    "all-frames": ALL_FRAMES_PROTOCOL,
}
POC_BITS_PATTERN = re.compile(
    r"\bPOC\s+(-?\d+)\b.*?\)\s+([0-9][0-9,]*)\s+bits\b",
    flags=re.IGNORECASE,
)
POC_PROGRESS_PATTERN = re.compile(r"\bPOC\s+(-?\d+)\b", flags=re.IGNORECASE)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def resolve_executable(value: str, label: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise FileNotFoundError(f"{label} executable not found: {value}")


def run_command(command: list[str], label: str) -> str:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}.\n"
            f"Command: {subprocess.list2cmdline(command)}\n{tail}"
        )
    return completed.stdout


def ffconcat_quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def write_concat_file(sequence: VideoSequence, output_path: Path) -> None:
    lines = ["ffconcat version 1.0"]
    lines.extend(f"file '{ffconcat_quote(path)}'" for path in sequence.frame_paths)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def raw_frame_bytes(width: int, height: int, bit_depth: int) -> int:
    bytes_per_sample = 1 if bit_depth == 8 else 2
    return width * height * 3 // 2 * bytes_per_sample


def rgb_to_yuv(
    ffmpeg: str,
    sequence: VideoSequence,
    concat_path: Path,
    yuv_path: Path,
    bit_depth: int,
) -> None:
    pixel_format = "yuv420p" if bit_depth == 8 else "yuv420p10le"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vsync",
        "0",
        "-frames:v",
        str(sequence.frame_count),
        "-vf",
        "scale=in_range=pc:out_range=tv:out_color_matrix=bt709",
        "-pix_fmt",
        pixel_format,
        "-f",
        "rawvideo",
        "-y",
        str(yuv_path),
    ]
    run_command(command, f"FFmpeg RGB-to-YUV for {sequence.name}")
    expected_size = (
        raw_frame_bytes(sequence.width, sequence.height, bit_depth)
        * sequence.frame_count
    )
    actual_size = yuv_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{sequence.name}: converted YUV has {actual_size} bytes; "
            f"expected {expected_size}. Check FFmpeg frame enumeration."
        )


def hm_encode(
    encoder: str,
    config: Path,
    sequence: VideoSequence,
    input_yuv: Path,
    reconstructed_yuv: Path,
    bitstream_path: Path,
    qp: int,
    bit_depth: int,
    extra_arguments: list[str],
    progress: tqdm | None = None,
) -> str:
    rounded_fps = int(round(sequence.fps))
    if abs(sequence.fps - rounded_fps) > 1e-6:
        raise ValueError(
            f"HM requires an integer frame rate; {sequence.name} uses {sequence.fps}"
        )
    bitstream_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        encoder,
        "-c",
        str(config),
        f"--InputFile={input_yuv}",
        f"--BitstreamFile={bitstream_path}",
        f"--ReconFile={reconstructed_yuv}",
        f"--SourceWidth={sequence.width}",
        f"--SourceHeight={sequence.height}",
        f"--FrameRate={rounded_fps}",
        f"--FramesToBeEncoded={sequence.frame_count}",
        "--FrameSkip=0",
        f"--QP={qp}",
        f"--InputBitDepth={bit_depth}",
        f"--InternalBitDepth={bit_depth}",
        f"--OutputBitDepth={bit_depth}",
        "--InputChromaFormat=420",
        *extra_arguments,
    ]
    # ``subprocess.run`` only returns after HM has encoded the *entire*
    # sequence.  HM emits one "POC n" line per coded picture, so consume its
    # output incrementally and expose that information to the notebook.
    # This is especially useful for the CPU-only HM reference encoder, for
    # which a low-QP sequence can take many minutes.
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_lines: list[str] = []
    latest_poc = -1
    try:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            match = POC_PROGRESS_PATTERN.search(line)
            if match:
                latest_poc = int(match.group(1))
                # Updating every picture makes the progress visible without
                # flooding notebook output with HM's full log.
                if progress is not None:
                    progress.set_postfix_str(
                        f"{sequence.name}: HM {latest_poc + 1}/"
                        f"{sequence.frame_count} frames"
                    )
                    progress.refresh()
        return_code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise

    output = "".join(output_lines)
    if return_code:
        tail = "".join(output_lines[-40:])
        raise RuntimeError(
            f"HM encode for {sequence.name} at QP {qp} failed with exit code "
            f"{return_code}.\nCommand: {subprocess.list2cmdline(command)}\n{tail}"
        )
    if progress is not None:
        progress.set_postfix_str(
            f"{sequence.name}: HM completed ({sequence.frame_count} frames)"
        )
        progress.refresh()
    return output


def yuv_to_rgb_frames(
    ffmpeg: str,
    sequence: VideoSequence,
    reconstructed_yuv: Path,
    output_dir: Path,
    bit_depth: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pixel_format = "yuv420p" if bit_depth == 8 else "yuv420p10le"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        "-video_size",
        f"{sequence.width}x{sequence.height}",
        "-framerate",
        str(sequence.fps),
        "-i",
        str(reconstructed_yuv),
        "-frames:v",
        str(sequence.frame_count),
        "-vf",
        "scale=in_range=tv:out_range=pc:in_color_matrix=bt709",
        "-pix_fmt",
        "rgb24",
        "-start_number",
        "0",
        "-y",
        str(output_dir / "%08d.png"),
    ]
    run_command(command, f"FFmpeg YUV-to-RGB for {sequence.name}")
    frames = sorted(output_dir.glob("*.png"))
    if len(frames) != sequence.frame_count:
        raise RuntimeError(
            f"{sequence.name}: decoded {len(frames)} RGB frames; "
            f"expected {sequence.frame_count}"
        )
    return frames


def reported_picture_bits(encoder_log: str) -> dict[int, int]:
    result = {}
    for match in POC_BITS_PATTERN.finditer(encoder_log):
        poc = int(match.group(1))
        result[poc] = int(match.group(2).replace(",", ""))
    return result


def conditional_rate(
    bitstream_path: Path,
    encoder_log: str,
    protocol_key: str,
) -> tuple[int, int]:
    full_bits = bitstream_path.stat().st_size * 8
    if protocol_key == "all-frames":
        return full_bits, 0

    picture_bits = reported_picture_bits(encoder_log)
    if 0 not in picture_bits:
        raise RuntimeError(
            "Could not parse HM's reported POC 0 bits. Conditional-P-frame "
            "evaluation requires the standard per-picture HM log. Use "
            "--protocol all-frames only if complete-stream comparison is intended."
        )
    excluded_seed_bits = picture_bits[0]
    actual_bits = full_bits - excluded_seed_bits
    if actual_bits <= 0:
        raise RuntimeError(
            f"Invalid conditional rate: full stream={full_bits} bits, "
            f"POC0={excluded_seed_bits} bits"
        )
    return actual_bits, excluded_seed_bits


def as_yolo_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


@torch.inference_mode()
def evaluate_reconstructions(
    detector,
    evaluator: DetectionMAP,
    dataset: AnnotatedVideoDataset,
    sequence: VideoSequence,
    reconstructed_frames: list[Path],
    detector_size: int,
    first_image_id: int,
    protocol_key: str,
    progress_label: str,
    detector_batch_size: int,
) -> int:
    first_frame = 0 if protocol_key == "all-frames" else 1
    image_id = first_image_id
    if detector_batch_size < 1:
        raise ValueError("detector_batch_size must be positive")

    with tqdm(
        total=sequence.frame_count - first_frame,
        desc=progress_label,
        unit="frame",
        leave=False,
    ) as frame_progress:
        for batch_start in range(
            first_frame, sequence.frame_count, detector_batch_size
        ):
            batch_indices = list(
                range(
                    batch_start,
                    min(batch_start + detector_batch_size, sequence.frame_count),
                )
            )
            # AutoShape accepts a list of RGB arrays and performs independent
            # letterboxing/NMS per image. Batch inference changes throughput,
            # not the evaluated frames, labels, or codec rate.
            batch_detections = detector(
                [as_yolo_image(reconstructed_frames[index]) for index in batch_indices],
                size=detector_size,
            ).xyxy
            if len(batch_detections) != len(batch_indices):
                raise RuntimeError(
                    f"YOLO returned {len(batch_detections)} outputs for "
                    f"a batch of {len(batch_indices)} frames"
                )
            for frame_index, detections in zip(
                batch_indices, batch_detections, strict=True
            ):
                target_boxes, target_classes = dataset.load_ground_truth(
                    sequence.label_paths[frame_index],
                    sequence.width,
                    sequence.height,
                )
                if (
                    len(target_classes)
                    and int(target_classes.max()) >= len(detector.names)
                ):
                    raise ValueError(
                        f"{sequence.label_paths[frame_index]} contains class "
                        f"{int(target_classes.max())}, but the task model only "
                        f"defines {len(detector.names)} classes"
                    )
                detections = detections.detach().cpu()
                evaluator.add(
                    image_id=image_id,
                    predicted_boxes=detections[:, :4],
                    predicted_scores=detections[:, 4],
                    predicted_classes=detections[:, 5].long(),
                    target_boxes=target_boxes,
                    target_classes=target_classes,
                )
                image_id += 1
            frame_progress.update(len(batch_indices))
    return image_id


def aggregate_rate(records: list[dict]) -> dict[str, float | int]:
    total_bits = sum(record["actual_bits"] for record in records)
    total_pixels = sum(
        record["coded_frames"] * record["width"] * record["height"]
        for record in records
    )
    total_duration = sum(
        record["coded_frames"] / record["fps"] for record in records
    )
    return {
        "actual_bits": int(total_bits),
        "actual_bpp": float(total_bits / total_pixels),
        "kbps": float(total_bits / total_duration / 1000.0),
        "coded_frames": int(sum(record["coded_frames"] for record in records)),
    }


def evaluate_hevc(args: argparse.Namespace) -> None:
    if len(set(args.qps)) != RATE_POINT_COUNT:
        raise ValueError("Exactly four distinct HEVC QPs are required")
    if not torch.cuda.is_available():
        raise RuntimeError("YOLO mAP evaluation requires CUDA")

    encoder = resolve_executable(args.hm_encoder, "HM encoder")
    ffmpeg = resolve_executable(args.ffmpeg, "FFmpeg")
    config = Path(args.hm_config).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"HM configuration not found: {config}")

    dataset = AnnotatedVideoDataset(args.data_dir, args.dataset_manifest)
    sequences = list(dataset)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if not sequences:
        raise RuntimeError("No evaluation sequences were selected")
    for sequence in sequences:
        if sequence.width % 2 or sequence.height % 2:
            raise ValueError(
                f"{sequence.name} is {sequence.width}x{sequence.height}; "
                "YUV 4:2:0 requires even dimensions"
            )

    device = torch.device(f"cuda:{args.cuda_index}")
    detector = load_yolov5(
        args.task_model,
        repository=args.yolov5_repo,
        weights=args.yolov5_weights,
    ).to(device).eval()
    detector.conf = args.confidence_threshold
    detector.iou = args.nms_iou_threshold
    detector.max_det = args.max_detections

    method_name = safe_name(args.method_name)
    bitstream_root = Path(args.bitstream_dir) / method_name
    log_root = Path(args.encoder_log_dir) / method_name
    reconstruction_root = Path(args.reconstruction_dir) / method_name
    work_parent = Path(args.work_dir) if args.work_dir else None
    if work_parent is not None:
        work_parent.mkdir(parents=True, exist_ok=True)

    points = []
    for qp in args.qps:
        evaluator = DetectionMAP()
        records = []
        next_image_id = 0
        progress = tqdm(sequences, desc=f"{args.method_name}: HM QP {qp}")
        for sequence in progress:
            with tempfile.TemporaryDirectory(
                prefix=f"hevc_{safe_name(sequence.name)}_",
                dir=work_parent,
            ) as temporary_name:
                temporary = Path(temporary_name)
                concat_path = temporary / "frames.ffconcat"
                input_yuv = temporary / "input.yuv"
                reconstructed_yuv = temporary / "reconstructed.yuv"
                reconstructed_dir = temporary / "reconstructed"
                bitstream_path = (
                    bitstream_root
                    / f"qp_{qp:02d}"
                    / f"{safe_name(sequence.name)}.bin"
                )

                write_concat_file(sequence, concat_path)
                progress.set_postfix_str(f"{sequence.name}: RGB -> YUV")
                progress.refresh()
                rgb_to_yuv(
                    ffmpeg,
                    sequence,
                    concat_path,
                    input_yuv,
                    args.bit_depth,
                )
                encoder_log = hm_encode(
                    encoder,
                    config,
                    sequence,
                    input_yuv,
                    reconstructed_yuv,
                    bitstream_path,
                    qp,
                    args.bit_depth,
                    args.hm_extra_arg,
                    progress,
                )
                log_path = (
                    log_root
                    / f"qp_{qp:02d}"
                    / f"{safe_name(sequence.name)}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(encoder_log, encoding="utf-8")

                progress.set_postfix_str(f"{sequence.name}: YUV -> RGB")
                progress.refresh()
                reconstructed_frames = yuv_to_rgb_frames(
                    ffmpeg,
                    sequence,
                    reconstructed_yuv,
                    reconstructed_dir,
                    args.bit_depth,
                )
                next_image_id = evaluate_reconstructions(
                    detector,
                    evaluator,
                    dataset,
                    sequence,
                    reconstructed_frames,
                    args.detector_size,
                    next_image_id,
                    args.protocol,
                    f"{args.method_name}: QP {qp} YOLO {sequence.name}",
                    args.detector_batch_size,
                )
                if args.save_reconstructions:
                    destination = (
                        reconstruction_root
                        / f"qp_{qp:02d}"
                        / safe_name(sequence.name)
                    )
                    shutil.copytree(
                        reconstructed_dir,
                        destination,
                        dirs_exist_ok=True,
                    )

                actual_bits, excluded_seed_bits = conditional_rate(
                    bitstream_path,
                    encoder_log,
                    args.protocol,
                )
                coded_frames = (
                    sequence.frame_count
                    if args.protocol == "all-frames"
                    else sequence.frame_count - 1
                )
                records.append(
                    {
                        "name": sequence.name,
                        "bitstream_file": str(bitstream_path),
                        "full_bitstream_bits": bitstream_path.stat().st_size * 8,
                        "excluded_seed_bits": excluded_seed_bits,
                        "actual_bits": actual_bits,
                        "actual_bpp": actual_bits
                        / (coded_frames * sequence.width * sequence.height),
                        "kbps": actual_bits
                        * sequence.fps
                        / (1000.0 * coded_frames),
                        "fps": sequence.fps,
                        "width": sequence.width,
                        "height": sequence.height,
                        "coded_frames": coded_frames,
                    }
                )

        points.append(
            {
                "base_qp": qp,
                **aggregate_rate(records),
                **evaluator.compute(),
                "sequences": records,
            }
        )

    protocol = PROTOCOLS[args.protocol]
    rate_source = (
        "complete HM bitstream bytes including headers"
        if args.protocol == "all-frames"
        else (
            "complete HM bitstream bytes minus HM-reported POC0 picture bits; "
            "sequence headers retained"
        )
    )
    output = {
        "schema_version": 4,
        "method": args.method_name,
        "codec": "HEVC HM reference encoder",
        "codec_config": {
            "encoder": encoder,
            "configuration_file": str(config),
            "configuration_name": args.configuration_name,
            "qps": list(args.qps),
            "bit_depth": args.bit_depth,
            "chroma_format": "4:2:0",
            "color_conversion": "BT.709 RGB full <-> YUV limited",
            "extra_arguments": list(args.hm_extra_arg),
        },
        "protocol": protocol,
        "rate_source": rate_source,
        "rate_points": RATE_POINT_COUNT,
        "task": "object_detection",
        "task_model": args.task_model,
        "ground_truth": "normalized YOLO labels from evaluation manifest",
        "evaluation_id": evaluation_id(dataset, sequences),
        "dataset": dataset_summary(dataset, sequences),
        "detector_config": detector_config(
            args.task_model,
            args.detector_size,
            args.confidence_threshold,
            args.nms_iou_threshold,
            args.max_detections,
            args.yolov5_weights,
        ),
        "points": points,
    }
    output_path = Path(args.output_dir) / f"{method_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved HEVC actual-rate mAP results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--hm-encoder", required=True)
    parser.add_argument("--hm-config", required=True)
    parser.add_argument(
        "--configuration-name",
        default="Low-Delay P",
        help="Descriptive name recorded in the result JSON",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOLS),
        default="conditional-pframes",
    )
    parser.add_argument(
        "--qps",
        type=int,
        nargs=RATE_POINT_COUNT,
        default=(22, 27, 32, 37),
        metavar=("QP1", "QP2", "QP3", "QP4"),
    )
    parser.add_argument("--bit-depth", type=int, choices=(8, 10), default=8)
    parser.add_argument(
        "--hm-extra-arg",
        action="append",
        default=[],
        help="Repeat for each HM override, e.g. --hm-extra-arg=--IntraPeriod=-1",
    )
    parser.add_argument("--method-name", default="HEVC HM Low-Delay P")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--task-model", default="yolov5s")
    parser.add_argument("--yolov5-repo")
    parser.add_argument("--yolov5-weights")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument(
        "--detector-batch-size",
        type=int,
        default=16,
        help="YOLO inference frames per GPU batch; changes speed only (default: 16)",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--bitstream-dir", default="output/hevc_bitstreams")
    parser.add_argument("--encoder-log-dir", default="output/hevc_logs")
    parser.add_argument("--output-dir", default="output/hevc_evaluation")
    parser.add_argument("--work-dir")
    parser.add_argument("--save-reconstructions", action="store_true")
    parser.add_argument(
        "--reconstruction-dir",
        default="output/hevc_reconstructions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_hevc(parse_args())
