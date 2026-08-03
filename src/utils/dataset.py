"""Dataset loader for contiguous frame folders.

The DCVC-RT hierarchical training schedule uses groups of eight pictures.
Standard Vimeo-90K septuplets only contain seven pictures, so exact training
requires the longer Vimeo clips described by DCVC-FM/DCVC-RT (or another
frame-folder dataset with at least eight contiguous frames per sequence).
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _natural_key(path: Path) -> list[tuple[int, int | str]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    ]


class VideoSequenceDataset(Dataset):
    """Return a consistently transformed contiguous clip from each directory.

    ``root_dir`` contains sequence directories. A list entry is a sequence
    path relative to ``root_dir``. If the list is omitted or cannot be found,
    sequence directories are discovered recursively.
    """

    def __init__(
        self,
        root_dir: str | Path,
        list_file: str | Path | None = None,
        crop_size: int = 256,
        num_frames: int = 8,
        training: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.crop_size = int(crop_size)
        self.num_frames = int(num_frames)
        self.training = bool(training)

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Video sequences directory not found: {self.root_dir}")
        if self.num_frames < 2:
            raise ValueError("num_frames must contain one reference seed and at least one P-frame")

        list_path = self._resolve_list_path(list_file)
        if list_path is not None:
            self.sequence_ids = [
                line.strip()
                for line in list_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            self.sequence_ids = self._discover_sequences()

        if not self.sequence_ids:
            raise RuntimeError(f"No frame sequences found under {self.root_dir}")

        first_sequence = self.root_dir / self.sequence_ids[0]
        first_count = len(self._frame_paths(first_sequence))
        if first_count < self.num_frames:
            raise ValueError(
                f"{first_sequence} contains {first_count} frames, but the clip requires "
                f"{self.num_frames}. Standard Vimeo-90K septuplets have only 7 frames; "
                "use processed long Vimeo clips for the 8-picture DCVC-RT schedule."
            )

    def _resolve_list_path(self, list_file: str | Path | None) -> Path | None:
        if list_file is None:
            return None
        candidate = Path(list_file)
        candidates = (candidate, self.root_dir / candidate, self.root_dir.parent / candidate)
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"Sequence list not found: {list_file}")

    def _discover_sequences(self) -> list[str]:
        directories = {
            frame.parent
            for frame in self.root_dir.rglob("*")
            if frame.is_file() and frame.suffix.lower() in IMAGE_EXTENSIONS
        }
        return sorted(str(path.relative_to(self.root_dir)) for path in directories)

    @staticmethod
    def _frame_paths(sequence_dir: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in sequence_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=_natural_key,
        )

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        sequence_dir = self.root_dir / self.sequence_ids[index]
        frame_paths = self._frame_paths(sequence_dir)
        if len(frame_paths) < self.num_frames:
            raise ValueError(
                f"{sequence_dir} contains {len(frame_paths)} frames; "
                f"{self.num_frames} contiguous frames are required"
            )
        max_start = len(frame_paths) - self.num_frames
        first_frame = random.randint(0, max_start) if self.training else 0
        images = [
            self._load_rgb(path)
            for path in frame_paths[first_frame:first_frame + self.num_frames]
        ]

        width, height = images[0].size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(
                f"Crop size {self.crop_size} exceeds frame size {width}x{height} "
                f"in {sequence_dir}"
            )

        if self.training:
            top, left, crop_height, crop_width = RandomCrop.get_params(
                images[0], output_size=(self.crop_size, self.crop_size)
            )
            if random.random() < 0.5:
                images = [transforms.hflip(image) for image in images]
        else:
            crop_height = crop_width = self.crop_size
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2

        return torch.stack(
            [
                transforms.to_tensor(
                    transforms.crop(image, top, left, crop_height, crop_width)
                )
                for image in images
            ]
        )


# Kept as a compatibility import for evaluation scripts and older checkpoints.
VimeoSeptupletDataset = VideoSequenceDataset
