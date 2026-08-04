"""Frozen YOLOv5 feature extractor used by the VCM objective."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn


DEFAULT_FEATURE_LAYER_INDICES = (4, 6, 9)
LAST_SEQUENTIAL_BACKBONE_LAYER = 9


@contextmanager
def _allow_legacy_yolov5_checkpoint_loading():
    """Temporarily support YOLOv5 v7 checkpoints with PyTorch >= 2.6."""
    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load
    try:
        yield
    finally:
        torch.load = original_load


def load_yolov5(
    model_name: str = "yolov5s",
    repository: str | Path | None = None,
    weights: str | Path | None = None,
):
    """Load pinned YOLOv5 v7 online or from Kaggle/local files.

    When ``repository`` is provided, it must point to a local YOLOv5 v7
    checkout. ``weights`` may point to a local ``.pt`` file in either mode.
    """
    repository_path = Path(repository).expanduser() if repository else None
    weights_path = Path(weights).expanduser() if weights else None
    if repository_path is not None and not repository_path.is_dir():
        raise FileNotFoundError(f"YOLOv5 repository not found: {repository_path}")
    if weights_path is not None and not weights_path.is_file():
        raise FileNotFoundError(f"YOLOv5 weights not found: {weights_path}")

    source = "local" if repository_path is not None else "github"
    repo_or_dir = (
        str(repository_path)
        if repository_path is not None
        else "ultralytics/yolov5:v7.0"
    )
    entrypoint = "custom" if weights_path is not None else model_name
    load_kwargs = {
        "source": source,
        "trust_repo": True,
    }
    if weights_path is not None:
        load_kwargs["path"] = str(weights_path)
    else:
        load_kwargs["pretrained"] = True

    with _allow_legacy_yolov5_checkpoint_loading():
        return torch.hub.load(
            repo_or_dir,
            entrypoint,
            **load_kwargs,
        )


class YOLOv5FeatureExtractor(nn.Module):
    """Frozen multi-scale backbone features from pretrained YOLOv5.

    The module parameters are frozen, but its forward pass remains
    differentiable with respect to the input. This lets the VCM loss update
    DCVC-RT while measuring distortion in a stable task-feature space.

    YOLOv5 layers 4, 6 and 9 are the default stride-8, stride-16 and stride-32
    backbone stages. They provide detail, middle-level structure and deep
    semantics without retaining the much larger detection-neck graph for every
    frame in a training GOP.
    """

    def __init__(
        self,
        model_name: str = "yolov5s",
        feature_layer_indices: Sequence[int] = DEFAULT_FEATURE_LAYER_INDICES,
        repository: str | Path | None = None,
        weights: str | Path | None = None,
    ):
        super().__init__()
        indices = tuple(int(index) for index in feature_layer_indices)
        if not indices:
            raise ValueError("feature_layer_indices must contain at least one layer")
        if any(index < 0 for index in indices):
            raise ValueError("feature layer indices must be non-negative")
        if len(set(indices)) != len(indices):
            raise ValueError("feature layer indices must be unique")
        if tuple(sorted(indices)) != indices:
            raise ValueError("feature layer indices must be in ascending order")
        if indices[-1] > LAST_SEQUENTIAL_BACKBONE_LAYER:
            raise ValueError(
                "multi-level VCM features are limited to YOLOv5 backbone layers "
                f"0..{LAST_SEQUENTIAL_BACKBONE_LAYER}; later neck layers require "
                "skip connections and retain substantially more training memory"
            )

        yolo = load_yolov5(model_name, repository=repository, weights=weights)
        layers = list(yolo.model.model.model.children())
        if indices[-1] >= len(layers):
            raise ValueError(
                f"feature layer index {indices[-1]} exceeds YOLOv5's "
                f"last layer index ({len(layers) - 1})"
            )

        self.feature_layer_indices = indices
        self._selected_layer_indices = frozenset(indices)
        self.backbone_prefix = nn.ModuleList(layers[: indices[-1] + 1])
        for parameter in self.backbone_prefix.parameters():
            parameter.requires_grad_(False)
        self.backbone_prefix.eval()

    def train(self, mode: bool = True):
        """Keep BatchNorm and other stateful layers fixed during codec training."""
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = {}
        output = images
        for layer_index, layer in enumerate(self.backbone_prefix):
            output = layer(output)
            if layer_index in self._selected_layer_indices:
                features[layer_index] = output
        return tuple(features[index] for index in self.feature_layer_indices)
