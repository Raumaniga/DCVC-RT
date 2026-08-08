"""Machine-only rate-distortion objective for DCVC-RT."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .yolov5_extractor import DEFAULT_FEATURE_LAYER_INDICES, YOLOv5FeatureExtractor


class VCMLoss(nn.Module):
    """Rate plus YOLO task-feature distortion.

    The teacher extracts targets from uncompressed frames under ``no_grad``.
    The reconstruction extractor is a cloned YOLO front end and is jointly
    optimized with DMC by default.  Its BatchNorm statistics remain frozen and
    the original teacher remains completely frozen.
    """

    def __init__(
        self,
        model_name: str = "yolov5s",
        feature_layer_indices: Sequence[int] = DEFAULT_FEATURE_LAYER_INDICES,
        feature_layer_weights: Sequence[float] | None = None,
        yolov5_repository: str | Path | None = None,
        yolov5_weights: str | Path | None = None,
        train_cloned_frontend: bool = True,
    ):
        super().__init__()
        self.feature_layer_indices = tuple(int(index) for index in feature_layer_indices)
        if feature_layer_weights is None:
            feature_layer_weights = (1.0,) * len(self.feature_layer_indices)
        weights = tuple(float(weight) for weight in feature_layer_weights)
        if len(weights) != len(self.feature_layer_indices):
            raise ValueError(
                "feature_layer_weights and feature_layer_indices must have the same length"
            )
        if any(weight <= 0 for weight in weights):
            raise ValueError("all feature layer weights must be positive")
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("layer_weights", weight_tensor / weight_tensor.sum())
        self.layer_metric_names = tuple(
            f"feature_mse_l{index}" for index in self.feature_layer_indices
        )

        self.teacher_extractor = YOLOv5FeatureExtractor(
            model_name,
            self.feature_layer_indices,
            repository=yolov5_repository,
            weights=yolov5_weights,
            trainable=False,
        )
        self.reconstruction_extractor = copy.deepcopy(self.teacher_extractor)
        self.train_cloned_frontend = bool(train_cloned_frontend)
        self.reconstruction_extractor.set_trainable(self.train_cloned_frontend)

    @property
    def last_backbone_layer(self) -> int:
        return self.reconstruction_extractor.last_backbone_layer

    def cloned_frontend_state_dict(self) -> dict[str, torch.Tensor]:
        return self.reconstruction_extractor.backbone_prefix.state_dict()

    def load_cloned_frontend_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.reconstruction_extractor.backbone_prefix.load_state_dict(
            state_dict,
            strict=True,
        )

    def cloned_frontend_named_parameters(self):
        if not self.train_cloned_frontend:
            return iter(())
        return self.reconstruction_extractor.backbone_prefix.named_parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_extractor.eval()
        # Trainable weights still receive gradients in eval mode.  Keeping this
        # prefix in eval prevents tiny video batches from corrupting BN state.
        self.reconstruction_extractor.eval()
        return self

    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        estimated_bpp: torch.Tensor,
        lambda_rd: float,
        distortion_weight: float = 1.0,
        return_details: bool = False,
    ):
        if lambda_rd <= 0:
            raise ValueError("lambda_rd must be positive")
        if distortion_weight <= 0:
            raise ValueError("distortion_weight must be positive")

        with torch.no_grad():
            teacher_features = self.teacher_extractor(original)
        reconstructed_features = self.reconstruction_extractor(reconstructed)
        if len(teacher_features) != len(reconstructed_features):
            raise RuntimeError("teacher and reconstruction feature pyramids do not match")

        layer_losses = torch.stack(
            [
                F.mse_loss(reconstructed_feature, teacher_feature)
                for reconstructed_feature, teacher_feature in zip(
                    reconstructed_features,
                    teacher_features,
                    strict=True,
                )
            ]
        )
        feature_mse = torch.sum(
            layer_losses * self.layer_weights.to(dtype=layer_losses.dtype)
        )
        weighted_feature_mse = (
            float(lambda_rd) * float(distortion_weight) * feature_mse
        )
        total_loss = estimated_bpp + weighted_feature_mse

        if not return_details:
            return total_loss

        details = {
            "total_loss": total_loss,
            "estimated_bpp": estimated_bpp,
            "feature_mse": feature_mse,
            "weighted_feature_mse": weighted_feature_mse,
            "lambda_rd": total_loss.new_tensor(float(lambda_rd)),
            "distortion_weight": total_loss.new_tensor(float(distortion_weight)),
        }
        details.update(
            {
                metric_name: layer_loss
                for metric_name, layer_loss in zip(
                    self.layer_metric_names,
                    layer_losses,
                    strict=True,
                )
            }
        )
        return details
