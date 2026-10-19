"""3D U-Net model and segmentation metrics for MSD Task09 Spleen."""

from __future__ import annotations

import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet


def create_spleen_unet() -> UNet:
    """Create a compact 3D U-Net suitable for a 6 GB GPU pilot."""
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )


def dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """Compute binary Dice score for a single-channel prediction batch."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction shape {tuple(predictions.shape)} does not match "
            f"target shape {tuple(targets.shape)}."
        )

    binary_predictions = (torch.sigmoid(predictions) >= threshold).float()
    targets = targets.float()

    intersection = (binary_predictions * targets).sum()
    denominator = binary_predictions.sum() + targets.sum()

    if denominator == 0:
        return 1.0

    return float((2.0 * intersection + epsilon) / (denominator + epsilon))