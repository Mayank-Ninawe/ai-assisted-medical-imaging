"""Tests for the 3D spleen U-Net and Dice metric."""

import pytest
import torch

from ml.segmentation.model import create_spleen_unet, dice_score


def test_create_spleen_unet_output_shape():
    model = create_spleen_unet()
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 1, 32, 32, 32))

    assert output.shape == (1, 1, 32, 32, 32)
    assert torch.isfinite(output).all()


def test_dice_score_perfect_prediction():
    targets = torch.zeros(1, 1, 8, 8, 8)
    targets[:, :, 2:5, 2:5, 2:5] = 1.0

    logits = torch.where(
        targets.bool(),
        torch.tensor(10.0),
        torch.tensor(-10.0),
    )

    assert dice_score(logits, targets) == 1.0


def test_dice_score_empty_prediction_and_target():
    logits = torch.full((1, 1, 8, 8, 8), -10.0)
    targets = torch.zeros(1, 1, 8, 8, 8)

    assert dice_score(logits, targets) == 1.0


def test_dice_score_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        dice_score(
            torch.randn(1, 1, 8, 8, 8),
            torch.zeros(1, 1, 8, 8, 4),
        )