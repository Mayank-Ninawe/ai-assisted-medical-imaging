"""Tests for segmentation evaluation reporting."""

import torch

from ml.segmentation.model import dice_score


def test_dice_score_handles_nonempty_prediction_and_target():
    targets = torch.zeros(1, 1, 8, 8, 8)
    targets[:, :, 2:5, 2:5, 2:5] = 1.0

    logits = targets * 10.0 - 5.0

    score = dice_score(logits, targets)

    assert 0.0 <= score <= 1.0
    assert score > 0.9