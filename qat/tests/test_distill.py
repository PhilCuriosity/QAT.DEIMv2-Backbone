"""Unit tests for the distillation losses."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qat.distill import memory_loss, output_distill_loss  # noqa: E402


def test_memory_loss_identical_is_zero():
    x = torch.randn(2, 2000, 128)
    assert float(memory_loss(x, x)) < 1e-6


def test_output_distill_loss_identical_is_zero():
    logits = torch.randn(2, 300, 80)
    boxes = torch.rand(2, 300, 4).clamp(0.01, 0.99)
    l_logits, l_boxes, l_giou = output_distill_loss(
        logits, boxes, logits, boxes, topk=50
    )
    assert float(l_logits) < 1e-6
    assert float(l_boxes) < 1e-6
    assert float(l_giou) < 1e-6


def test_output_distill_loss_handles_query_permutation():
    torch.manual_seed(0)
    logits = torch.randn(1, 300, 80)
    boxes = torch.rand(1, 300, 4).clamp(0.01, 0.99)
    perm = torch.randperm(300)
    l_logits, l_boxes, l_giou = output_distill_loss(
        logits,
        boxes,
        logits[:, perm],
        boxes[:, perm],
        topk=100,
    )
    assert torch.isfinite(l_logits) and torch.isfinite(l_boxes) and torch.isfinite(l_giou)
    assert float(l_boxes) < 1e-4
