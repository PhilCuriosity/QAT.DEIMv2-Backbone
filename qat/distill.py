"""Distillation losses for backbone+encoder QAT.

Output distillation uses Hungarian matching because the encoder TopK can
permute queries; index-aligned logits/boxes losses would corrupt the student.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, box_iou

from engine.deim.box_ops import generalized_box_iou


def _safe_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    xyxy = box_convert(boxes, "cxcywh", "xyxy").clamp(min=0.0, max=1.0)
    lower = torch.minimum(xyxy[..., :2], xyxy[..., 2:])
    upper = torch.maximum(xyxy[..., :2], xyxy[..., 2:])
    return torch.cat([lower, upper], dim=-1)


def memory_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student = student.flatten(1)
    teacher = teacher.flatten(1)
    return (1.0 - F.cosine_similarity(student, teacher, dim=1)).mean()


def output_distill_loss(
    student_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_boxes: torch.Tensor,
    *,
    topk: int | None = None,
    temperature: float = 1.0,
):
    """Hungarian-matched logits/box/GIoU distillation.

    Returns ``(logits_loss, boxes_loss, giou_loss)``.
    """
    batch = student_logits.shape[0]
    if topk is not None and topk < teacher_logits.shape[1]:
        scores = teacher_logits.sigmoid().max(dim=-1).values
        indices = scores.topk(topk, dim=1).indices
        teacher_boxes = teacher_boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        teacher_logits = teacher_logits.gather(
            1, indices[..., None].expand(-1, -1, teacher_logits.shape[-1])
        )

    loss_logits = student_logits.new_zeros(())
    loss_boxes = student_boxes.new_zeros(())
    loss_giou = student_boxes.new_zeros(())
    for b in range(batch):
        sb = student_boxes[b]
        tb = teacher_boxes[b]
        cost = torch.cdist(sb, tb, p=1)
        iou = box_iou(_safe_xyxy(sb), _safe_xyxy(tb))
        cost = cost + (1.0 - iou)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        rows_t = torch.as_tensor(rows, device=sb.device)
        cols_t = torch.as_tensor(cols, device=sb.device)
        matched_student = sb[rows_t]
        matched_teacher = tb[cols_t]
        loss_boxes = loss_boxes + F.l1_loss(matched_student, matched_teacher)
        loss_giou = loss_giou + (
            1.0
            - torch.diag(
                generalized_box_iou(
                    _safe_xyxy(matched_student),
                    _safe_xyxy(matched_teacher),
                )
            )
        ).mean()
        loss_logits = loss_logits + F.mse_loss(
            (student_logits[b, rows_t] / temperature).sigmoid(),
            (teacher_logits[b, cols_t] / temperature).sigmoid(),
        )
    return loss_logits / batch, loss_boxes / batch, loss_giou / batch
