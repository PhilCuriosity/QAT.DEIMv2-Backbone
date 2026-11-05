"""Export the backbone+encoder subgraph for PT2E."""
from __future__ import annotations

import torch
from torch.export import Dim

from qat.boundaries import BackboneEncoderQAT
from qat.compat import install_export_compat


def _set_explicit_attention(model: torch.nn.Module) -> list[tuple[torch.nn.Module, bool]]:
    """Enable the explicit attention path for tracing; return previous flags."""
    previous = []
    for module in model.modules():
        if hasattr(module, "explicit_attention"):
            previous.append((module, bool(module.explicit_attention)))
            module.explicit_attention = True
    return previous


def export_backbone_encoder(
    model: torch.nn.Module,
    example_images: torch.Tensor,
    *,
    dynamic_batch_max: int = 64,
) -> torch.fx.GraphModule:
    install_export_compat()
    wrapper = BackboneEncoderQAT(model).eval()
    precompute = getattr(getattr(model, "backbone", None), "precompute_rope", None)
    if precompute is not None:
        precompute((example_images.shape[-2], example_images.shape[-1]))
    dynamic_shapes = None
    if dynamic_batch_max is not None and int(dynamic_batch_max) > 1:
        if example_images.shape[0] < 2:
            example_images = example_images.repeat(2, 1, 1, 1)
        dynamic_shapes = {
            "images": {0: Dim("batch", min=1, max=max(int(dynamic_batch_max), 2))}
        }
    previous = _set_explicit_attention(model)
    try:
        exported = torch.export.export_for_training(
            wrapper, (example_images,), dynamic_shapes=dynamic_shapes
        )
    finally:
        for module, value in previous:
            module.explicit_attention = value
    return exported.module()
