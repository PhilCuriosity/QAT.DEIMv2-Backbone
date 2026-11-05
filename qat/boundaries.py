"""Model split at the decoder input boundary.

QAT graph:
    images -> backbone -> encoder -> decoder._get_encoder_input -> memory

The decoder from ``_get_decoder_input`` onward stays FP32 and frozen.
"""
from __future__ import annotations

import torch
from torch import nn


class BackboneEncoderQAT(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        feats = self.encoder(feats)
        memory, _ = self.decoder.encode(feats)
        return memory

    def trainable_parameters(self):
        for module in (self.backbone, self.encoder, self.decoder.input_proj):
            yield from module.parameters()


class FrozenDecoder(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

    def forward(self, memory, spatial_shapes, targets=None):
        return self.decoder.decode(memory, spatial_shapes, targets)


def spatial_shapes_for(images: torch.Tensor, feat_strides) -> list[list[int]]:
    height, width = images.shape[-2:]
    return [[int(height) // int(stride), int(width) // int(stride)] for stride in feat_strides]
