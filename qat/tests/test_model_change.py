"""Regression tests for backbone+encoder QAT and regional discovery.

Run with:
    python -m pytest \
        qat/tests/test_model_change.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from qat.boundaries import FrozenDecoder, spatial_shapes_for  # noqa: E402
from qat.export import export_backbone_encoder  # noqa: E402
from qat.pt2e import prepare_model  # noqa: E402
from qat.regional import (  # noqa: E402
    build_regional_config,
    discover_regions,
    validate_regions,
)

CONFIG = REPO_ROOT / "configs/deimv2/deimv2_hgnetv2_n_coco.yml"
QUANT_CONFIG = REPO_ROOT / "qat/configs/quant/w8a16.json"


class ExtraEncoder(nn.Module):
    """Simulate a user adding a block after the encoder."""

    def __init__(self, base: nn.Module, channels: list[int]):
        super().__init__()
        self.base = base
        self.extra = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channel, channel, 1, bias=False),
                nn.BatchNorm2d(channel),
                nn.SiLU(),
            )
            for channel in channels
        )

    def forward(self, feats):
        feats = self.base(feats)
        return [block(feat) for block, feat in zip(self.extra, feats)]


def build_model(*, modified: bool = False) -> nn.Module:
    cfg = YAMLConfig(str(CONFIG))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    if modified:
        channels = [model.encoder.hidden_dim] * len(model.encoder.in_channels)
        model.encoder = ExtraEncoder(model.encoder, channels)
    return model.eval()


def test_regional_discovery_tracks_structure_change(tmp_path):
    base = export_backbone_encoder(build_model(), torch.randn(2, 3, 640, 640))
    base_regions = discover_regions(base)

    modified = export_backbone_encoder(
        build_model(modified=True), torch.randn(2, 3, 640, 640)
    )
    regions = discover_regions(modified)
    missing = validate_regions(regions, modified)
    assert not missing

    base_conv = len(base_regions["conv"])
    changed_conv = len(regions["conv"])
    assert changed_conv > base_conv, (base_conv, changed_conv)

    extra_nodes = [
        item for item in regions["conv"] if "encoder.extra" in item["module"]
    ]
    assert len(extra_nodes) == 2, extra_nodes

    output = tmp_path / "w8a16_modified.json"
    config = build_regional_config(str(QUANT_CONFIG), regions)
    output.write_text(json.dumps(config))

    from utils.ax_quantizer import ax_load_config

    _, regional = ax_load_config(str(output), is_qat=True)
    discovered_names = {
        name for entry in regional for name in (entry.module_names or [])
    }
    # Every new block must be discoverable in the graph; only role-selected
    # convs (first/last, depthwise) are written into regional_configs.
    assert {item["name"] for item in extra_nodes} <= {
        item["name"] for item in regions["conv"]
    }
    assert all(name in {node.name for node in modified.graph.nodes} for name in discovered_names)


def test_prepare_and_train_step_with_modified_encoder():
    model = build_model(modified=True)
    prepared = prepare_model(
        model,
        torch.randn(2, 3, 640, 640),
        str(QUANT_CONFIG),
        device="cpu",
    )
    prepared.train()
    torch.ao.quantization.enable_fake_quant(prepared)
    torch.ao.quantization.enable_observer(prepared)

    model.decoder.num_denoising = 0
    decoder = FrozenDecoder(model.decoder).train()
    images = torch.rand(2, 3, 640, 640)
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
        for _ in range(2)
    ]

    memory = prepared(images)
    assert memory.shape == (2, 2000, 128)
    assert torch.isfinite(memory).all()

    shapes = spatial_shapes_for(images, model.decoder.feat_strides)
    criterion = YAMLConfig(str(CONFIG)).criterion
    outputs = decoder(memory, shapes, targets)
    loss = sum(criterion(outputs, targets, epoch=0).values())
    assert torch.isfinite(loss)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in prepared.parameters() if parameter.requires_grad],
        lr=1e-6,
    )
    loss.backward()
    optimizer.step()


@pytest.mark.parametrize("modified", [False, True])
def test_discovery_report_contains_roles(modified):
    gm = export_backbone_encoder(
        build_model(modified=modified), torch.randn(1, 3, 640, 640)
    )
    regions = discover_regions(gm)
    assert regions["conv"]
    assert regions["softmax"]
    assert regions["first_last_conv"]
    assert validate_regions(regions, gm) == []
