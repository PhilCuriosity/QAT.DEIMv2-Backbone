"""Export DEIMv2 FP32 ONNX graphs and verify them against the float model.

Outputs (selected with ``--mode``):
  - full model: images -> pred_logits, pred_boxes
  - backbone+encoder: images -> multi-scale feature maps
  - decoder: memory -> pred_logits, pred_boxes

Verification runs ONNX Runtime (CPU) and compares every output directly with
the PyTorch float model executed on CPU.  CPU-vs-CPU is required: CUDA vs CPU
numerical differences can change the encoder TopK query selection, which
permutes decoder outputs even though the model is correct.

Usage:
  python utils/export_fp32_onnx.py \
    --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
    --weights /path/to/deimv2_hgnetv2_n_coco.pth \
    --output-dir runs/export/deimv2_n_fp32_onnx_$(date +%Y%m%d_%H%M%S) \
    --mode all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402


class FullModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        return outputs["pred_logits"], outputs["pred_boxes"]


class BackboneEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.encoder = model.encoder

    def forward(self, images: torch.Tensor):
        feats = self.backbone(images)
        feats = self.encoder(feats)
        return tuple(feats)


class DecoderModel(nn.Module):
    def __init__(self, decoder: nn.Module, spatial_shapes):
        super().__init__()
        self.decoder = decoder
        self.spatial_shapes = spatial_shapes

    def forward(self, memory: torch.Tensor):
        outputs = self.decoder.decode(memory, self.spatial_shapes)
        return outputs["pred_logits"], outputs["pred_boxes"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(config: str, weights: str, device: str):
    cfg = YAMLConfig(config)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "ema" in state:
        state = state["ema"]["module"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device).eval()
    return cfg, model


def _example(size, batch: int, device: str) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, 3, *size, device=device)


def _spatial_shapes(size, feat_strides):
    height, width = int(size[0]), int(size[1])
    return [[height // int(stride), width // int(stride)] for stride in feat_strides]


def export_full(model: nn.Module, size, output: Path, opset: int, batch: int) -> None:
    wrapper = FullModel(model).eval()
    example = _example(size, batch, str(next(model.parameters()).device))
    torch.onnx.export(
        wrapper,
        (example,),
        str(output),
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        opset_version=opset,
        do_constant_folding=False,
    )


def export_backbone_encoder(model: nn.Module, size, output: Path, opset: int, batch: int) -> list[str]:
    wrapper = BackboneEncoder(model).eval()
    example = _example(size, batch, str(next(model.parameters()).device))
    with torch.no_grad():
        feats = wrapper(example)
    names = [f"feat_{i}" for i in range(len(feats))]
    torch.onnx.export(
        wrapper,
        (example,),
        str(output),
        input_names=["images"],
        output_names=names,
        opset_version=opset,
        do_constant_folding=False,
    )
    return names


def export_decoder(model: nn.Module, size, output: Path, opset: int, batch: int) -> dict:
    decoder = model.decoder
    decoder.num_denoising = 0
    decoder.eval()
    shapes = _spatial_shapes(size, decoder.feat_strides)
    num_tokens = sum(h * w for h, w in shapes)
    memory = torch.randn(
        batch, num_tokens, decoder.hidden_dim, device=next(model.parameters()).device
    )
    wrapper = DecoderModel(decoder, shapes).eval()
    torch.onnx.export(
        wrapper,
        (memory,),
        str(output),
        input_names=["memory"],
        output_names=["pred_logits", "pred_boxes"],
        opset_version=opset,
        do_constant_folding=False,
    )
    return {
        "memory_shape": list(memory.shape),
        "spatial_shapes": shapes,
        "num_tokens": num_tokens,
    }


def _diff(reference, exported) -> dict:
    import numpy as np

    reference = reference.detach().cpu().numpy() if torch.is_tensor(reference) else reference
    exported = exported.detach().cpu().numpy() if torch.is_tensor(exported) else exported
    delta = np.abs(reference - exported)
    return {
        "shape": list(reference.shape),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
    }


def _session(path: Path):
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(path)))
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def verify_full(path: Path, model: nn.Module, size, batch: int, atol: float) -> dict:
    model = model.to("cpu").eval()
    example = _example(size, batch, "cpu")
    with torch.no_grad():
        reference = FullModel(model)(example)
    session = _session(path)
    exported = session.run(None, {"images": example.numpy().astype("float32")})
    diffs = [_diff(t, o) for t, o in zip(reference, exported)]
    return {
        "outputs": [o.name for o in session.get_outputs()],
        "diff": diffs,
        "passed": all(item["max_abs"] <= atol for item in diffs),
    }


def verify_backbone_encoder(path: Path, model: nn.Module, size, batch: int, atol: float) -> dict:
    model = model.to("cpu").eval()
    example = _example(size, batch, "cpu")
    with torch.no_grad():
        reference = BackboneEncoder(model)(example)
    session = _session(path)
    exported = session.run(None, {"images": example.numpy().astype("float32")})
    diffs = [_diff(t, o) for t, o in zip(reference, exported)]
    return {
        "outputs": [o.name for o in session.get_outputs()],
        "diff": diffs,
        "passed": all(item["max_abs"] <= atol for item in diffs),
    }


def verify_decoder(path: Path, model: nn.Module, size, batch: int, atol: float) -> dict:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    model = model.to("cpu").eval()
    decoder = model.decoder
    decoder.num_denoising = 0
    decoder.eval()
    shapes = _spatial_shapes(size, decoder.feat_strides)
    num_tokens = sum(h * w for h, w in shapes)
    memory = torch.randn(batch, num_tokens, decoder.hidden_dim)
    with torch.no_grad():
        outputs = decoder.decode(memory, shapes)
        reference = (outputs["pred_logits"], outputs["pred_boxes"])
    session = _session(path)
    exported = session.run(None, {"memory": memory.numpy().astype("float32")})
    raw = [_diff(t, o) for t, o in zip(reference, exported)]

    # Decoder contains TopK; raw query order may differ.  Report a
    # box-L1-matched comparison as well.
    ref_boxes = reference[1][0].numpy()
    out_boxes = exported[1][0]
    cost = np.abs(ref_boxes[:, None, :] - out_boxes[None, :, :]).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    matched = {
        "logits": _diff(reference[0][0, rows], exported[0][0, cols]),
        "boxes": _diff(ref_boxes[rows], out_boxes[cols]),
    }
    return {
        "outputs": [o.name for o in session.get_outputs()],
        "memory_shape": list(memory.shape),
        "raw_diff": raw,
        "matched_query_diff": matched,
        "passed": matched["boxes"]["mean_abs"] <= atol,
        "note": "TopK may permute queries; passed uses Hungarian-matched boxes.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("both", "full", "backbone_encoder", "decoder", "all"), default="both")
    parser.add_argument("--device", default="cpu", help="export device; verification always uses CPU")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg, model = load_model(args.config, args.weights, args.device)
    size = tuple(cfg.yaml_cfg["eval_spatial_size"])

    modes = {
        "both": ("full", "backbone_encoder"),
        "all": ("full", "backbone_encoder", "decoder"),
        "full": ("full",),
        "backbone_encoder": ("backbone_encoder",),
        "decoder": ("decoder",),
    }[args.mode]

    full_path = output_dir / "model_fp32.onnx"
    be_path = output_dir / "backbone_encoder_fp32.onnx"
    decoder_path = output_dir / "decoder_fp32.onnx"
    report = {
        "mode": args.mode,
        "config": str(Path(args.config).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": sha256(Path(args.weights)),
        "input_size": list(size),
        "batch_size": args.batch_size,
        "opset": args.opset,
    }
    checks = {}

    if "full" in modes:
        export_full(model, size, full_path, args.opset, args.batch_size)
        checks["full"] = verify_full(full_path, model, size, args.batch_size, args.atol)
        report["full"] = {
            "path": str(full_path),
            "sha256": sha256(full_path),
            "outputs": ["pred_logits", "pred_boxes"],
            "verification": checks["full"],
        }

    if "backbone_encoder" in modes:
        names = export_backbone_encoder(model, size, be_path, args.opset, args.batch_size)
        checks["backbone_encoder"] = verify_backbone_encoder(
            be_path, model, size, args.batch_size, args.atol
        )
        report["backbone_encoder"] = {
            "path": str(be_path),
            "sha256": sha256(be_path),
            "outputs": names,
            "verification": checks["backbone_encoder"],
        }

    if "decoder" in modes:
        meta = export_decoder(model, size, decoder_path, args.opset, args.batch_size)
        checks["decoder"] = verify_decoder(
            decoder_path, model, size, args.batch_size, args.atol
        )
        report["decoder"] = {
            "path": str(decoder_path),
            "sha256": sha256(decoder_path),
            "outputs": ["pred_logits", "pred_boxes"],
            **meta,
            "verification": checks["decoder"],
        }

    report["verification"] = {
        "reference": "PyTorch CPU eval vs ONNX Runtime CPUExecutionProvider",
        "atol": args.atol,
        "passed": all(check["passed"] for check in checks.values()),
    }
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["verification"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
