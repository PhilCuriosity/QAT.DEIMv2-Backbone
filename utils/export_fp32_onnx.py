"""Export DEIMv2 FP32 ONNX graphs and verify them against the float model.

Outputs:
  - full model: images -> pred_logits, pred_boxes
  - backbone+encoder: images -> multi-scale feature maps

Verification runs the exported ONNX with ONNX Runtime (CPU) and compares every
output directly with the PyTorch float model executed on CPU.  CPU-vs-CPU is
required: CUDA vs CPU numerical differences can change the encoder TopK query
selection, which permutes the decoder outputs even though the model is correct.

Usage:
  python utils/export_fp32_onnx.py \
    --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
    --weights /path/to/deimv2_hgnetv2_n_coco.pth \
    --output-dir runs/deimv2_hgnetv2_n_coco
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


def verify(
    full_path: Path,
    be_path: Path,
    model: nn.Module,
    size,
    batch: int,
    atol: float,
) -> dict:
    import onnx
    import onnxruntime as ort

    # ONNX Runtime uses CPUExecutionProvider here; the reference must run on CPU
    # too, otherwise CUDA-vs-CPU kernel differences can flip encoder TopK and
    # permute the 300 decoder queries.
    model = model.to("cpu").eval()
    example = _example(size, batch, "cpu")

    with torch.no_grad():
        torch_full = FullModel(model)(example)
        torch_be = BackboneEncoder(model)(example)

    for path in (full_path, be_path):
        onnx.checker.check_model(onnx.load(str(path)))

    full_sess = ort.InferenceSession(str(full_path), providers=["CPUExecutionProvider"])
    be_sess = ort.InferenceSession(str(be_path), providers=["CPUExecutionProvider"])
    inp = example.numpy().astype("float32")
    full_out = full_sess.run(None, {"images": inp})
    be_out = be_sess.run(None, {"images": inp})

    full_diffs = [_diff(t, o) for t, o in zip(torch_full, full_out)]
    be_diffs = [_diff(t, o) for t, o in zip(torch_be, be_out)]
    full_passed = all(item["max_abs"] <= atol for item in full_diffs)
    be_passed = all(item["max_abs"] <= atol for item in be_diffs)

    return {
        "reference": "PyTorch CPU eval vs ONNX Runtime CPUExecutionProvider",
        "atol": atol,
        "full": {
            "outputs": ["pred_logits", "pred_boxes"],
            "diff": full_diffs,
            "passed": full_passed,
        },
        "backbone_encoder": {
            "outputs": [o.name for o in be_sess.get_outputs()],
            "diff": be_diffs,
            "passed": be_passed,
        },
        "passed": full_passed and be_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", help="export device; verification always uses CPU")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, model = load_model(args.config, args.weights, args.device)
    size = tuple(cfg.yaml_cfg["eval_spatial_size"])
    full_path = output_dir / "model_fp32.onnx"
    be_path = output_dir / "backbone_encoder_fp32.onnx"

    export_full(model, size, full_path, args.opset, args.batch_size)
    names = export_backbone_encoder(model, size, be_path, args.opset, args.batch_size)
    verification = verify(full_path, be_path, model, size, args.batch_size, args.atol)

    report = {
        "config": str(Path(args.config).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": sha256(Path(args.weights)),
        "input_size": list(size),
        "batch_size": args.batch_size,
        "opset": args.opset,
        "full": {
            "path": str(full_path),
            "sha256": sha256(full_path),
            "outputs": ["pred_logits", "pred_boxes"],
        },
        "backbone_encoder": {
            "path": str(be_path),
            "sha256": sha256(be_path),
            "outputs": names,
        },
        "verification": verification,
    }
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
