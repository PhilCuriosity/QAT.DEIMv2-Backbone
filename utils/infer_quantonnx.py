"""Joint inference with backbone_encoder_qat.onnx + decoder_fp32.onnx.

Reference: the QAT checkpoint (``qat_state.pt``) contains the prepared PT2E
graph state, including fake-quant observers/scales.  We rebuild that graph,
load the state, run the fake-quant encoder and the FP32 decoder, then compare
with the two ONNX Runtime sessions.

Usage:
    python utils/infer_quantonnx.py \
      --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
      --checkpoint runs/qat/<run>/qat_state.pt \
      --image example.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from qat.boundaries import spatial_shapes_for  # noqa: E402
from qat.pt2e import convert, disable_observers, prepare_model  # noqa: E402


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_qat_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu")
    return state["qat_model"] if "qat_model" in state else state


def _prepare_reference(config: str, checkpoint: Path, quant_config: Path, size, mode: str = "converted", weights: Path | None = None):
    cfg = YAMLConfig(config)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    if weights is not None:
        state = torch.load(weights, map_location="cpu")
        state = state.get("ema", state).get("module", state.get("model", state))
        model.load_state_dict(state)
    prepared = prepare_model(
        model,
        torch.randn(1, 3, *size),
        str(quant_config),
        device="cpu",
        dynamic_batch_max=0,
    )
    prepared.load_state_dict(_load_qat_state(checkpoint), strict=True)
    disable_observers(prepared)
    prepared.eval()
    if mode == "converted":
        prepared = convert(prepared)
        try:
            torch.ao.quantization.allow_exported_model_train_eval(prepared)
        except (AttributeError, RuntimeError):
            pass
        prepared.eval()
    model.decoder.num_denoising = 0
    model.decoder.eval()
    return cfg, model, prepared


def _preprocess(path: Path, size, normalize: bool):
    image = Image.open(path).convert("RGB")
    image = resize(image, [int(size[0]), int(size[1])], antialias=True)
    tensor = pil_to_tensor(image).float().unsqueeze(0) / 255.0
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tensor = (tensor - mean) / std
    return tensor


def _has_normalize(cfg) -> bool:
    ops = cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {}).get("transforms", {}).get("ops", [])
    return any(isinstance(op, dict) and op.get("type") == "Normalize" for op in ops)


def _diff(a, b) -> dict:
    a = a.detach().cpu().numpy() if torch.is_tensor(a) else a
    b = b.detach().cpu().numpy() if torch.is_tensor(b) else b
    delta = np.abs(a - b)
    return {"shape": list(a.shape), "max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def _matched_diff(ref_logits, ref_boxes, onnx_logits, onnx_boxes) -> dict:
    from scipy.optimize import linear_sum_assignment

    ref_boxes = ref_boxes[0].detach().cpu().numpy()
    onnx_boxes = onnx_boxes[0]
    cost = np.abs(ref_boxes[:, None, :] - onnx_boxes[None, :, :]).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    return {
        "logits": _diff(ref_logits[0, rows], onnx_logits[0, cols]),
        "boxes": _diff(ref_boxes[rows], onnx_boxes[cols]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="optional qat_state.pt for pt-reference comparison")
    parser.add_argument("--weights", default=None, help="FP32 checkpoint; required with --checkpoint")
    parser.add_argument("--reference-mode", choices=("converted", "prepared"), default="converted",
                        help="converted PT2E Q/DQ matches ONNX export; prepared is the raw fake-quant graph")
    parser.add_argument("--quant-config", default=None)
    parser.add_argument("--onnx-dir", default=None)
    parser.add_argument("--image", default=str(REPO_ROOT / "example.jpg"))
    parser.add_argument("--random", action="store_true", help="use a random image instead of --image")
    parser.add_argument("--output", default=None, help="optional JSON result path")
    parser.add_argument("--save-npz", default=None, help="optional .npz with memory/logits/boxes")
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()

    cfg = YAMLConfig(args.config)
    size = tuple(cfg.yaml_cfg["eval_spatial_size"])
    if args.random:
        image = torch.rand(1, 3, *size)
    else:
        image = _preprocess(Path(args.image), size, _has_normalize(cfg))

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None:
        run_dir = checkpoint.parent
        onnx_dir = Path(args.onnx_dir) if args.onnx_dir else run_dir / "quantonnx"
        quant_config = Path(args.quant_config) if args.quant_config else run_dir / "quant_config_discovered.json"
        if not quant_config.is_file():
            quant_config = REPO_ROOT / "qat/configs/quant/w8a16.json"
    else:
        if not args.onnx_dir:
            raise SystemExit("--onnx-dir is required when --checkpoint is omitted")
        onnx_dir = Path(args.onnx_dir)
        quant_config = None

    encoder_onnx = onnx_dir / "backbone_encoder_qat.onnx"
    decoder_onnx = onnx_dir / "decoder_fp32.onnx"
    if not encoder_onnx.is_file() or not decoder_onnx.is_file():
        raise FileNotFoundError(f"missing ONNX files in {onnx_dir}")

    enc_sess = ort.InferenceSession(str(encoder_onnx), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(decoder_onnx), providers=["CPUExecutionProvider"])
    memory_onnx = enc_sess.run(None, {"images": image.numpy().astype("float32")})[0]
    logits_onnx, boxes_onnx = dec_sess.run(None, {"memory": memory_onnx.astype("float32")})

    report = {
        "mode": "onnxruntime",
        "config": str(Path(args.config).resolve()),
        "encoder_onnx": str(encoder_onnx.resolve()),
        "decoder_onnx": str(decoder_onnx.resolve()),
        "image": "random" if args.random else str(Path(args.image).resolve()),
        "input_shape": list(image.shape),
        "memory_shape": list(memory_onnx.shape),
        "logits_shape": list(logits_onnx.shape),
        "boxes_shape": list(boxes_onnx.shape),
    }

    # Save the raw joint output for downstream use / comparison.
    save_npz = Path(args.save_npz) if args.save_npz else onnx_dir / "joint_output.npz"
    np.savez(save_npz, memory=memory_onnx, pred_logits=logits_onnx, pred_boxes=boxes_onnx)
    report["output_npz"] = str(save_npz.resolve())

    if checkpoint is not None:
        if not args.weights:
            raise SystemExit("--weights is required with --checkpoint for the FP32 decoder reference")
        cfg, model, prepared = _prepare_reference(
            args.config, checkpoint, quant_config, size, args.reference_mode, Path(args.weights)
        )
        with torch.no_grad():
            memory_ref = prepared(image)
            shapes = spatial_shapes_for(image, model.decoder.feat_strides)
            outputs_ref = model.decoder.decode(memory_ref, shapes)
        ref_logits = outputs_ref["pred_logits"]
        ref_boxes = outputs_ref["pred_boxes"]
        matched = _matched_diff(ref_logits, ref_boxes, logits_onnx, boxes_onnx)
        report.update(
            {
                "mode": f"onnxruntime+{args.reference_mode}_reference",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "quant_config": str(quant_config.resolve()),
                "memory": _diff(memory_ref, memory_onnx),
                "logits": _diff(ref_logits, logits_onnx),
                "boxes": _diff(ref_boxes, boxes_onnx),
                "matched_query": matched,
                "atol": args.atol,
                "passed": (
                    matched["logits"]["mean_abs"] <= args.atol
                    and matched["boxes"]["mean_abs"] <= args.atol
                ),
            }
        )
    else:
        report["passed"] = True

    output = Path(args.output) if args.output else onnx_dir / "joint_infer_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
