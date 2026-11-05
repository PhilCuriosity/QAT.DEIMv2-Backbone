"""Export the backbone+encoder QAT graph to QuantONNX.

The QAT checkpoint contains a prepared PT2E graph state.  Rebuild the same
backbone+encoder graph, restore the state, convert to real Q/DQ and export:

  backbone_encoder_qat.onnx : images [1,3,H,W] -> memory [1,N_tokens,C]
  decoder_fp32.onnx         : memory [1,N_tokens,C] -> pred_logits/pred_boxes

Both files are verified with ONNX Runtime CPU against PyTorch CPU.
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
from qat.boundaries import spatial_shapes_for  # noqa: E402
from qat.pt2e import convert, disable_observers, prepare_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simplify_onnx(path: Path) -> dict:
    """Run onnxslim on the exported graph and overwrite the file."""
    import onnx
    from onnxslim import slim

    model = onnx.load(str(path))
    before = len(model.graph.node)
    before_constants = sum(1 for node in model.graph.node if node.op_type == "Constant")
    before_functions = len(model.functions)
    try:
        simplified = slim(model)
        mode = "full"
    except AttributeError:
        # onnxslim 0.1.48 has a constant-folding bug on some Q/DQ graphs
        # (list-valued constant: `list` object has no attribute `nbytes`).
        simplified = slim(onnx.load(str(path)), no_constant_folding=True)
        mode = "no_constant_folding"
    onnx.save(simplified, str(path))
    after = len(simplified.graph.node)
    return {
        "enabled": True,
        "mode": mode,
        "nodes_before": before,
        "nodes_after": after,
        "constant_nodes_before": before_constants,
        "constant_nodes_after": sum(
            1 for node in simplified.graph.node if node.op_type == "Constant"
        ),
        "functions_before": before_functions,
        "functions_after": len(simplified.functions),
    }

def merge_consecutive_qdq(path: Path) -> dict:
    """Collapse ``Q -> DQ -> Q -> DQ`` chains into one Q/DQ pair.

    The two pairs quantize the same float values, so the merge keeps a single
    pair.  When the qparams differ, the pair with the larger dequantized float
    range (``scale * (qmax - qmin)``) wins; ties keep the upstream pair.
    """
    import numpy as np
    import onnx
    from collections import defaultdict
    from onnx import numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    dtype_bounds = {
        onnx.TensorProto.UINT8: (0, 255),
        onnx.TensorProto.INT8: (-128, 127),
        onnx.TensorProto.UINT16: (0, 65535),
        onnx.TensorProto.INT16: (-32768, 32767),
    }
    tensor_type = {vi.name: vi.type.tensor_type.elem_type
                   for vi in list(graph.value_info) + list(graph.input) + list(graph.output)
                   if vi.type.tensor_type.elem_type}

    def initializer_array(name: str):
        init = initializers.get(name)
        return numpy_helper.to_array(init) if init is not None else None

    def quantized_dtype(node):
        if len(node.input) > 2:
            zero_point = initializer_array(node.input[2])
            if zero_point is not None:
                return onnx.helper.np_dtype_to_tensor_dtype(zero_point.dtype)
        return tensor_type.get(node.output[0])

    def float_range(node):
        scale = initializer_array(node.input[1])
        bounds = dtype_bounds.get(quantized_dtype(node))
        if scale is None or bounds is None:
            return None
        return float(np.max(scale)) * (bounds[1] - bounds[0])

    merged = 0
    while True:
        consumers = defaultdict(list)
        for node in graph.node:
            for inp in node.input:
                consumers[inp].append(node)

        removed = set()
        touched = []
        for q1 in list(graph.node):
            if id(q1) in removed or q1.op_type != "QuantizeLinear":
                continue
            users = consumers.get(q1.output[0], [])
            if len(users) != 1 or users[0].op_type != "DequantizeLinear":
                continue
            d1 = users[0]
            # d1 may feed several consumers (e.g. a residual branch and a
            # slice); merging one of its Q branches is still valid.
            q2s = [u for u in consumers.get(d1.output[0], []) if u.op_type == "QuantizeLinear"]
            if not q2s:
                continue
            q2 = q2s[0]
            users = consumers.get(q2.output[0], [])
            if len(users) != 1 or users[0].op_type != "DequantizeLinear":
                continue
            d2 = users[0]
            if quantized_dtype(q1) != quantized_dtype(q2):
                continue
            range1, range2 = float_range(q1), float_range(q2)
            if range1 is None or range2 is None:
                continue
            if range2 > range1:
                # Keep the qparams with the wider float range on the first pair.
                q1.input[1] = q2.input[1]
                if len(q2.input) > 2:
                    if len(q1.input) > 2:
                        q1.input[2] = q2.input[2]
                    else:
                        q1.input.append(q2.input[2])
            for consumer in consumers.get(d2.output[0], []):
                for index, inp in enumerate(consumer.input):
                    if inp == d2.output[0]:
                        consumer.input[index] = d1.output[0]
            removed.add(id(q2))
            removed.add(id(d2))
            merged += 1

        if not removed:
            break
        kept = [node for node in graph.node if id(node) not in removed]
        del graph.node[:]
        graph.node.extend(kept)

    if merged:
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    return {"merged_pairs": merged}


def insert_upstream_qdq_after_views(path: Path) -> dict:
    """Re-quantize a view-chain output with the upstream U8 grid before an S8 requant.

    ``DQ(U8) -> Slice/Squeeze -> Q(S8)`` loses the U8 pair right after the view.
    Backends that require every op output to carry the upstream quantization
    need ``view -> Q(U8) -> DQ(U8) -> Q(S8) -> DQ(S8)``.  Insert a copy of the
    nearest upstream U8 Q/DQ pair between the last view and the S8 Quantize.
    """
    import onnx
    from collections import defaultdict
    from onnx import helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    view_ops = {"Squeeze", "Slice", "Reshape", "Transpose", "Unsqueeze", "Flatten", "Gather", "Identity"}
    producer = {out: node for node in graph.node for out in node.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumers[inp].append(node)

    def zp_dtype(node):
        if node is None or len(node.input) < 3:
            return None
        init = next((i for i in graph.initializer if i.name == node.input[2]), None)
        return numpy_helper.to_array(init).dtype if init is not None else None

    inserted = 0
    insert_before = defaultdict(list)
    used_names = {node.name for node in graph.node}

    def unique(base: str) -> str:
        name = base
        index = 1
        while name in used_names:
            name = f"{base}_{index}"
            index += 1
        used_names.add(name)
        return name

    for quant in list(graph.node):
        if quant.op_type != "QuantizeLinear":
            continue
        if zp_dtype(quant) != "int8" or len(quant.input) < 3:
            continue
        views = []
        current = quant.input[0]
        while current in producer and producer[current].op_type in view_ops:
            views.append(producer[current])
            current = producer[current].input[0]
        if not views:
            continue
        upstream = producer.get(current)
        if upstream is None or upstream.op_type != "DequantizeLinear":
            continue
        if zp_dtype(upstream) != "uint8" or len(upstream.input) < 3:
            continue
        last_view = views[0]
        if len(list(consumers.get(last_view.output[0], []))) != 1:
            continue
        q_name = unique(f"upstream_{last_view.name}_q")
        dq_name = unique(f"upstream_{last_view.name}_dq")
        id_name = unique(f"upstream_{last_view.name}_requant")
        q_out = f"{q_name}_out"
        dq_out = f"{dq_name}_out"
        id_out = f"{id_name}_out"
        insert_before[id(quant)].append((
            helper.make_node(
                "QuantizeLinear", [last_view.output[0], upstream.input[1], upstream.input[2]],
                [q_out], name=q_name,
            ),
            helper.make_node(
                "DequantizeLinear", [q_out, upstream.input[1], upstream.input[2]],
                [dq_out], name=dq_name,
            ),
            helper.make_node(
                "Identity", [dq_out], [id_out], name=id_name,
            ),
        ))
        quant.input[0] = id_out
        inserted += 1

    if inserted:
        new_nodes = []
        for node in graph.node:
            for pair in insert_before.get(id(node), []):
                new_nodes.extend(pair)
            new_nodes.append(node)
        del graph.node[:]
        graph.node.extend(new_nodes)
        # helper.make_node leaves the new tensors without value_info; run shape
        # inference so each inserted edge carries dtype/shape like the original.
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    return {"inserted_pairs": inserted}


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def load_state_dict(path: Path) -> dict:
    state = load_checkpoint(path)
    if "qat_model" in state:
        return state["qat_model"]
    return state


class DecoderONNX(nn.Module):
    def __init__(self, decoder: nn.Module, spatial_shapes):
        super().__init__()
        self.decoder = decoder
        self.spatial_shapes = spatial_shapes

    def forward(self, memory: torch.Tensor):
        outputs = self.decoder.decode(memory, self.spatial_shapes)
        return outputs["pred_logits"], outputs["pred_boxes"]


def export_dynamo(model: nn.Module, example, path: Path, input_names, output_names, opset: int) -> None:
    program = torch.onnx.export(
        model,
        (example,),
        dynamo=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
    )
    program.optimize()
    graph = getattr(program.model, "graph", None)
    if graph is not None and hasattr(graph, "sort"):
        graph.sort()
    program.save(path)


def export_legacy(model: nn.Module, example, path: Path, input_name: str, output_names, opset: int) -> None:
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=[input_name],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=False,
    )


def diff(reference, exported) -> dict:
    import numpy as np

    reference = reference.detach().cpu().numpy() if torch.is_tensor(reference) else reference
    exported = exported.detach().cpu().numpy() if torch.is_tensor(exported) else exported
    delta = np.abs(reference - exported)
    return {"shape": list(reference.shape), "max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def verify_encoder(converted: nn.Module, onnx_path: Path, example) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort

    converted = converted.to("cpu")
    converted.eval()
    with torch.no_grad():
        reference = converted(example)
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    from collections import Counter
    op_counts = Counter(node.op_type for node in onnx_model.graph.node)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = session.run(None, {"images": example.numpy().astype("float32")})[0]
    metrics = diff(reference, out)
    return {
        "outputs": [o.name for o in session.get_outputs()],
        "quantize_linear": op_counts.get("QuantizeLinear", 0),
        "dequantize_linear": op_counts.get("DequantizeLinear", 0),
        "op_count": len(onnx_model.graph.node),
        "constant_nodes": op_counts.get("Constant", 0),
        "functions": len(onnx_model.functions),
        "reference": "converted PT2E Q/DQ PyTorch CPU",
        "diff": metrics,
        "passed": metrics["mean_abs"] <= 1e-3,
    }


def verify_decoder(decoder: nn.Module, onnx_path: Path, memory, spatial_shapes) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort

    wrapper = DecoderONNX(decoder.to("cpu").eval(), spatial_shapes)
    with torch.no_grad():
        reference = wrapper(memory)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = session.run(None, {"memory": memory.numpy().astype("float32")})
    metrics = [diff(t, o) for t, o in zip(reference, out)]
    return {
        "outputs": [o.name for o in session.get_outputs()],
        "diff": metrics,
        "passed": all(item["mean_abs"] <= 5e-3 for item in metrics),
        "note": "Decoder contains encoder TopK; raw query order may differ, compare mean_abs or postprocessed detections.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--run-dir", required=True, help="directory with qat_state.pt")
    parser.add_argument("--quant-config", default=None, help="defaults to run-dir/quant_config_discovered.json")
    parser.add_argument("--output-dir", default=None, help="defaults to run-dir/quantonnx")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=21)
    parser.add_argument("--export-batch", type=int, default=1)
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--lsq", action="store_true", help="force LSQ quantizer (auto-detected from checkpoint)")
    parser.add_argument("--slim", action=argparse.BooleanOptionalAction, default=True,
                        help="run onnxslim after export")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "quantonnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    quant_config = Path(args.quant_config) if args.quant_config else run_dir / "quant_config_discovered.json"
    if not quant_config.is_file():
        quant_config = REPO_ROOT / "qat/configs/quant/w8a16.json"

    # Register quantized_decomposed ONNX decompositions before dynamo export.
    import utils.quantized_decomposed_dequantize_per_channel  # noqa: F401

    cfg = YAMLConfig(args.config)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    state = torch.load(args.weights, map_location="cpu")
    state = state.get("ema", state).get("module", state.get("model", state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(args.device).eval()

    raw_checkpoint = load_checkpoint(run_dir / "qat_state.pt")
    use_lsq = bool(args.lsq or raw_checkpoint.get("use_lsq", False))
    prepared = prepare_model(
        model,
        torch.randn(1, 3, *args.input_size),
        str(quant_config),
        device=args.device,
        dynamic_batch_max=(64 if use_lsq else 0),
        use_lsq=use_lsq,
    )
    qat_state = raw_checkpoint["qat_model"] if "qat_model" in raw_checkpoint else raw_checkpoint
    prepared.load_state_dict(qat_state, strict=True)
    disable_observers(prepared)
    converted = convert(prepared).to(args.device)
    torch.ao.quantization.allow_exported_model_train_eval(converted)
    converted.eval()

    example = torch.randn(args.export_batch, 3, *args.input_size)
    print('example shape:', example.shape)
    encoder_path = output_dir / "backbone_encoder_qat.onnx"
    export_dynamo(converted, example, encoder_path, ["images"], ["memory"], args.opset)
    encoder_slim = simplify_onnx(encoder_path) if args.slim else {"enabled": False}
    encoder_merge = merge_consecutive_qdq(encoder_path)
    encoder_upstream = insert_upstream_qdq_after_views(encoder_path)
    encoder_report = verify_encoder(converted, encoder_path, example)

    # Decoder FP32 graph for composition / debugging.
    model.decoder.num_denoising = 0
    model.decoder.eval()
    spatial_shapes = spatial_shapes_for(example, model.decoder.feat_strides)
    num_tokens = sum(h * w for h, w in spatial_shapes)
    memory = torch.randn(args.export_batch, num_tokens, model.decoder.hidden_dim)
    decoder_path = output_dir / "decoder_fp32.onnx"
    decoder_wrapper = DecoderONNX(model.decoder, spatial_shapes).eval()
    export_dynamo(decoder_wrapper, memory, decoder_path, ["memory"], ["pred_logits", "pred_boxes"], args.opset)
    decoder_slim = simplify_onnx(decoder_path) if args.slim else {"enabled": False}
    decoder_report = verify_decoder(model.decoder, decoder_path, memory, spatial_shapes)

    report = {
        "model_config": str(Path(args.config).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": sha256(Path(args.weights)),
        "qat_state": str(run_dir / "qat_state.pt"),
        "qat_state_sha256": sha256(run_dir / "qat_state.pt"),
        "quant_config": str(quant_config.resolve()),
        "quant_config_sha256": sha256(quant_config),
        "use_lsq": use_lsq,
        "backbone_encoder": {
            "path": str(encoder_path),
            "sha256": sha256(encoder_path),
            "simplify": encoder_slim,
            "merge_qdq": encoder_merge,
            "insert_upstream_qdq": encoder_upstream,
            "verification": encoder_report,
        },
        "decoder": {
            "path": str(decoder_path),
            "sha256": sha256(decoder_path),
            "simplify": decoder_slim,
            "verification": decoder_report,
        },
        "passed": encoder_report["passed"],
    }
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
