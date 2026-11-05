"""Discover regional quantization boundaries from an exported FX graph.

Regional configs in QAT.axera address nodes by FX node name (``module_names``).
Hard-coded names break as soon as the backbone/encoder changes.  This module
discovers the nodes by role from the current graph and regenerates the names.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.fx

CONV_OPS = {torch.ops.aten.conv2d.default}
MATMUL_OPS = {torch.ops.aten.matmul.default}
BMM_OPS = {torch.ops.aten.bmm.default}
SOFTMAX_OPS = {torch.ops.aten.softmax.int}
GRIDSAMPLE_OPS = {
    torch.ops.aten.grid_sampler_2d.default,
}


def _module_paths(node: torch.fx.Node) -> list[str]:
    stack = node.meta.get("nn_module_stack") or {}
    paths = []
    for value in stack.values():
        if isinstance(value, tuple) and len(value) >= 2:
            paths.append(str(value[0]))
    return paths


def _module_path(node: torch.fx.Node) -> str:
    paths = _module_paths(node)
    return paths[-1] if paths else ""


def _attention_owner(node: torch.fx.Node) -> str | None:
    """Return the attention module path that owns this node."""
    for path in _module_paths(node):
        lowered = path.lower()
        if "self_attn" in lowered:
            return path.split("self_attn")[0] + "self_attn"
        if ".attn" in lowered:
            return path.split(".attn")[0] + ".attn"
        if "multiheadattention" in lowered:
            return path
    return None


def _is_attention(node: torch.fx.Node) -> bool:
    path = _module_path(node).lower()
    return any(token in path for token in ("self_attn", "attention", ".attn", "multiheadattention"))


def _is_depthwise_conv(node: torch.fx.Node) -> bool:
    weight = node.args[1] if len(node.args) > 1 else None
    if isinstance(weight, torch.fx.Node):
        value = weight.meta.get("val")
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 2:
            # depthwise conv weight: [out_channels, 1, kH, kW]
            return int(shape[1]) == 1
    return "depthwise" in _module_path(node).lower() or ".dw" in _module_path(node).lower()


def discover_regions(gm: torch.fx.GraphModule) -> dict[str, Any]:
    """Return node roles discovered from the current exported graph."""
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    convs = []
    softmax = []
    attention = []
    gridsample = []
    unsupported = []

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        entry = {
            "name": node.name,
            "target": str(target),
            "module": _module_path(node),
        }
        if target in CONV_OPS:
            convs.append({**entry, "depthwise": _is_depthwise_conv(node)})
        elif target in SOFTMAX_OPS:
            softmax.append({**entry, "owner": _attention_owner(node)})
        elif target in MATMUL_OPS or target in BMM_OPS:
            if _is_attention(node):
                attention.append({
                    **entry,
                    "owner": _attention_owner(node),
                    "supported": target in MATMUL_OPS,
                })
            elif target in BMM_OPS:
                unsupported.append({**entry, "reason": "bmm has no AXERA annotator"})
        elif target in GRIDSAMPLE_OPS:
            gridsample.append(entry)

    convs.sort(key=lambda item: order[next(n for n in gm.graph.nodes if n.name == item["name"])])
    by_prefix: dict[str, list[dict[str, Any]]] = OrderedDict()
    for item in convs:
        prefix = item["module"].split(".", 1)[0] or "root"
        by_prefix.setdefault(prefix, []).append(item)

    first_last = {}
    for prefix, items in by_prefix.items():
        first_last[prefix] = [items[0], items[-1]]

    grouped: dict[str, list[dict[str, Any]]] = OrderedDict()
    for item in attention:
        grouped.setdefault(item.get("owner") or "_", []).append(item)
    for items in grouped.values():
        items.sort(key=lambda it: order[next(n for n in gm.graph.nodes if n.name == it["name"])])
        if len(items) == 2:
            items[0]["role"] = "first"
            items[1]["role"] = "second"
        else:
            for item in items:
                item["role"] = "unknown"

    return {
        "conv": convs,
        "softmax": softmax,
        "attention": attention,
        "first_last_conv": first_last,
        "gridsample": gridsample,
        "unsupported": unsupported,
    }


def _s8_region(module_type: str, names, *, output: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {
        "is_symmetric": True,
        "input": {"dtype": "S8", "qmin": -127, "qmax": 127},
    }
    if output:
        config["output_is_symmetric"] = True
        config["output"] = {"dtype": "S8", "qmin": -127, "qmax": 127}
    return {
        "module_names": sorted(set(names)) if names else None,
        "module_type": module_type,
        "module_config": config,
    }


def _fp32_region(module_type: str, names: list[str]) -> dict[str, Any]:
    return {
        "module_names": sorted(set(names)),
        "module_type": module_type,
        "module_config": {
            "is_symmetric": True,
            "input": {"dtype": "FP32"},
            "output": {"dtype": "FP32"},
        },
    }


def _w8a8_region(module_type: str, names: list[str]) -> dict[str, Any]:
    return {
        "module_names": sorted(set(names)),
        "module_type": module_type,
        "module_config": {
            "is_symmetric": False,
            "input": {"dtype": "U8", "qmin": 0, "qmax": 255},
            "weight": {"dtype": "S8", "qmin": -127, "qmax": 127},
            "output": {"dtype": "U8", "qmin": 0, "qmax": 255},
        },
    }


def _w8a16_region(module_type: str, names: list[str]) -> dict[str, Any]:
    return {
        "module_names": sorted(set(names)),
        "module_type": module_type,
        "module_config": {
            "is_symmetric": False,
            "input": {"dtype": "U16", "qmin": 0, "qmax": 65535},
            "weight": {"dtype": "S8", "qmin": -127, "qmax": 127},
            "output": {"dtype": "U16", "qmin": 0, "qmax": 65535},
        },
    }


def build_regional_config(
    base_config: str | Path,
    regions: dict[str, Any],
    *,
    policy: str = "w8a16",
) -> dict[str, Any]:
    """Create a quant config with regional entries discovered from the graph."""
    config = json.loads(Path(base_config).read_text())
    regional = list(config.get("regional_configs", []))

    # Softmax is left to the global qspec (U8/U16).  It is still reported by
    # discover_regions() for inspection, but no FP32 override is emitted.
    # GridSample is float in the current backbone/encoder graph; keep it FP32
    # when a future model introduces it.
    if regions.get("gridsample"):
        regional.append(_fp32_region("gridsample", [item["name"] for item in regions["gridsample"]]))

    if policy in ("w8a16", "w8a8"):
        region_fn = _w8a8_region if policy == "w8a8" else _w8a16_region
        depthwise = [item["name"] for item in regions.get("conv", []) if item.get("depthwise")]
        if depthwise:
            regional.append(region_fn("conv", depthwise))

        edge_convs = []
        for items in regions.get("first_last_conv", {}).values():
            edge_convs.extend(item["name"] for item in items)
        if edge_convs:
            regional.append(region_fn("conv", edge_convs))

        if policy == "w8a8":
            # Attention S8 region: all MatMul inputs are S8; the first MatMul
            # and Softmax keep S8 on output so the backend sees one contiguous
            # S8 domain and does not insert requant nodes inside attention.
            regional.append(_s8_region("matmul", None))
            first_matmuls = [
                item["name"] for item in regions.get("attention", [])
                if item.get("role") == "first"
            ]
            if first_matmuls:
                regional.append(_s8_region("matmul", first_matmuls, output=True))
            attention_softmax = [
                item["name"] for item in regions.get("softmax", [])
                if item.get("owner")
            ]
            if attention_softmax:
                regional.append(_s8_region("softmax", attention_softmax, output=True))
        else:
            attention = [
                item["name"] for item in regions.get("attention", [])
                if item.get("supported")
            ]
            if attention:
                regional.append(region_fn("matmul", attention))

    config["regional_configs"] = regional
    return config


def validate_regions(regions: dict[str, Any], gm: torch.fx.GraphModule) -> list[str]:
    """Return regional names that do not exist in the graph."""
    names = {node.name for node in gm.graph.nodes}
    missing = []
    for item in regions.get("softmax", []):
        if item["name"] not in names:
            missing.append(item["name"])
    for item in regions.get("gridsample", []):
        if item["name"] not in names:
            missing.append(item["name"])
    for item in regions.get("conv", []):
        if item["name"] not in names:
            missing.append(item["name"])
    for item in regions.get("attention", []):
        if item["name"] not in names:
            missing.append(item["name"])
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="DEIMv2 model YAML")
    parser.add_argument("--weights", default=None, help="optional checkpoint")
    parser.add_argument("--base-config", required=True, help="base W8A16 quant config")
    parser.add_argument("--output", required=True, help="generated quant config JSON")
    parser.add_argument("--report", default=None, help="discovery report JSON")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", default="w8a16")
    args = parser.parse_args()

    from engine.core import YAMLConfig
    from qat.export import export_backbone_encoder

    cfg = YAMLConfig(args.config)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        state = state.get("ema", state).get("module", state.get("model", state))
        model.load_state_dict(state)
    model.to(args.device).eval()
    gm = export_backbone_encoder(
        model,
        torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=args.device),
    )

    regions = discover_regions(gm)
    missing = validate_regions(regions, gm)
    if missing:
        raise RuntimeError(f"discovered names missing from graph: {missing[:5]}")

    config = build_regional_config(args.base_config, regions, policy=args.policy)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(config, indent=2) + "\n")

    report = {
        "model_config": str(Path(args.config).resolve()),
        "weights": args.weights,
        "base_config": str(Path(args.base_config).resolve()),
        "output": str(Path(args.output).resolve()),
        "policy": args.policy,
        "counts": {key: len(value) if isinstance(value, list) else len(value) for key, value in regions.items()},
        "regions": regions,
    }
    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(".discovery.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "regions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
