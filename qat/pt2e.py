"""PT2E prepare/convert helpers for backbone+encoder QAT."""
from __future__ import annotations

import types
from typing import Any

import torch
import torch.fx
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

from qat.export import export_backbone_encoder


def disable_observers(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        disable = getattr(module, "disable_observer", None)
        if callable(disable):
            disable()
            count += 1
    return count


def disable_fake_quant(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        disable = getattr(module, "disable_fake_quant", None)
        if callable(disable):
            disable()
            count += 1
    return count


def _remove_empty_annotations(gm: torch.fx.GraphModule) -> int:
    removed = 0
    for node in gm.graph.nodes:
        annotation = node.meta.get("quantization_annotation")
        if annotation is None:
            continue
        if (
            node.op == "call_function"
            and node.kwargs
            and not getattr(annotation, "input_qspec_map", None)
            and getattr(annotation, "output_qspec", None) is None
        ):
            del node.meta["quantization_annotation"]
            removed += 1
    return removed


def _remove_invalid_conversion_annotations(gm: torch.fx.GraphModule) -> int:
    removed = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or str(node.target) != "aten.to.dtype_layout":
            continue
        annotation = node.meta.get("quantization_annotation")
        if annotation is None:
            continue
        qspecs = list(getattr(annotation, "input_qspec_map", {}).values())
        if getattr(annotation, "output_qspec", None) is not None:
            qspecs.append(annotation.output_qspec)
        if any(getattr(qspec, "ch_axis", None) is not None for qspec in qspecs):
            node.meta.pop("quantization_annotation", None)
            removed += 1
    return removed


def _repair_dangling_shared_qspecs(gm: torch.fx.GraphModule) -> int:
    """Drop SharedQuantizationSpec references to nodes that lost annotation.

    Some quantizer annotations (notably concat in DINOv3 attention) reference a
    shared node whose own annotation is empty and removed.  PT2E then fails
    with "please make sure only refer to edge or node that has observer/fake_quant
    inserted".  Removing the dangling reference keeps the remaining graph valid.
    """
    try:
        from torch.ao.quantization.quantizer.quantizer import SharedQuantizationSpec
    except ImportError:
        return 0

    def target_node(value):
        if isinstance(value, torch.fx.Node):
            return value
        candidate = getattr(value, "node", None)
        return candidate if isinstance(candidate, torch.fx.Node) else None

    annotated = {
        node for node in gm.graph.nodes
        if node.meta.get("quantization_annotation") is not None
    }
    repaired = 0
    for node in gm.graph.nodes:
        annotation = node.meta.get("quantization_annotation")
        if annotation is None:
            continue
        input_map = getattr(annotation, "input_qspec_map", None)
        if input_map:
            kept = {}
            for input_node, qspec in list(input_map.items()):
                if isinstance(qspec, SharedQuantizationSpec):
                    target = target_node(getattr(qspec, "edge_or_node", None))
                    if target is not None and target not in annotated:
                        repaired += 1
                        continue
                kept[input_node] = qspec
            annotation.input_qspec_map = kept
        output_qspec = getattr(annotation, "output_qspec", None)
        if isinstance(output_qspec, SharedQuantizationSpec):
            target = target_node(getattr(output_qspec, "edge_or_node", None))
            if target is not None and target not in annotated:
                annotation.output_qspec = None
                repaired += 1
        if not getattr(annotation, "input_qspec_map", None) and getattr(annotation, "output_qspec", None) is None:
            node.meta.pop("quantization_annotation", None)
    return repaired


def _rank_one_guards(model: torch.nn.Module) -> int:
    installed = 0
    for module in model.modules():
        forward = getattr(module, "forward", None)
        if not callable(forward):
            continue
        per_channel_fq = bool(getattr(module, "is_per_channel", False))
        per_channel_obs = "Observer" in type(module).__name__ and getattr(module, "ch_axis", None) is not None
        if not (per_channel_fq or per_channel_obs):
            continue
        if getattr(module, "_rank_one_guard", False):
            continue

        def guarded(*args, _original=forward, **kwargs):
            if args and torch.is_tensor(args[0]) and args[0].ndim <= 1:
                return args[0]
            return _original(*args, **kwargs)

        module.forward = guarded
        module._rank_one_guard = True
        installed += 1
    return installed


def _load_quantizer(config_path: str, use_lsq: bool = False):
    if use_lsq:
        from utils.ax_quantizer_lsq import AXQuantizer, ax_load_config
    else:
        from utils.ax_quantizer import AXQuantizer, ax_load_config
    global_config, regional_configs = ax_load_config(str(config_path), is_qat=True)
    quantizer = AXQuantizer()
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)
    return quantizer


def prepare_model(
    float_model: torch.nn.Module,
    example_images: torch.Tensor,
    quant_config: str,
    *,
    device: str = "cpu",
    dynamic_batch_max: int = 64,
    use_lsq: bool = False,
) -> torch.fx.GraphModule:
    for module in float_model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

    exported = export_backbone_encoder(
        float_model.to(device),
        example_images.to(device),
        dynamic_batch_max=dynamic_batch_max,
    ).to(device)

    quantizer = _load_quantizer(quant_config, use_lsq=use_lsq)
    original_annotate = quantizer.annotate

    def annotate(_self: Any, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        model = original_annotate(model)
        _remove_empty_annotations(model)
        _remove_invalid_conversion_annotations(model)
        dangling = _repair_dangling_shared_qspecs(model)
        if dangling:
            print(f"removed {dangling} dangling shared qspec reference(s)")
        return model

    quantizer.annotate = types.MethodType(annotate, quantizer)
    prepared = prepare_qat_pt2e(exported, quantizer).to(device)
    _rank_one_guards(prepared)
    torch.ao.quantization.allow_exported_model_train_eval(prepared)
    return prepared


def calibrate(
    prepared: torch.nn.Module,
    loader,
    *,
    device: str = "cpu",
    max_batches: int = 16,
    disable_observer: bool = True,
) -> int:
    prepared.eval()
    torch.ao.quantization.enable_observer(prepared)
    torch.ao.quantization.enable_fake_quant(prepared)
    seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            if not torch.is_tensor(images):
                continue
            prepared(images.to(device))
            seen += 1
            if seen >= max_batches:
                break
    if disable_observer:
        disable_observers(prepared)
    return seen


def convert(prepared: torch.nn.Module) -> torch.fx.GraphModule:
    _remove_invalid_conversion_annotations(prepared)
    return convert_pt2e(prepared)
