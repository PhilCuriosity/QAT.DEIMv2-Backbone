"""Backbone+encoder W8A16 QAT with a frozen FP32 decoder.

Usage:
    python -m qat.train_backbone_encoder \
        --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
        --weights weights/deimv2_hgnetv2_n_coco.pth \
        --output-dir runs/qat_n_w8a16 --epochs 1 --max-steps 500
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from qat.boundaries import BackboneEncoderQAT, FrozenDecoder, spatial_shapes_for  # noqa: E402
from qat.compat import install_torchvision_compat  # noqa: E402
from qat.distill import memory_loss, output_distill_loss  # noqa: E402
from qat.export import export_backbone_encoder  # noqa: E402
from qat.pt2e import calibrate, disable_observers, freeze_dropout, prepare_model  # noqa: E402
from qat.regional import build_regional_config, discover_regions  # noqa: E402


class QATEvalModel(nn.Module):
    def __init__(self, prepared: nn.Module, decoder: nn.Module, feat_strides):
        super().__init__()
        self.prepared = prepared
        self.decoder = decoder
        self.feat_strides = feat_strides

    def forward(self, images: torch.Tensor):
        memory = self.prepared(images)
        shapes = spatial_shapes_for(images, self.feat_strides)
        return self.decoder(memory, shapes)


class _LimitedLoader:
    """Wrap a DataLoader and expose at most ``max_batches`` batches."""

    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches

    def __len__(self):
        if hasattr(self.loader, "__len__"):
            return min(len(self.loader), self.max_batches)
        return self.max_batches

    def __iter__(self):
        for index, batch in enumerate(self.loader):
            if index >= self.max_batches:
                break
            yield batch

    def __getattr__(self, name):
        return getattr(self.loader, name)


def _evaluate_fake_quant(prepared, decoder, model, cfg, criterion, val_loader, device):
    """Run full validation without permanently changing observer/train state."""
    from engine.solver.det_engine import evaluate

    observer_states = []
    for module in prepared.modules():
        if hasattr(module, "observer_enabled"):
            observer_states.append((module, module.observer_enabled.clone()))
    prepared_was_training = prepared.training
    decoder_was_training = decoder.training

    prepared.eval()
    disable_observers(prepared)
    decoder.eval()
    eval_model = QATEvalModel(prepared, decoder, model.decoder.feat_strides).to(device)
    try:
        stats, _ = evaluate(
            eval_model, criterion, cfg.postprocessor, val_loader, cfg.evaluator, device
        )
    finally:
        if prepared_was_training:
            prepared.train()
            freeze_dropout(prepared)
        if decoder_was_training:
            decoder.train()
        criterion.train()
        for module, state in observer_states:
            module.observer_enabled.copy_(state)
    bbox = stats["coco_eval_bbox"]
    return {"mAP50-95": bbox[0], "mAP50": bbox[1], "coco_eval_bbox": bbox}


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _save_run_metadata(args, output_dir: Path, cfg=None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    command = shlex.join(sys.argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        command = f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} {command}"
    (output_dir / "command.txt").write_text(command + "\n")

    snapshot = output_dir / "config_snapshot"
    snapshot.mkdir(exist_ok=True)
    for key in ("config", "quant_config", "regional_config"):
        value = getattr(args, key, None)
        if value and Path(value).is_file():
            shutil.copy2(value, snapshot / Path(value).name)
    if cfg is not None:
        (snapshot / "resolved_config.json").write_text(
            json.dumps(cfg.yaml_cfg, indent=2, default=str) + "\n"
        )

    environment = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": _git_commit(),
    }
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")


def _build_cfg(args):
    install_torchvision_compat()
    cfg = YAMLConfig(args.config)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    for name, split in (("train_dataloader", "train2017"), ("val_dataloader", "val2017")):
        loader = cfg.yaml_cfg[name]
        loader["dataset"]["img_folder"] = str(Path(args.data_root) / split) + "/"
        loader["dataset"]["ann_file"] = str(
            Path(args.data_root) / "annotations" / f"instances_{split}.json"
        )
        loader["total_batch_size"] = args.batch_size
        loader["num_workers"] = args.workers
    return cfg


def _load_weights(model: nn.Module, weights: str | None) -> None:
    if not weights:
        print("warning: no --weights; using random initialization")
        return
    state = torch.load(weights, map_location="cpu")
    if "ema" in state:
        state = state["ema"]["module"]
    elif "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def _resolve_quant_config(args, model, output_dir: Path, device: str) -> Path:
    if args.regional_config:
        return Path(args.regional_config)
    if not args.auto_regional:
        return Path(args.quant_config)

    base = json.loads(Path(args.quant_config).read_text())
    input_dtype = base.get("global_config", {}).get("input", {}).get("dtype", "U16")
    policy = "w8a8" if input_dtype == "U8" else "w8a16"

    example = torch.randn(2, 3, *args.input_size, device=device)
    gm = export_backbone_encoder(model, example, dynamic_batch_max=args.dynamic_batch_max)
    regions = discover_regions(gm)
    config = build_regional_config(args.quant_config, regions, policy=policy)
    quant_path = output_dir / "quant_config_discovered.json"
    quant_path.write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "base_config": args.quant_config,
        "counts": {
            key: len(value) if isinstance(value, list) else len(value)
            for key, value in regions.items()
        },
        "regions": regions,
    }
    (output_dir / "regional_discovery.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"discovered regional config: {quant_path}")
    return quant_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="~/dataset/coco")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--keep-observers", action="store_true",
                        help="leave observers enabled after calibration; default disables them")
    parser.add_argument("--observer-steps", type=int, default=-1,
                        help="freeze observers at this global step; -1 means never freeze")
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--dynamic-batch-max", type=int, default=64)
    parser.add_argument("--quant-config", default=str(REPO_ROOT / "qat/configs/quant/w8a16.json"))
    parser.add_argument("--regional-config", default=None)
    parser.add_argument("--auto-regional", action="store_true", default=True)
    parser.add_argument("--no-auto-regional", dest="auto_regional", action="store_false")
    parser.add_argument("--lsq", action="store_true", help="use learnable step-size quantizer")
    parser.add_argument("--teacher-weights", default=None, help="FP32 teacher checkpoint for distillation")
    parser.add_argument("--kd-memory-weight", type=float, default=0.1)
    parser.add_argument("--kd-logits-weight", type=float, default=1.0)
    parser.add_argument("--kd-boxes-weight", type=float, default=5.0)
    parser.add_argument("--kd-giou-weight", type=float, default=2.0)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--kd-topk", type=int, default=100)
    parser.add_argument("--eval", action="store_true", help="run full val2017 at the end")
    parser.add_argument("--eval-every", "--eval-interval", dest="eval_every", type=int, default=0,
                        help="evaluate every N epochs and at the final epoch; 0 disables periodic eval")
    parser.add_argument("--train-max-batches", type=int, default=0,
                        help="cap training batches per epoch (smoke tests; 0 = full epoch)")
    parser.add_argument("--eval-max-batches", type=int, default=0,
                        help="cap validation batches (smoke tests; 0 = full val2017)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_metadata(args, output_dir)
    cfg = _build_cfg(args)
    _save_run_metadata(args, output_dir, cfg)
    model = cfg.model
    _load_weights(model, args.weights)
    model.to(args.device).eval()

    quant_config = _resolve_quant_config(args, model, output_dir, args.device)
    prepared = prepare_model(
        model,
        torch.randn(2, 3, *args.input_size),
        str(quant_config),
        device=args.device,
        dynamic_batch_max=args.dynamic_batch_max,
        use_lsq=args.lsq,
    )

    val_loader = cfg.val_dataloader
    eval_loader = val_loader
    if args.eval_max_batches:
        eval_loader = _LimitedLoader(val_loader, args.eval_max_batches)
        print(
            f"WARNING: eval limited to {args.eval_max_batches} batch(es); "
            "mAP is a subset value, not full val2017"
        )
    calibrated = calibrate(
        prepared,
        val_loader,
        device=args.device,
        max_batches=args.calibration_batches,
        disable_observer=not args.keep_observers,
    )
    print(f"calibrated {calibrated} batches")

    model.decoder.num_denoising = 0
    decoder = FrozenDecoder(model.decoder).to(args.device).train()

    teacher = None
    teacher_encoder = None
    if args.teacher_weights:
        teacher_cfg = _build_cfg(args)
        teacher = teacher_cfg.model
        _load_weights(teacher, args.teacher_weights)
        teacher.to(args.device).eval()
        teacher.decoder.num_denoising = 0
        teacher.decoder.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher_encoder = BackboneEncoderQAT(teacher).to(args.device).eval()
        print(f"distillation teacher: {args.teacher_weights}")

    train_loader = cfg.train_dataloader
    if hasattr(train_loader, "set_epoch"):
        train_loader.set_epoch(1000)

    prepared.train()
    frozen = freeze_dropout(prepared)
    if frozen:
        print(f"frozen {frozen} dropout node(s) to eval")
    torch.ao.quantization.enable_fake_quant(prepared)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in prepared.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = cfg.criterion.to(args.device)
    total = args.max_steps or len(train_loader)
    if args.train_max_batches:
        total = min(total, args.train_max_batches)
    print(f"QAT: {total} steps/epoch, {args.epochs} epoch(s)")

    history = []
    eval_history = []
    last_eval_epoch = -1
    global_step = 0
    for epoch in range(args.epochs):
        running = 0.0
        started = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if step >= total:
                break
            images = images.to(args.device)
            targets = [{key: value.to(args.device) for key, value in target.items()} for target in targets]
            memory = prepared(images)
            shapes = spatial_shapes_for(images, model.decoder.feat_strides)
            outputs = decoder(memory, shapes, targets)
            loss_dict = criterion(outputs, targets, epoch=epoch)
            loss = sum(loss_dict.values())

            kd_dict = {}
            if teacher is not None:
                with torch.no_grad():
                    teacher_memory = teacher_encoder(images)
                    teacher_outputs = teacher.decoder.decode(teacher_memory, shapes)
                kd_memory = memory_loss(memory, teacher_memory)
                kd_logits, kd_boxes, kd_giou = output_distill_loss(
                    outputs["pred_logits"],
                    outputs["pred_boxes"],
                    teacher_outputs["pred_logits"],
                    teacher_outputs["pred_boxes"],
                    topk=args.kd_topk,
                    temperature=args.kd_temperature,
                )
                loss = (
                    loss
                    + args.kd_memory_weight * kd_memory
                    + args.kd_logits_weight * kd_logits
                    + args.kd_boxes_weight * kd_boxes
                    + args.kd_giou_weight * kd_giou
                )
                kd_dict = {
                    "kd_memory": float(kd_memory),
                    "kd_logits": float(kd_logits),
                    "kd_boxes": float(kd_boxes),
                    "kd_giou": float(kd_giou),
                }
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prepared.parameters(), 0.1)
            optimizer.step()
            if args.observer_steps >= 0 and global_step == args.observer_steps:
                disable_observers(prepared)
                print("observers frozen")
            global_step += 1
            running += float(loss.detach())
            if step % 50 == 0:
                suffix = "".join(f" {k} {v:.5f}" for k, v in kd_dict.items())
                print(f"epoch {epoch} step {step}/{total} loss {float(loss):.5f}{suffix}")
        record = {
            "epoch": epoch,
            "steps": min(total, len(train_loader)),
            "loss": running / max(min(total, len(train_loader)), 1),
            "sec": round(time.time() - started, 1),
        }
        history.append(record)
        torch.save(
            {
                "qat_model": prepared.state_dict(),
                "quant_config": str(quant_config),
                "use_lsq": bool(args.lsq),
                "keep_observers": bool(args.keep_observers),
                "distill": teacher is not None,
                "history": history,
            },
            output_dir / "qat_state.pt",
        )
        print(record)
        (output_dir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")

        if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1):
            eval_report = _evaluate_fake_quant(
                prepared, decoder, model, cfg, criterion, eval_loader, args.device
            )
            eval_record = {"epoch": epoch, **eval_report}
            eval_history.append(eval_record)
            last_eval_epoch = epoch
            (output_dir / "eval_history.jsonl").write_text(
                "\n".join(json.dumps(item) for item in eval_history) + "\n"
            )
            print("eval", json.dumps(eval_record))

    if args.eval and last_eval_epoch != args.epochs - 1:
        eval_report = _evaluate_fake_quant(
            prepared, decoder, model, cfg, criterion, eval_loader, args.device
        )
        eval_history.append({"epoch": args.epochs - 1, **eval_report})
        last_eval_epoch = args.epochs - 1
        (output_dir / "eval_history.jsonl").write_text(
            "\n".join(json.dumps(item) for item in eval_history) + "\n"
        )
        print("eval", json.dumps(eval_history[-1]))

    if args.eval or args.eval_every > 0:
        final_report = eval_history[-1]
        (output_dir / "eval.json").write_text(json.dumps(final_report, indent=2) + "\n")
        print(json.dumps(final_report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
