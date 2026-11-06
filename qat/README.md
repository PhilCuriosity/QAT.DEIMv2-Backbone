# qat

Backbone + encoder QAT（路线 B），decoder 保持 FP32 冻结。量化目标优先 W8A16。

以下命令假设已激活项目 Python 环境（torch 2.6）；COCO 数据根通过 `$COCO_ROOT` 或
`--data-root` 指定：

```bash
export COCO_ROOT=/path/to/coco
```

## 结构

```text
images
  → backbone
  → encoder
  → decoder._get_encoder_input
  → memory [B, N_tokens, C]      ← QAT 图边界
  → FP32 decoder (frozen)
  → pred_logits / pred_boxes
  → PostProcessor
```

TopK、anchor、decoder、LQE 全部在 FP32 decoder 内，不进入 QAT 图。

## 文件

| 文件 | 作用 |
|---|---|
| `boundaries.py` | `BackboneEncoderQAT`、`FrozenDecoder`、spatial shapes |
| `compat.py` | RepNCSPELAN / torchvision 0.21 导出兼容 |
| `export.py` | backbone+encoder FX export |
| `pt2e.py` | AXQuantizer prepare、校准、convert、rank-one guard |
| `regional.py` | 从 FX 图自动发现 regional config 节点 |
| `train_backbone_encoder.py` | W8A16 QAT 训练 CLI |
| `export_quantonnx.py` | QAT checkpoint → backbone+encoder QuantONNX + decoder FP32 ONNX |
| `export_fp32_onnx.py` | FP32 ONNX 导出（full / backbone_encoder / decoder） |
| `infer_quantonnx.py` | QuantONNX + decoder 联合推理，可选 converted PT2E 参考对比 |
| `configs/quant/w8a16.json` | U16 激活 / S8 权重 |
| `configs/task/deimv2_n_w8a16.json` | N 任务配置模板 |
| `tests/test_model_change.py` | 模型变更 + regional discovery 回归测试 |

## 训练

```bash
python -m qat.train_backbone_encoder \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --weights /path/to/deimv2_hgnetv2_n_coco.pth \
  --output-dir runs/qat/deimv2_n_w8a16_qat_$(date +%Y%m%d_%H%M%S) \
  --data-root "$COCO_ROOT" \
  --device cuda:0 --epochs 1 --max-steps 500 \
  --batch-size 16 --calibration-batches 16
```

默认启用 regional config 自动发现：

- 导出 backbone+encoder FX 图；
- 发现 softmax、depthwise conv、首/尾 conv、attention 等角色；
- 生成 `<output>/quant_config_discovered.json` 和 `regional_discovery.json`；
- 使用生成的配置进行 prepare。

关闭自动发现：

```bash
--no-auto-regional
```

## 测试

```bash
python -m pytest \
  qat/tests/test_model_change.py -q
```

测试覆盖：

- 在 encoder 后插入新的 Conv+BN+SiLU block，QAT export/prepare 仍成功；
- 新 block 的节点被 regional discovery 识别；
- 生成的 regional config 可被 `ax_load_config` 加载；
- 修改模型后仍能完成一次 QAT forward/backward/optimizer step。

## LSQ 与蒸馏

### LSQ

```bash
python -m qat.train_backbone_encoder ... --lsq
```

- 使用 `utils/ax_quantizer_lsq.py`
- checkpoint 写入 `"use_lsq": true`
- 导出时自动读取该字段重建 LSQ 图；也可显式 `--lsq`
- 已验证：1-step LSQ + KD smoke、LSQ checkpoint → QuantONNX 导出通过
  - `runs/smoke/<smoke_dir>`（历史 N smoke 目录已删除）

### 蒸馏

```bash
python -m qat.train_backbone_encoder ... \
  --teacher-weights /path/to/fp32.pth \
  --kd-memory-weight 0.1 \
  --kd-logits-weight 1.0 \
  --kd-boxes-weight 5.0 \
  --kd-giou-weight 2.0
```

- teacher：FP32 全模型，冻结；保持 eager SDPA，不切 `explicit_attention`
- student：backbone+encoder QAT 图使用导出时的显式 attention
- KD 以真实 FP32 输出为目标，接受 SDPA 与显式路径的数值差异；最终精度看全量 COCO mAP
- memory cosine loss
- 最终 logits/boxes/GIoU 使用 Hungarian 匹配，避免 encoder TopK 重排
- 1-step smoke 已通过；精度待全量训练验证

checkpoint 记录 `use_lsq`、`distill` 字段。

## W8A8 纯 QAT（DEIMv2-S，50 epoch，本轮不启用 KD）

本轮先验证 **DEIMv2-S 的单纯 QAT**：

- 模型：DINOv3-S，`configs/deimv2/deimv2_dinov3_s_coco.yml`
- 权重：`weights/deimv2_dinov3_s_coco.pth`
- W8A8（U8 激活 / S8 权重）
- backbone + encoder QAT
- decoder 冻结 FP32
- 不启用 LSQ
- 不启用蒸馏

```bash
CUDA_VISIBLE_DEVICES=0 python -m qat.train_backbone_encoder \
  --config configs/deimv2/deimv2_dinov3_s_coco.yml \
  --weights weights/deimv2_dinov3_s_coco.pth \
  --quant-config qat/configs/quant/w8a8.json \
  --output-dir runs/qat/deimv2_s_w8a8_qat_$(date +%Y%m%d_%H%M%S) \
  --data-root "$COCO_ROOT" \
  --device cuda:0 \
  --batch-size 32 \
  --workers 8 \
  --epochs 50 \
  --lr 1e-5 \
  --calibration-batches 32 \
  --eval-every 1
```

说明：

- 权重 SHA256：`fa07a020c8c7474dc20c70a34f2b16a6a6920b9d2d421146dd79a5d9d72fb2da`
- 不传 `--teacher-weights`，因此不构建 teacher，也不计算 KD loss。
- 不传 `--lsq`，使用普通 AXQuantizer。
- 不传 `--max-steps`，每个 epoch 跑完整 COCO train2017。
- `--calibration-batches 32`：用 val loader 的 32 个 batch 做 observer 校准；
  按 batch 32 计算约 1024 张图。校准只初始化/更新量化范围，不更新权重。
- 默认校准后关闭 observer，50 epoch 使用固定 qparams；
  可加 `--keep-observers` 让 observer 在训练中继续更新。
- `--eval-every 1`：每个 epoch 结束后做一次全量 val2017 评估，并写入
  `eval_history.jsonl`；最终结果同时写入 `eval.json`。
- `--eval-interval N` 是 `--eval-every` 的别名；默认 0 表示不做周期性评估。
- 每个 epoch 都会保存 checkpoint，但 `eval-every` 本身不会改动训练权重。
- 注意：每次评估都跑完整 5000 张，50 epoch 会增加较多时间。

产物：

```text
<output-dir>/
├── qat_state.pt
├── quant_config_discovered.json
├── regional_discovery.json
├── train_history.json
├── eval_history.jsonl        # 每个评估 epoch 的 mAP
└── eval.json                 # 最后一次评估结果
```

KD 说明：`qat/distill.py` 和 `--teacher-weights` / `--kd-*` 仍保留，但本轮不启用；
后续 KD 对照实验再单独开 run。

observer 行为：

- 默认：`calibrate()` 用 `--calibration-batches` 校准后**关闭 observer**，QAT 全程使用校准得到的固定 qparams。
- `--keep-observers`：校准后保留 observer，训练中继续更新量化范围。
- 若同时设置 `--observer-steps N`，会在第 N 个 global step 冻结 observer；默认 `-1` 表示不主动冻结。
- checkpoint 会记录 `keep_observers` 字段。

## FP32 decoder 微调（不进入 PT2E）

默认 decoder 保持冻结。加 `--train-decoder` 后：

- decoder 参数解冻，作为独立的 optimizer param group，学习率由 `--decoder-lr` 控制
  （默认 `1e-5`）；
- decoder 仍是 FP32，不进入 backbone+encoder 的 PT2E 图；
- 梯度裁剪同时覆盖 prepared 和 decoder；
- checkpoint 增加 `"decoder"` state 和 `"train_decoder": true`；
- `qat.export_quantonnx` 会自动从 checkpoint 加载该 decoder，生成微调后的
  `decoder_fp32.onnx`。

```bash
CUDA_VISIBLE_DEVICES=0 python -m qat.train_backbone_encoder \
  --config configs/deimv2/deimv2_dinov3_s_coco.yml \
  --weights weights/deimv2_dinov3_s_coco.pth \
  --quant-config qat/configs/quant/w8a8.json \
  --output-dir runs/qat/deimv2_s_w8a8_decoder_$(date +%Y%m%d_%H%M%S) \
  --data-root "$COCO_ROOT" \
  --device cuda:0 --batch-size 32 --workers 8 \
  --epochs 50 --lr 1e-5 --decoder-lr 1e-6 \
  --calibration-batches 32 --eval-every 1 \
  --train-decoder
```

注意：decoder 微调不是 decoder QAT；decoder 仍然没有 Q/DQ 图。

## FP32 ONNX 导出

```bash
python -m qat.export_fp32_onnx \
  --config configs/deimv2/deimv2_dinov3_s_coco.yml \
  --weights weights/deimv2_dinov3_s_coco.pth \
  --output-dir runs/export/deimv2_s_fp32_onnx \
  --mode both --device cpu
```

`--mode {both,full,backbone_encoder,decoder,all}`，导出后用 ONNX Runtime CPU 与
PyTorch CPU 逐输出对比，结果写入 `export_report.json`。

## QuantONNX 导出

```bash
python -m qat.export_quantonnx \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --weights /path/to/deimv2_hgnetv2_n_coco.pth \
  --run-dir runs/qat/<run_dir> \
  --device cpu
```

导出为**静态 batch=1**；`export_quantonnx.py` 不再提供 `--dynamic-batch-max`。
QAT 训练仍可使用 dynamic batch，导出时会重建静态 prepared 图并加载训练 checkpoint。

产物：

- `quantonnx/backbone_encoder_qat.onnx`：`images [1,3,640,640] → memory [1,2000,128]`
- `quantonnx/decoder_fp32.onnx`：`memory → pred_logits / pred_boxes`
- `quantonnx/export_report.json`：SHA256、Q/DQ 计数、ORT vs PyTorch 误差

当前 N W8A16 QAT 导出结果：233 QuantizeLinear / 376 DequantizeLinear，
`memory` 对 converted PT2E 的 mean_abs = 2.39e-4，max_abs = 0.035。

导出流程：

1. `torch.onnx.export(..., dynamo=True)`；
2. `program.optimize()`：折叠常量、内联 ONNXScript function、去掉多余的
   `Constant` / `pkg.onnxscript.torch_lib` 节点；
3. `onnxslim.slim`：做最终图清理。

backbone+encoder 图从 900 个节点简化为 899 个节点，`Constant` 节点 0，函数 0；
decoder FP32 图 545 → 548 个节点（`program.optimize()` 后 onnxslim 仅做少量清理）。
可用 `--no-slim` 关闭 onnxslim。

## 联合推理

```bash
# 仅 ONNX Runtime
python -m qat.infer_quantonnx \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --onnx-dir runs/qat/<run_dir>/quantonnx \
  --image example.jpg

# 与 converted PT2E 参考对比（需要 FP32 权重构建 decoder）
python -m qat.infer_quantonnx \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --checkpoint runs/qat/<run_dir>/qat_state.pt \
  --weights /path/to/deimv2_hgnetv2_n_coco.pth \
  --image example.jpg
```

onnxruntime 输出保存为 `quantonnx/joint_output.npz`（memory / pred_logits / pred_boxes）；
参考对比写入 `quantonnx/joint_infer_report.json`。

## 当前初步结果（全量 val2017）

| 实验 | 设置 | mAP50-95 | Δ |
|---|---|---:|---:|
| EXP-001 | W8A16 PTQ，16 校准 batch | 21.8150 | -21.1674 |
| EXP-002 | W8A16 QAT，200 step，lr 1e-5 | 31.9316 | -11.0508 |

N W8A16 的两个运行目录（`deimv2_n_w8a16_ptq_20261104_184900`、
`deimv2_n_w8a16_qat200_lr1e-5_20261104_184700`）已删除，指标保留在本文档与
`docs/qat-experiment-log.md`；`runs/export/deimv2_n_fp32_onnx_20261104_165100` 仍保留。

FP32 基线为 42.9824。200 step 约 3200 张图，远未完成 1 epoch；后续需要更多训练、lr 调参、
LSQ 或混合精度。

## 注意

- 默认导出 `wrapper.eval()` 图，使用 eval 分支的 cached position embedding；当前按固定 640 输入。
- decoder 参数冻结，但保持 `train()` 以产生 criterion 需要的 aux 输出。
- `num_denoising` 在 QAT 时设为 0，避免可变 denoising query 影响导出/训练统计。
- 路线 A（全网络 QAT）只记录在 `docs/qat-pipeline.md`，当前不实现。
