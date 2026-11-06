# DEIMv2 QAT

本仓库基于 [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)（上游 `main` 分支），提供
backbone + encoder 的 PT2E QAT、QuantONNX 导出和 ONNX Runtime 验收工具链，
decoder 默认保持 FP32 冻结；如需微调，使用 `--train-decoder`，但 decoder 仍不进入
PT2E 图。当前主目标是 `W8A8`（U8 激活 / S8 权重），`W8A16`（U16 激活 / S8 权重）
作为对照和回退方案。目标覆盖上游已发布型号
Atto / Femto / Pico / N / S / M / L / X；当前已复现 FP32 的 N / Pico / S，
QAT 先以 S `W8A8` 和 N `W8A16` 验证链路。

上游原始 README 见 [DEIMv2 仓库](https://github.com/Intellindust-AI-Lab/DEIMv2)；
本仓库保留的副本为 `README-org.md`。许可证见 [`LICENSE.md`](./LICENSE.md)。

操作建议：先用小批量数据跑 1 个 epoch / 少量 step 的 smoke，检查 `eval_history.jsonl`、
`export_report.json` 和 `joint_infer_report.json`，再启动 50 epoch 全量训练；
最终用全量 COCO val2017 比较 FP32、QAT fake-quant 与 QuantONNX 的 `mAP50-95`。

## 精度结果

### 检测

本仓库只覆盖 COCO detection。指标口径为 COCO val2017 全量 5000 张，
`mAP50-95` 取项目 `CocoEvaluator` 的 `coco_eval_bbox[0]`；预处理、后处理和评估脚本
必须与 FP32 基线一致。完整协议见 [`docs/baseline.md`](docs/baseline.md)。

| 模型 | 模式 | 配置 | FP32 mAP50-95 | 量化 mAP50-95 | Err mAP:50~95 | 耗时 |
|---|---|---|---:|---:|---:|---:|
| HGNetv2-N | `FP32` | `deimv2_hgnetv2_n_coco.yml` | 42.9824 | - | - | - |
| HGNetv2-Pico | `FP32` | `deimv2_hgnetv2_pico_coco.yml` | 38.4593 | - | - | - |
| DINOv3-S | `FP32` | `deimv2_dinov3_s_coco.yml` | 50.9000 | - | - | - |
| HGNetv2-N | `W16A16 fake-quant` | `U16 / S16` | 42.9824 | 42.9678 | -0.0146 | - |
| HGNetv2-N | `W16A16 ORT` | `U16 / S16` | 42.9824 | 42.4903 | -0.4921 | - |
| HGNetv2-Pico | `W16A16 fake-quant` | `U16 / S16` | 38.4593 | 38.4756 | +0.0163 | - |
| HGNetv2-Pico | `W16A16 ORT` | `U16 / S16` | 38.4593 | 38.2732 | -0.1861 | - |

注：`W16A16` 行为历史**全网络 QAT**结果，产物已按要求回退，仅作为量化链路参考；
当前仓库实现的是 backbone + encoder QAT，不能直接把历史数字当作当前代码的交付结果。
`ORT` 表示 Converted PT2E 导出到 ONNX Runtime 后的结果。`耗时` 尚未在统一板端环境测量，
统一写 `-`。

> **⚠️ 当前 W8A8 / W8A16 状态**
>
> **S W8A8** 的 backbone + encoder QAT 已在 epoch 12 暂停（修复前代码），尚未完成 50 epoch。
> 当前 fake-quant `mAP50-95 = 44.40`，不能作为交付结果。QuantONNX 的全量 mAP 评估入口尚未接入；
> `export_report.json` 和 `joint_infer_report.json` 目前只做 tensor 级链路验收。
>
> **N W8A16** 的 PTQ 为 `21.8150`，200-step QAT 为 `31.9316`，均未达到 `< 1.0` 目标。
> W8A16 的 U16 fused qparams 问题见 [`docs/architecture.md`](docs/architecture.md) §11.7，
> 暂不处理。

## 量化配置

当前提供两个基础配置；实际训练默认启用 regional config 自动发现，并在 run 目录生成
`quant_config_discovered.json`。

| 配置 | 激活 | 权重 | 用途 |
|---|---|---|---|
| `qat/configs/quant/w8a8.json` | U8 | S8 | 当前主目标 |
| `qat/configs/quant/w8a16.json` | U16 | S8 | 对照与回退 |

- 默认 `--auto-regional`：导出 backbone + encoder 的 FX 图，自动发现 matmul、softmax、
  depthwise conv 等敏感区域，生成 S8 regional config；关闭用 `--no-auto-regional`。
- 默认校准后关闭 observer，QAT 全程使用固定 qparams；保留 observer 用 `--keep-observers`，
  或在第 N 个 global step 冻结用 `--observer-steps N`。
- S8 matmul / softmax、TopK、FDR / `Integral`、anchor 与 RoPE 的量化边界，见
  [`docs/architecture.md`](docs/architecture.md) 与 [`docs/qat-pipeline.md`](docs/qat-pipeline.md)。
- `symmetric` qspec 使用非 fused `FakeQuantize`，避免 fused kernel 把 symmetric 算成 affine；
  细节见 [`docs/architecture.md`](docs/architecture.md) §11.6。
- 实验计划和记录见 [`docs/qat-experiment-plan.md`](docs/qat-experiment-plan.md)、
  [`docs/qat-experiment-log.md`](docs/qat-experiment-log.md)。

## 快速开始

以下命令假设已激活项目 Python 环境（torch 2.6），并直接使用 `python`。
所有命令使用 `env PYTHONPATH="$PWD"`，确保直接调用仓库内模块。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

已验证环境为 `torch 2.6.0+cu124` / `torchvision 0.21.0+cu124`。`requirements.txt` 包含
QAT 所需的 `torch`、`onnxruntime`、`onnxscript`、`onnxslim` 等依赖；QAT 入口会自动安装
torchvision 0.21 兼容层，上游 `train.py` 的 FP32 基线复现见
[`docs/baseline.md`](docs/baseline.md)。

### 2. 准备数据集与权重

COCO 数据目录：

```text
$COCO_ROOT/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

`COCO_ROOT` 指向本地 COCO 数据根目录；项目 YAML 中的数据路径可能是占位路径，运行时用
`--data-root "$COCO_ROOT"` 覆盖。FP32 基线需要按 [`docs/baseline.md`](docs/baseline.md)
生成临时 YAML。开始前先设置：

```bash
export COCO_ROOT=/path/to/coco
```

权重不提交到仓库。按模型下载官方 checkpoint，并转换为 `{'model': state_dict}`：

```text
weights/
├── deimv2_hgnetv2_n_coco.pth
├── deimv2_hgnetv2_pico_coco.pth
└── deimv2_dinov3_s_coco.pth
```

来源：HuggingFace `Intellindust/DEIMv2_*_COCO` 或 README-org.md 中的 Google Drive / Quark。
DINOv3-S 必须使用项目 YAML 的 `interaction_indexes: [3, 7, 11]`；HF `config.json` 的
`[5, 8, 11]` 只有 `47.0`。

### 3. 运行 Smoke

先用 N 的 W8A16 跑 2 step，验证 prepare、伪量化、反向、保存和评估链路：

```bash
env PYTHONPATH="$PWD" \
  python -m qat.train_backbone_encoder \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --weights weights/deimv2_hgnetv2_n_coco.pth \
  --quant-config qat/configs/quant/w8a16.json \
  --output-dir runs/qat/smoke_n_w8a16_$(date +%Y%m%d_%H%M%S) \
  --data-root "$COCO_ROOT" \
  --device cuda:0 \
  --epochs 1 --max-steps 2 --batch-size 2 --workers 2 \
  --calibration-batches 2 --eval-every 1 --eval-max-batches 1
```

- **Smoke 指标**：`--eval-max-batches 1` 只评估 1 个 batch，mAP 仅用于链路验收，
  不能与全量结果比较。
- **已支持**：FX export、PT2E prepare/calibrate、fake-quant QAT、每 epoch checkpoint、
  评估历史、QuantONNX 导出。
- **未接入**：独立的 QAT checkpoint 全量评估 CLI；当前评估由训练脚本 `--eval-every` 完成。
- **⚠️ 仅链路验收，精度未调优**：2-step smoke 不能判断精度；完整精度需要跑满 epoch
  后评估全量 val2017。

smoke 产物：

```text
<output-dir>/
├── qat_state_epoch00.pt
├── qat_state.pt              # 最新 epoch 的 hardlink
├── qat_state_best.pt         # 最佳 epoch 的真实副本
├── quant_config_discovered.json
├── regional_discovery.json
├── train_history.json
└── eval_history.jsonl
```

### 4. 正式训练

#### 检测

以 S W8A8 为例；N / Pico 只替换 `--config`、`--weights` 和 batch 大小：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python -m qat.train_backbone_encoder \
  --config configs/deimv2/deimv2_dinov3_s_coco.yml \
  --weights weights/deimv2_dinov3_s_coco.pth \
  --quant-config qat/configs/quant/w8a8.json \
  --output-dir runs/qat/deimv2_s_w8a8_qat_$(date +%Y%m%d_%H%M%S) \
  --data-root "$COCO_ROOT" \
  --device cuda:0 \
  --batch-size 32 --workers 8 \
  --epochs 50 --lr 1e-5 \
  --calibration-batches 32 --eval-every 1
```

模型与入口：

| 模型 | config | weights | 建议 batch |
|---|---|---|---:|
| HGNetv2-N | `configs/deimv2/deimv2_hgnetv2_n_coco.yml` | `weights/deimv2_hgnetv2_n_coco.pth` | 32 |
| HGNetv2-Pico | `configs/deimv2/deimv2_hgnetv2_pico_coco.yml` | `weights/deimv2_hgnetv2_pico_coco.pth` | 32 |
| DINOv3-S | `configs/deimv2/deimv2_dinov3_s_coco.yml` | `weights/deimv2_dinov3_s_coco.pth` | 32 |

注：上表按 `W8A8` 训练；N 的 `W8A16` 建议 batch 16（见
`qat/configs/task/deimv2_n_w8a16.json`）。

- **输出契约**：`backbone_encoder_qat.onnx` 输出 `memory [1, N_tokens, hidden_dim]`；
  `decoder_fp32.onnx` 输出 `pred_logits [1, 300, 80]`、`pred_boxes [1, 300, 4]`。
- **checkpoint 契约**：每个 epoch 保存 `qat_state_epoch{epoch:02d}.pt`；
  `qat_state.pt` 是指向最新 epoch 的 hardlink；`qat_state_best.pt` 是新的最佳 epoch
  **真实副本**，不是指针。
- **decoder 训练**：默认冻结；加 `--train-decoder --decoder-lr 1e-6` 可做 FP32 微调，
  decoder 不进入 PT2E 图。此时 checkpoint 额外保存 `decoder` state，导出会自动加载。
- **已支持**：W8A8 / W8A16、regional config 自动发现、W8A16 LSQ 与 Hungarian 匹配蒸馏
  （`--lsq`、`--teacher-weights`，本轮默认不启用）。
- **未接入**：`TopK` 分割的全网络 QAT；当前只实现 backbone + encoder QAT。
- **⚠️ 全量 val2017 评估**：`--eval-every 1` 每个 epoch 都会跑 5000 张，50 epoch 耗时较长；
  中断训练前请保留 `qat_state_best.pt` 和对应 `qat_state_epochXX.pt`。

### 5. 评估精度

#### 检测

FP32 基线按 [`docs/baseline.md`](docs/baseline.md) 的完整命令复现，使用
`train.py --test-only` 和临时 YAML 覆盖 COCO 路径：

```bash
mkdir -p runs/baseline
cat > runs/baseline/deimv2_baseline_s.yml <<EOF
__include__: ['$PWD/configs/deimv2/deimv2_dinov3_s_coco.yml']

output_dir: runs/baseline/deimv2_baseline_s
train_dataloader:
  dataset:
    img_folder: $COCO_ROOT/train2017/
    ann_file: $COCO_ROOT/annotations/instances_train2017.json
val_dataloader:
  dataset:
    img_folder: $COCO_ROOT/val2017/
    ann_file: $COCO_ROOT/annotations/instances_val2017.json
  num_workers: 8
EOF

env PYTHONPATH="$PWD" \
  python train.py \
  -c runs/baseline/deimv2_baseline_s.yml \
  --test-only \
  -r weights/deimv2_dinov3_s_coco.pth \
  -d cuda:0
```

如遇 torchvision 0.21 的 transform API 不兼容，使用
[`docs/baseline.md`](docs/baseline.md) 中的兼容脚本运行同一份 YAML。

QAT fake-quant 的全量评估由训练脚本的 `--eval-every` / `--eval` 完成，结果写入：

```text
<run-dir>/eval_history.jsonl   # 每个评估 epoch 的完整 12 项 COCO 指标
<run-dir>/eval.json            # 最后一次评估结果
```

```bash
# 查看每个 epoch 的 mAP50-95
cat runs/qat/<run-dir>/eval_history.jsonl
```

当前没有独立的 “QAT checkpoint → 全量 mAP” CLI；评估必须复用训练脚本的
`_evaluate_fake_quant`，保证 observer 冻结、decoder FP32、后处理与基线一致。

### 6. 导出 QuantONNX

#### 检测

导出用 FP32 权重构建模型，再用 QAT checkpoint 覆盖 backbone + encoder；输出静态
batch=1 的 `backbone_encoder_qat.onnx` 和 `decoder_fp32.onnx`：

```bash
env PYTHONPATH="$PWD" \
  python -m qat.export_quantonnx \
  --config configs/deimv2/deimv2_dinov3_s_coco.yml \
  --weights weights/deimv2_dinov3_s_coco.pth \
  --run-dir runs/qat/<run-dir> \
  --checkpoint runs/qat/<run-dir>/qat_state_best.pt \
  --quant-config runs/qat/<run-dir>/quant_config_discovered.json \
  --output-dir runs/export/<run-dir>_quantonnx \
  --device cpu --opset 21 --input-size 640 640
```

- `--checkpoint` 可指定 `qat_state_best.pt` 或任意 `qat_state_epochXX.pt`；默认使用
  `run-dir/qat_state.pt`。
- 如果 checkpoint 包含微调后的 `decoder` state，导出会自动加载，并写入
  `decoder_fp32.onnx`。
- 默认使用 `onnxslim`；关闭用 `--no-slim`。
- `export_report.json` 记录 SHA256、Q/DQ 计数、encoder / decoder 的 PyTorch vs ORT 误差，
  以及 `merge_consecutive_qdq` / `insert_upstream_qdq_after_views` 的结果。
- **⚠️ tensor 误差通过不等于全量 mAP 通过**；最终验收必须跑 COCO val2017。

### 7. 验收 QuantONNX

#### 检测

用 `infer_quantonnx` 同时跑 ONNX Runtime 的 encoder + decoder，并与 converted PT2E
参考做 Hungarian 匹配对比：

```bash
env PYTHONPATH="$PWD" \
  python -m qat.infer_quantonnx \
  --config configs/deimv2/deimv2_dinov3_s_coco.yml \
  --checkpoint runs/qat/<run-dir>/qat_state_best.pt \
  --weights weights/deimv2_dinov3_s_coco.pth \
  --onnx-dir runs/export/<run-dir>_quantonnx \
  --image example.jpg \
  --reference-mode converted
```

产物：

```text
runs/export/<run-dir>_quantonnx/
├── backbone_encoder_qat.onnx
├── decoder_fp32.onnx
├── export_report.json
├── joint_output.npz
└── joint_infer_report.json
```

`joint_infer_report.json` 包含：

- `memory`：encoder 输出的逐元素误差；
- `logits` / `boxes`：按 Hungarian 匹配后的误差，避免 TopK query 顺序影响；
- `passed`：`matched_query` 的 logits / boxes `mean_abs` 是否都不超过 `--atol`
  （默认 `5e-3`）。

#### 输出契约

| 图 | 输入 | 输出 |
|---|---|---|
| `backbone_encoder_qat.onnx` | `images [1,3,640,640]` | `memory [1,N_tokens,hidden_dim]` |
| `decoder_fp32.onnx` | `memory [1,N_tokens,hidden_dim]` | `pred_logits [1,300,80]`、`pred_boxes [1,300,4]` |

后处理 `PostProcessor` 在模型图外，保持 FP32；focal 模式在 query × class 上做全局
TopK300。

### 8. 图片验证

#### 检测

只跑 ONNX Runtime，不构建 PyTorch 参考：

```bash
env PYTHONPATH="$PWD" \
  python -m qat.infer_quantonnx \
  --config configs/deimv2/deimv2_dinov3_s_coco.yml \
  --onnx-dir runs/export/<run-dir>_quantonnx \
  --image example.jpg \
  --save-npz runs/export/<run-dir>_quantonnx/example.npz \
  --output runs/export/<run-dir>_quantonnx/example.json
```

`example.npz` 保存 `memory`、`pred_logits`、`pred_boxes`；`example.json` 记录输入尺寸、
输出 shape 和 ONNX 路径。FP32 图片推理可用 `tools/inference/torch_inf.py`，但
torchvision 0.21 下可能需要按 [`docs/baseline.md`](docs/baseline.md) 的兼容脚本处理。

## 模型部署

当前交付格式是 **QuantONNX**：backbone + encoder 为 Q/DQ 图，decoder 为 FP32；
部署端需要支持 `QuantizeLinear` / `DequantizeLinear`（或等价的 Q/DQ 融合）。
ONNX Runtime 的链路验收由 `export_quantonnx.py` + `infer_quantonnx.py` 完成；
AXERA Pulsar2 / 板端部署流程尚未固化，部署前需要确认：

1. 后端是否支持 Q/DQ 图、per-channel 权重和 S8 matmul / softmax；
2. `TopK`、`grid_sample`、`LayerNorm` / `RMSNorm` 是否保持 FP32 或可被后端接受；
3. 板端输出是否与 `pred_logits` / `pred_boxes` 的浮点语义一致。

结构、TopK 分布和量化敏感点见 [`docs/architecture.md`](docs/architecture.md)；
QAT 形态和边界接口见 [`docs/qat-pipeline.md`](docs/qat-pipeline.md)；
实验记录与验收口径见 [`docs/qat-experiment-log.md`](docs/qat-experiment-log.md)。

## 许可证

详见 [`LICENSE.md`](./LICENSE.md)。
