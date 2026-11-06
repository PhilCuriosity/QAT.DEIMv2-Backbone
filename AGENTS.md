# DEIMv2 QAT 工作约定

## 目标

在 DEIMv2 上实现量化感知训练（QAT）。验收标准：

- 覆盖已发布型号：Atto / Femto / Pico / N / S / M / L / X。
- 每个型号 QAT 模型在 COCO val2017（5000 张）上的 `mAP50-95` 与同协议 FP32 基线差距 **< 1.0**。
- 精度对比必须使用同一份 checkpoint、同一套预处理、同一后处理和同一个评估脚本；不得用子集或不同口径替代全量结果。

## 工作区与环境

- 工作目录：仓库根目录（上游 DEIMv2，`main` 分支）。
- 项目解释器：已安装 torch 2.6 的 Python 环境（下文简称“项目 Python”）。
  - Python 3.10.19
  - torch 2.6.0+cu124
  - torchvision 0.21.0+cu124
  - onnx 1.17.0 / onnxruntime 1.21.0
  - faster-coco-eval 1.8.0
  - tensorboard 2.21.0（`engine/core/_config.py` 导入需要）
- GPU：4 × NVIDIA A100-SXM4-80GB。
- COCO 数据根：本地 COCO 数据集目录
  - 目录下应包含 `train2017/`、`val2017/`、`annotations/instances_{train,val}2017.json`。
  - 项目 YAML 中的数据路径可能与本地实际路径不一致；运行时必须覆盖为实际数据根。
- 权重：当前仓库没有 `weights/`、`ckpts/`。需要时从 HuggingFace `Intellindust/DEIMv2_*_COCO`
  下载，并按项目 YAML 建图后转换为 `{'model': state_dict}`。不要直接相信 HF `config.json`（见下方 S 型号注意事项）。
- 参考实现：同级的 `QAT.DEIMv2` 仓库（HGNetv2-N 的 PT2E QAT / QuantONNX / AXERA Pulsar2 经验，可复用思路，但其结论不自动适用于其他型号）。

### 环境注意

- `requirements.txt` 固定 `torch==2.5.1` / `torchvision==0.20.1`，当前环境是 2.6 / 0.21。
- torchvision 0.21 把 `Transform._transform/_get_params/_check_inputs` 改名为
  `transform/make_params/check_inputs`；项目的 `ConvertBoxes`、`ConvertPILImage` 等会报
  `NotImplementedError`。需要兼容层或固定 torchvision 版本。
- QAT 需要 torch 2.6 的 `torch.export` / PT2E；若切回 torchvision 0.20.1，需重新验证兼容性。

环境自检：

```bash
python - <<'EOF'
import torch, torchvision, onnx, onnxruntime, tensorboard
print(torch.__version__, torchvision.__version__, onnx.__version__, onnxruntime.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
EOF
```

## 模型结构

统一拓扑：`backbone → encoder → decoder → postprocessor`，入口在 `engine/deim/deim.py` 的 `DEIM.forward`。

### Backbone

- **HGNetv2**（Atto / Femto / Pico / N，以及未发布权重的 S/M/L/X 配置）
  - `engine/backbone/hgnetv2.py`：`ConvBNAct`（Conv+BN+Act，可选 `LearnableAffineBlock`）、`LightConvBNAct`、CSP 堆叠。
  - `use_lab: True` 的型号含逐通道可学习仿射 `scale * x + bias`。
- **DINOv3STAs**（S / M / L / X）
  - `engine/backbone/dinov3_adapter.py`：DINOv3 ViT（L/X）或蒸馏 ViT-Tiny/Tiny+（S/M）。
  - ViT 含 RoPE、`F.scaled_dot_product_attention`、LayerNorm、GELU MLP。
  - `SpatialPriorModulev2`（STA）用 Conv+SyncBN+GELU+MaxPool 产生细节特征。
  - 三尺度语义特征 + STA 细节特征 concat，经 1×1 Conv + SyncBN 融合。
  - `interaction_indexes` 决定取 ViT 哪几层，必须与权重训练配置一致。

### Encoder

- **HybridEncoder**（N / S / M / L / X）：`engine/deim/hybrid_encoder.py`
  - `input_proj`：1×1 Conv + BN。
  - AIFI：单层 `nn.MultiheadAttention` + LayerNorm + GELU FFN。
  - FPN/PAN：`RepNCSPELAN4`（部分型号）/ `RepNCSPELAN5`，含 `ConvNormLayer_fuse`、
    `VGGBlock`（可重参数化）、`SCDown`、`CSPLayer2`；融合方式 `fuse_op: sum`。
- **LiteEncoder**（Atto / Femto / Pico）：`engine/deim/lite_encoder.py`
  - 1×1 Conv + BN 投影；AvgPool + Conv + BN + SiLU 生成小尺度特征；
  - `GAP_Fusion`：全局平均池化后 `x + gap`，再 1×1 Conv+BN+SiLU；
  - 两个 `RepNCSPELAN4` 分别做 FPN / PAN。

### Decoder

`engine/deim/deim_decoder.py` 的 `DEIMTransformer`：

- 已提供 `encode(feats)` / `decode(memory, spatial_shapes, targets)`，`forward` 由二者组合，
  供 backbone+encoder 部分 QAT 在 decoder 前切分；不改变原始输出。
- 输入投影 + 锚点生成（sigmoid/inverse-sigmoid 域）+ `enc_score_head` / `enc_bbox_head`。
- 训练时加入 contrastive denoising（`num_denoising=100`）；部署图不含 denoising。
- 每层：
  - self-attn：`nn.MultiheadAttention`
  - cross-attn：`MSDeformableAttention`（`engine/deim/dfine_decoder.py`），
    由 `sampling_offsets` / `attention_weights` 线性层 + softmax + `F.grid_sample(bilinear)` 组成。
  - 归一化：`RMSNorm`；FFN：`SwiGLUFFN`（`silu(x1) * x2`）；可选 `Gate`。
- 回归头采用 FDR：`dec_bbox_head` 输出 `4 × (reg_max+1)`（`reg_max=32`），
  `Integral` 做 softmax 期望，`distance2bbox` 解码；`LQE` 用分布 top-k 统计修正分类分数。
- 推理输出：`pred_logits [B, num_queries, 80]`、`pred_boxes [B, num_queries, 4]`。

### PostProcessor

`engine/deim/postprocessor.py`：

- `use_focal_loss=True`：对 `sigmoid(logits)` 在 **query × class** 上做全局 top-k（300），
  同一个 query 可对应多个类别；`labels = index % 80`，`index // 80` 取框。
- 框从 `cxcywh` 转 `xyxy` 并乘原图尺寸。
- 部署模式返回 `(labels, boxes, scores)`；ONNX 导出入口在 `tools/deployment/export_onnx.py`。

## 浮点基线

README 公布的 COCO val2017 `AP`（mAP50-95）：

| 型号 | Backbone | Encoder | Decoder 层数 | 输入 | Params | GFLOPs | FP32 AP |
|---|---|---|---:|---:|---:|---:|---:|
| Atto | HGNetv2-Atto | LiteEncoder | 3 | 320 | 0.5M | 0.8 | 23.8 |
| Femto | HGNetv2-Femto | LiteEncoder | 3 | 416 | 1.0M | 1.7 | 31.0 |
| Pico | HGNetv2-Pico | LiteEncoder | 3 | 640 | 1.5M | 5.2 | 38.5 |
| N | HGNetv2-B0 | HybridEncoder | 3 | 640 | 3.6M | 6.8 | 43.0 |
| S | DINOv3 ViT-Tiny | HybridEncoder | 4 | 640 | 9.7M | 49.5 | 50.9 |
| M | DINOv3 ViT-Tiny+ | HybridEncoder | 4 | 640 | 18.1M | 84.1 | 53.0 |
| L | DINOv3 ViT-S | HybridEncoder | 4 | 640 | 32.2M | 144.5 | 56.0 |
| X | DINOv3 ViT-S+ | HybridEncoder | 6 | 640 | 50.3M | 200.0 | 57.8 |

### 本机已复现（2026-11-03/04）

- N：`42.9824`，与 README 一致。
- Pico：`38.4593`，与 README 一致。
- S：`50.9`，与 README 一致；**必须使用项目 YAML 的 `interaction_indexes: [3, 7, 11]`**。
  - 使用 HuggingFace `config.json` 的 `[5, 8, 11]` 只有 `47.0`。HF config 与发布权重不一致，
    不要直接用 HF config 构建推理/训练图。
- 其余型号尚未在本机复测；QAT 前必须逐型号复现并记录 baseline。

### 基线评估协议

- 数据：COCO val2017 全部 5000 张。
- HGNetv2 型号：`Resize [H,W]` → `/255`，**不做 ImageNet Normalize**。
- DINOv3 型号：`Resize [640,640]` → `/255` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`。
- 评估：项目自带 `CocoEvaluator`（faster-coco-eval）+ `PostProcessor` 全局 top-k 300；
  报告 `coco_eval_bbox[0]`（mAP50-95）。
- 记录：权重 SHA256、config 路径、输入尺寸、GPU、随机种子、命令、完整 12 项 COCO 指标。

## 量化敏感点

1. **Deformable attention**：`grid_sample` 双线性采样、`sampling_offsets`（未归一化坐标）、
   `attention_weights` softmax；部署端可能需要改写采样或 `linear → bilinear`。
2. **TopK**：encoder query 选择、LQE 的 `prob.topk`、postprocessor 全局 top-k。
   索引/排序不能量化；values 的量化域必须与下游一致。
3. **FDR / Integral**：`softmax(reg_max+1)` + 固定投影 `weighting_function`，对量化误差敏感。
4. **Sigmoid / inverse sigmoid / log / exp**：锚点、框解码、gate；注意数值范围与 clamp。
5. **归一化层**：BN / SyncBN 可折叠进 Conv，但 QAT 训练中要保留统计更新；
   LayerNorm/RMSNorm 通常保持浮点。
6. **激活**：SiLU / GELU；逐层量化范围差异大，需要 observer 校准或 LSQ。
7. **ViT attention**：`scaled_dot_product_attention` + softmax + RoPE；RoPE 的 sin/cos 应保持浮点。
8. **Rep 重参数化**：`VGGBlock` / `ConvNormLayer_fuse` 的 `convert_to_deploy` 在 QAT 前后必须与量化图一致。
9. **DINOv3STA**：concat 前后通道域差异大，1×1 Conv + SyncBN 融合处需要逐通道 scale。

## 历史验证结论（W16A16，产物已回退）

2026-11-03/04 曾在 Pico/N 上跑通过完整链路：

`PT2E prepare → observer 校准 → QAT 蒸馏 → fake-quant 全量 eval → convert → QuantONNX → ORT 全量 eval`

全量 val2017 结果（当时产物）：

| 模型 | FP32 | W16A16 PTQ | W16A16 QAT | PTQ ORT | QAT ORT |
|---|---:|---:|---:|---:|---:|
| N | 42.9824 | 42.9528 | 42.9678 | 42.4955 | 42.4903 |
| Pico | 38.4593 | 38.4564 | 38.4756 | 38.2899 | 38.2732 |

所有差距均 < 1。该目录下的代码、权重、checkpoint 和 ONNX 已按要求回退/删除，重新开始时需要重建。

关键经验：

- 不要按输出索引直接做 QAT 蒸馏。encoder TopK 会改变 query 顺序，按索引对齐会破坏模型
  （N 会从 43.0 掉到 6.8）。
- 应使用 **Hungarian 匹配蒸馏**：按 teacher top-k 选查询，再与 student 查询做 L1+IoU 匹配后
  计算 logits/boxes/GIoU loss。
- 导出稳定性需要处理：anchor 的 `log(0)`/`inf`、deformable attention 的 device/dtype 混用、
  BN folding 产生的 `Reciprocal`、torchvision 0.21 transform API。

## 代码与实验约定

- 上游代码尽量少改；QAT 相关新增代码放在独立目录（如 `qat/`），不破坏原始训练入口。
- 所有实验写入唯一目录 `runs/<model>_<quant_config>_<date>/`，不覆盖历史结果。
- 不提交 `runs/`、`ckpts/`、`weights/`、`outputs/`、`*.pth`、`*.onnx`。
- 修改上游代码前先记录 `git diff`；任何“精度达标”结论必须附 FP32 指标、QAT 指标、差值、
  评估命令和 checkpoint SHA256。

## 当前状态（2026-11-04）

- 已按计划重新建立文档、工具和部分权重：
  - `docs/`：architecture / baseline / qat-pipeline / qat-experiment-plan / qat-experiment-log
  - `qat/export_fp32_onnx.py`：浮点全模型与 backbone+encoder ONNX 导出
  - `weights/`：DINOv3-S 权重与 HF safetensors（被 gitignore）
- `qat/` 路线 B 代码已落地：boundaries / pt2e / regional discovery / 训练 CLI / 回归测试。
- 已确认 N / Pico / S 的 FP32 基线；其余型号待复测。
- **首要实验为 W8A16**：backbone+encoder QAT + FP32 decoder，计划见
  `docs/qat-experiment-plan.md`，记录见 `docs/qat-experiment-log.md`。
- 模型变更回归测试：`qat/tests/test_model_change.py` 4 passed。
- N W8A16 backbone+encoder 初步结果（全量 val2017）：
  - PTQ：21.8150（Δ -21.1674）
  - QAT 200 step：31.9316（Δ -11.0508）
  - 尚未达到 <1；200 step 约 3200 张图，远未完成 1 epoch。
- QAT 图曾导出 QuantONNX（backbone+encoder QAT 2.7MB，decoder FP32 4.3MB；
  encoder 899 节点 / 233 Q / 376 DQ）；该次 N W8A16 运行目录已按要求删除，指标保留在文档。
- 联合推理脚本：`qat/infer_quantonnx.py`（默认纯 ONNX Runtime，可选与 converted PT2E 参考对比）。
- QAT 已支持 LSQ 与蒸馏：`--lsq`、`--teacher-weights` + KD 权重；checkpoint 记录 `use_lsq`/`distill`，导出自动识别 LSQ。
- LSQ+KD 1-step smoke 和 LSQ QuantONNX 导出已通过；精度待全量训练验证。
- S W8A8 纯 QAT 首次启动后手动终止（2026-11-05，epoch 0 step 300/3696，无 checkpoint）：
  产物保留在 `runs/qat/deimv2_s_w8a8_qat_20261105_104136/`（含 `STATUS.md`）。
- 下一步：用 `weights/deimv2_dinov3_s_coco.pth` 跑 S 的 W8A8 纯 QAT（50 epoch、batch 32、无 KD/LSQ），再扩展到 Pico。
