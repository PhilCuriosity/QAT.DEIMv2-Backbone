# DEIMv2 模型结构

本文梳理当前仓库的 DEIMv2 模型结构，重点覆盖已发布权重的 8 个型号：

- HGNetv2：Atto / Femto / Pico / N
- DINOv3：S / M / L / X

并标出 TopK、denoising、LQE、FDR 等对 QAT 敏感的位置。

## 1. 总体拓扑

```text
images
  │
  ▼
Backbone
  ├── HGNetv2（Atto / Femto / Pico / N）
  └── DINOv3STAs（S / M / L / X）
  │  多尺度特征
  ▼
Encoder
  ├── LiteEncoder（Atto / Femto / Pico）
  └── HybridEncoder（N / S / M / L / X）
  │  多尺度融合特征
  ▼
DEIMTransformer
  ├── input_proj
  ├── anchor 生成
  ├── enc_score_head / enc_bbox_head
  ├── encoder query TopK
  ├── decoder layers（self-attn + deformable cross-attn + FFN）
  ├── FDR box head / LQE
  └── pred_logits / pred_boxes
  │
  ▼
PostProcessor（sigmoid + 全局 query×class TopK）
  │
  ▼
labels / boxes / scores
```

入口：`engine/deim/deim.py::DEIM.forward`

```python
x = self.backbone(x)      # list[Tensor]
x = self.encoder(x)       # list[Tensor]
x = self.decoder(x, targets)  # dict
```

## 2. 型号配置矩阵

### 2.1 已发布权重

| 型号 | Backbone | Encoder | Decoder 层 | 输入 | Queries | Levels | Params | GFLOPs |
|---|---|---|---|---:|---:|---:|---:|---:|
| Atto | HGNetv2-Atto | LiteEncoder | 3 | 320 | 100 | 2 | 0.5M | 0.8 |
| Femto | HGNetv2-Femto | LiteEncoder | 3 | 416 | 150 | 2 | 1.0M | 1.7 |
| Pico | HGNetv2-Pico | LiteEncoder | 3 | 640 | 200 | 2 | 1.5M | 5.2 |
| N | HGNetv2-B0 | HybridEncoder | 3 | 640 | 300 | 2 | 3.6M | 6.8 |
| S | DINOv3 ViT-Tiny | HybridEncoder | 4 | 640 | 300 | 3 | 9.7M | 49.5 |
| M | DINOv3 ViT-Tiny+ | HybridEncoder | 4 | 640 | 300 | 3 | 18.1M | 84.1 |
| L | DINOv3 ViT-S | HybridEncoder | 4 | 640 | 300 | 3 | 32.2M | 144.5 |
| X | DINOv3 ViT-S+ | HybridEncoder | 6 | 640 | 300 | 3 | 50.3M | 200.0 |

仓库里还有 HGNetv2-S/M/L/X 的 YAML 配置（`deimv2_hgnetv2_{s,m,l,x}_coco.yml`），
但 README 没有对应的已发布 COCO 权重；它们主要用于从 ImageNet/自定义数据训练。

### 2.2 关键差异

| 型号 | hidden_dim | feat_strides | num_points | dim_feedforward | share_bbox_head | use_gateway |
|---|---:|---|---|---:|---|---|
| Atto | 64 | [16, 32] | [4, 2] | 160 | True | False |
| Femto | 96 | [16, 32] | [4, 2] | 256 | True | False |
| Pico | 112 | [16, 32] | [4, 2] | 320 | True | False |
| N | 128 | [16, 32] | [6, 6] | 512 | False | True |
| S | 192 | [8, 16, 32] | [3, 6, 3] | 512 | False | True |
| M | 256 | [8, 16, 32] | [3, 6, 3] | 512 | False | True |
| L | 224 | [8, 16, 32] | [3, 6, 3] | 1792 | False | True |
| X | 256 | [8, 16, 32] | [3, 6, 3] | 2048 | False | True |

所有已发布型号：

- `reg_max = 32`，`reg_scale = 4`
- `num_denoising = 100`（仅训练）
- `eval_idx = -1`
- LQE：`k=4`，`reg_conf` 为 2 层 MLP（隐藏 64）
- `PostProcessor.num_top_queries = 300`

## 3. Backbone

### 3.1 HGNetv2

代码：`engine/backbone/hgnetv2.py`

组成：

- `ConvBNAct`：Conv2d + BatchNorm2d + 激活；`use_lab=True` 时再乘
  `LearnableAffineBlock`（逐通道可学习 `scale * x + bias`）。
- `LightConvBNAct`：1×1 Conv + 3×3 depthwise Conv 组合。
- CSP/ELAN 堆叠，输出多个 stage 特征。
- 默认 `freeze_at=-1`、`freeze_norm=False`，即不冻结。
- YAML 中 `pretrained: True` 会下载 ImageNet 权重；加载检测 checkpoint 时应设为 `False`。

已发布型号：

| 型号 | name | return_idx | 输出 stride |
|---|---|---|---|
| Atto | Atto | [2] | 16 |
| Femto | Femto | [2] | 16 |
| Pico | Pico | [2] | 16 |
| N | B0 | [2, 3] | 16, 32 |
| S（HG 配置） | B0 | [1, 2, 3] | 8, 16, 32 |
| M（HG 配置） | B2 | [1, 2, 3] | 8, 16, 32 |
| L（HG 配置） | B4 | [1, 2, 3] | 8, 16, 32 |
| X（HG 配置） | B5 | [1, 2, 3] | 8, 16, 32 |

> HGNetv2-S/M/L/X 只有 YAML 配置，没有 README 中对应的已发布 COCO 权重。
> 已发布的是 Atto/Femto/Pico/N 四个 HGNetv2 型号。

### 3.2 DINOv3STAs

代码：`engine/backbone/dinov3_adapter.py`、`engine/backbone/vit_tiny.py`

组成：

1. `dinov3`：
   - L/X：真正的 `DinoVisionTransformer`（`engine/backbone/dinov3/`）。
   - S/M：蒸馏的 `VisionTransformer`（ViT-Tiny / Tiny+，`vit_tiny.py`）。
   - ViT 含 PatchEmbed、RoPE、`scaled_dot_product_attention`、LayerNorm、GELU MLP。
2. `SpatialPriorModulev2`（STA）：
   - Conv + SyncBatchNorm + GELU + MaxPool；
   - 产生 1/8、1/16、1/32 的细节特征；
   - `conv_inplane` 控制通道。
3. 融合：
   - 每个尺度的语义特征与 STA 细节特征 concat；
   - 1×1 Conv + SyncBN 投影到 `hidden_dim`。

关键参数：

| 型号 | name | embed_dim | heads | interaction_indexes | conv_inplane | hidden_dim |
|---|---|---:|---:|---|---:|---:|
| S | vit_tiny | 192 | 3 | [3, 7, 11] | 16（默认） | 192 |
| M | vit_tinyplus | 256 | 4 | [3, 7, 11] | 16（默认） | 256 |
| L | dinov3_vits16 | 由模型决定 | - | [5, 8, 11] | 32 | 224 |
| X | dinov3_vits16plus | 由模型决定 | - | [5, 8, 11] | 64 | 256 |

注意：`interaction_indexes` 必须与权重训练配置一致。S 的权重必须用 `[3, 7, 11]`；
HuggingFace `config.json` 的 `[5, 8, 11]` 会掉到 47.0（见 `baseline.md`）。

## 4. Encoder

### 4.1 LiteEncoder（Atto / Femto / Pico）

代码：`engine/deim/lite_encoder.py`

```text
backbone 单尺度特征
  → input_proj（1×1 Conv + BN）
  → down_sample1（AvgPool3×3/s2 + 1×1 Conv + BN + SiLU）
  → GAP_Fusion（x + GAP(x)，再 1×1 Conv + BN + SiLU）
  → FPN：proj[0] + upsample(proj[1]) → RepNCSPELAN4
  → PAN：proj[1] + downsample(outs[0]) → RepNCSPELAN4
```

输出两个尺度：stride 16 和 32。

### 4.2 HybridEncoder（N / S / M / L / X）

代码：`engine/deim/hybrid_encoder.py`

```text
backbone 多尺度特征
  → input_proj（1×1 Conv + BN，通道不等时）
  → AIFI（仅 use_encoder_idx 指定的最高层）
      MultiheadAttention + LayerNorm + GELU FFN
      + 2D sincos position embedding
  → FPN（top-down）：
      lateral 1×1 Conv + nearest upsample + sum
      → RepNCSPELAN4 / RepNCSPELAN5
  → PAN（bottom-up）：
      SCDown（1×1 + depthwise 3×3）+ sum
      → RepNCSPELAN4 / RepNCSPELAN5
```

关键模块：

- `ConvNormLayer_fuse`：Conv + BN，可重参数化；`convert_to_deploy()` 把 BN 融合进 Conv。
- `VGGBlock`：3×3 Conv + 1×1 Conv 双分支，部署时可融合成单个 3×3 Conv。
- `SCDown`：1×1 Conv + depthwise 3×3 Conv，stride 2。
- `RepNCSPELAN4`：N/S/M/L/X 使用 `version: dfine` 时为该类；
- `RepNCSPELAN5`：`version: deim` 时使用。
- 融合方式：DEIMv2 默认 `fuse_op: sum`，不是 concat。

各型号 encoder 参数：

| 型号 | in_channels | hidden_dim | depth_mult | expansion | dim_feedforward | use_encoder_idx | version |
|---|---|---:|---:|---:|---:|---|---|
| N | [512, 1024] | 128 | 0.5 | 0.34 | 512 | [1] | dfine |
| S | [192, 192, 192] | 192 | 0.67 | 0.34 | 512 | [2] | deim |
| M | [256, 256, 256] | 256 | 1.0 | 0.67 | 512 | [2] | deim |
| L | [224, 224, 224] | 224 | 1.0 | 1.0 | 896 | [2] | deim |
| X | [256, 256, 256] | 256 | 1.37 | 1.25 | 1024 | [2] | deim |

`use_encoder_idx` 指向 `in_channels` 的下标：N 只对 stride 32 的特征做 AIFI；
S/M/L/X 对 stride 32 做 AIFI。

## 5. Decoder：DEIMTransformer

代码：`engine/deim/deim_decoder.py`

### 5.1 输入与 query 选择

```text
encoder feats
  → input_proj（1×1 Conv + BN 或 Identity）
  → flatten + concat
  → memory [B, N_tokens, C]
  → valid_mask * memory
  → enc_score_head(memory)      [B, N_tokens, 80]
  → _select_topk                → 300 个 token
  → content = gather(memory, idx)
  → anchors = gather(anchors_unact, idx)
  → enc_bbox_head(content) + anchors → ref_points_unact
  → decoder
```

`N_tokens` 由输入尺寸和 `feat_strides` 决定：

```text
N_tokens = Σ_l (H / stride_l) * (W / stride_l)
```

DEIMv2-N @ 640：

```text
40×40 + 20×20 = 2000
scores:  [1, 2000, 80]
memory:  [1, 2000, 128]
anchors: [1, 2000, 4]
topk:    300
content: [1, 300, 128]
ref:     [1, 300, 4]
```

`query_select_method`：

- `default`（所有已发布型号）：`topk(scores.max(-1).values, num_queries)`
- `one2many`：`topk(scores.flatten(1), ...)`
- `agnostic`：`topk(scores.squeeze(-1), ...)`

### 5.2 Denoising

- 仅训练时启用：配置 `num_denoising=100`。
- `get_contrastive_denoising_training_group` 生成带噪声的 GT query，与 encoder query 拼接。
- 实际 denoising query 数由 batch 内最大 GT 数 `max_gt_num` 和 `num_group` 决定，
  约为 `max_gt_num * 2 * num_group`；配置里的 100 是上限参数，不是固定 query 数。
- 部署图通常关闭 denoising；QAT 导出一般使用 2 输出图，不包含 denoising。

### 5.3 Decoder layer

每层：

```text
x = x + Dropout(SelfAttn(q=x+pos, k=x+pos, v=x))
x = RMSNorm(x)

x = x + Dropout(MSDeformableAttention(x+pos, ref_points, memory, spatial_shapes))
x = Gateway(x, ...)  或  x = RMSNorm(x + ...)

x = x + Dropout(SwiGLUFFN(x))
x = RMSNorm(clamp(x, -65504, 65504))
```

- self-attn：`nn.MultiheadAttention(..., batch_first=True)`
- cross-attn：`MSDeformableAttention`（`engine/deim/dfine_decoder.py`）
  - `sampling_offsets`：`Linear(C, heads * levels * points * 2)`
  - `attention_weights`：`Linear(C, heads * levels * points)`
  - softmax + `F.grid_sample(bilinear)`
  - 每个 head、每个 level、每个 query 采样 `num_points` 个点
- 归一化：`RMSNorm`（`engine/deim/deim_utils.py`）
- FFN：`SwiGLUFFN`：`w12` 输出 2×hidden，`silu(x1) * x2`，再 `w3`
- `use_gateway=True`（N/S/M/L/X）：`Gate` 用 `sigmoid(Linear(cat[x1,x2]))` 做两路门控
- `use_gateway=False`（Atto/Femto/Pico）：残差 + RMSNorm

### 5.4 FDR 与 LQE

每层输出：

```text
pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
             [B, 300, 4*(reg_max+1)] = [B, 300, 132]

Integral:
  softmax(pred_corners.reshape(B, 300, 4, 33), -1)
  × project[0..32]
  → distance [B, 300, 4]

distance2bbox(ref_points, distance, reg_scale)
  → inter_ref_bbox [B, 300, 4]
```

- `reg_max=32`，每个 box edge 有 33 个离散 bin。
- `Integral` 的 `project` 由 `weighting_function(reg_max, up, reg_scale)` 生成；
  `up=0.5`、`reg_scale=4`。
- `LQE`（Location Quality Estimator）：
  ```text
  prob = softmax(pred_corners.reshape(B, 300, 4, 33), -1)
  topk(prob, k=4, dim=-1) → [B, 300, 4, 4]
  + mean → [B, 300, 4, 5]
  → reg_conf MLP → score 修正
  ```
- 最后一层输出 `pred_logits`、`pred_boxes`；训练时还输出 aux/enc/pre/dn 分支。

### 5.5 输出

推理：

```text
pred_logits [B, num_queries, 80]
pred_boxes  [B, num_queries, 4]   # cxcywh, 归一化
```

训练：

```text
pred_logits, pred_boxes, pred_corners, ref_points,
aux_outputs, enc_aux_outputs, pre_outputs,
dn_outputs / dn_pre_outputs（启用 denoising 时）
```

## 6. PostProcessor

代码：`engine/deim/postprocessor.py`

```text
logits [B, 300, 80]
  → sigmoid
  → flatten(1) → topk(300)          # query × class
  → labels = index % 80
  → query_idx = index // 80
  → boxes = gather(pred_boxes_xyxy, query_idx)
  → boxes *= orig_size
```

- `use_focal_loss=True`：全局 query×class TopK，同一个 query 可对应多个类别。
- `num_top_queries=300`。
- 部署模式（`deploy()`）返回 `(labels, boxes, scores)`。
- ONNX 导出入口：`tools/deployment/export_onnx.py`。

## 7. TopK 位置汇总

| # | 位置 | 代码 | 规模 | 是否量化 | 说明 |
|---|---|---|---|---|---|
| T1 | encoder query 选择 | `DEIMTransformer._select_topk` | 2000 → 300（N） | 建议 FP32 | 决定 decoder 使用哪些 token；量化误差会改变 query 顺序 |
| T2 | LQE 分布 TopK | `LQE.forward` | 33 → 4，每个 box edge | 建议 FP32 float island | 只影响分数修正，不影响 box 主路径 |
| T3 | postprocessor 全局 TopK | `PostProcessor.forward` | 300×80 → 300 | 不参与 QAT | 部署后处理 |
| — | matcher one-to-many | `matcher.get_top_k_matches` | 重复 Hungarian | 不导出 | 不是 `torch.topk`，仅训练 |

TopK 的 indices 是整数，不参与 fake-quant；values 的量化策略见
[`qat-pipeline.md`](qat-pipeline.md)。

## 8. 输入预处理

| 系列 | 预处理 |
|---|---|
| HGNetv2（Atto/Femto/Pico/N） | `Resize [H,W]` → `/255`，不做 ImageNet Normalize |
| DINOv3（S/M/L/X） | `Resize [640,640]` → `/255` → `Normalize(mean, std)` |

评估输入尺寸：

| 型号 | 尺寸 |
|---|---:|
| Atto | 320 |
| Femto | 416 |
| Pico | 640 |
| N/S/M/L/X | 640 |

## 9. 量化敏感点

1. **T1 TopK**：pre-topk scores 的量化误差会改变 query 集合；部分 QAT 以它为分割点。
2. **T2 LQE TopK**：softmax 分布 + topk values；建议 float island。
3. **Deformable attention**：`grid_sample`、`sampling_offsets`（未归一化）、softmax。
4. **FDR / Integral**：33 类 softmax + 固定投影，对 logits 量化敏感。
5. **Sigmoid / inverse sigmoid / log / exp**：anchor、框解码、gate。
6. **BN / SyncBN**：可折叠进 Conv；QAT 中保留 running stats。
7. **LayerNorm / RMSNorm**：通常保持浮点。
8. **SiLU / GELU / SwiGLU**：逐层范围差异大。
9. **ViT attention / RoPE**：RoPE 的 sin/cos 保持浮点。
10. **Rep 重参数化**：`VGGBlock` / `ConvNormLayer_fuse` 的 deploy 融合要与量化图一致。

## 10. 代码索引

| 功能 | 文件 |
|---|---|
| 总入口 | `engine/deim/deim.py` |
| Backbone HGNetv2 | `engine/backbone/hgnetv2.py` |
| Backbone DINOv3STA | `engine/backbone/dinov3_adapter.py` |
| ViT-Tiny/Tiny+ | `engine/backbone/vit_tiny.py` |
| HybridEncoder | `engine/deim/hybrid_encoder.py` |
| LiteEncoder | `engine/deim/lite_encoder.py` |
| DEIMTransformer / decoder | `engine/deim/deim_decoder.py` |
| MSDeformableAttention / LQE / Integral | `engine/deim/dfine_decoder.py` |
| FDR 工具 | `engine/deim/dfine_utils.py` |
| RMSNorm / SwiGLU / Gate | `engine/deim/deim_utils.py` |
| PostProcessor | `engine/deim/postprocessor.py` |
| Matcher | `engine/deim/matcher.py` |
| Criterion | `engine/deim/deim_criterion.py` |
| ONNX 导出 | `tools/deployment/export_onnx.py` |
