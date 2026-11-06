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

#### 3.2.1 RoPE 与 Neg 的导出改动（2026-11-05）

`vit_tiny.py` 中的 RoPE 做了两处与导出/量化相关的调整：

1. **`rotate_half` 去掉 `Neg`**
   - 原实现 `torch.cat((-x2, x1), dim=-1)` 在 ONNX 中产生 24 个 `Neg`（12 层 × q/k），
     且输入、输出都会被量化。
   - 改为 `torch.cat((x2 * -1, x1), dim=-1)`，导出为 `Mul`；`-1` 是 Python 标量，
     不会生成 Q/DQ。数值完全等价。
2. **RoPE 表预计算（导出前折叠）**
   - `RopePositionEmbedding.precompute(H, W)` 把 sin/cos 表写入非持久 buffer
     `_sin_table` / `_cos_table`；`forward` 在 `(H, W)` 命中时直接返回，
     否则回退到原来的 `arange → meshgrid → angles → sin/cos` 路径。
   - `VisionTransformer.precompute_rope(image_size)`、
     `DINOv3STAs.precompute_rope(image_size)` 提供入口；
     `qat/export.py::export_backbone_encoder` 和 `qat/export_fp32_onnx.py`
     在导出前调用。
   - 效果：
     - 导出图不再包含 `arange/meshgrid/.../Sin/Cos`，表变成常量
       （640 输入 → `[1,1,1600,64]`）。
     - W8A8 QuantONNX 中两张表是 2 个 uint8 initializer + 2 个 `DequantizeLinear`，
       中间链路不再插 Q/DQ。
     - 数值与现算逐位一致（max diff 0）；FP32 全量 val2017 mAP50-95 = 0.5086
       （基线 0.509）。
   - 注意：表按导出分辨率固化；当前导出图本来就被 `encoder.pos_embed2` 固定为 640。
     训练时若 H/W 不一致会自动回退动态计算。

#### 3.2.2 Attention：默认 SDPA，导出时切显式（2026-11-05）

`vit_tiny.py` 的 `Attention.forward` 默认使用融合的
`F.scaled_dot_product_attention`。该算子在进行 PT2E 注解时是单个 FX 节点，
而 `AXQuantizer.OPS` 里没有 `sdpa`，因此 attention 内部
（`QKᵀ` / `softmax` / `AV`）不会插入任何 Q/DQ。

处理方式：

- `Attention.explicit_attention`（类属性，默认 `False`）控制是否走显式路径：
  `q * √scale`、`k * √scale` → `q @ kᵀ` → `softmax` → `attn @ v`
  （scale 拆到 q/k 两侧，与融合 SDPA 的内部顺序一致）。
- `qat/export.py::export_backbone_encoder` 在 `torch.export` 期间把所有
  `Attention.explicit_attention` 置为 `True`，导出结束后恢复原值。
  因此：
  - eager FP32 评测仍走 SDPA，数值与基线逐位一致（实测 diff = 0）；
  - QAT 图（prepare / convert / QuantONNX）里是显式 `matmul/softmax`，
    可以正常插 Q/DQ。

数值差异（单条随机输入，对比 SDPA eager）：

| 输出 | CPU max_abs | CPU 相对误差 | CUDA max_abs | CUDA 相对误差 |
|---|---:|---:|---:|---:|
| backbone+encoder `memory` | 1.1e-04 | 2.4e-06 | 3.3e-02 | 4.9e-04 |
| `pred_logits` | 4.0e-02 | 4.1e-05 | 4.3 | 3.0e-02 |
| `pred_boxes` | 6.9e-04 | 1.6e-05 | 0.87 | cos 0.93 |

- `torch.export` 抓出的显式图与 eager 显式实现**逐位一致**（0 差异），
  说明导出本身无误差；与 SDPA 的差异全部来自融合 kernel 与显式
  `matmul/softmax` 的累加顺序（把 scale 拆到 q/k 两侧并不能消除）。
- 单层 attention 的差异约 8e-7，经 12 层 ViT 与 decoder TopK 放大到上表量级。
- CUDA 上差异经 12 层 ViT 和 decoder TopK 放大；QAT 训练使用显式图，
  量化误差远大于该差异。
- 该开关只覆盖 `vit_tiny.py` 的 Attention（S/M）。L/X 使用
  `engine/backbone/dinov3/layers/attention.py`，仍为 SDPA，未纳入。

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

`DEIMTransformer.forward` 已拆成 `encode(feats)` 与
`decode(memory, spatial_shapes, targets)`：`encode` 包含 input projection/flatten/concat，
`decode` 从 denoising 开始。该拆分供 backbone+encoder 部分 QAT 在 decoder 前切分，
输出与原始 `forward` 一致。

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
9. **ViT attention / RoPE**：RoPE 的 sin/cos 表在导出前预计算为常量
   （`precompute_rope`）；W8A8 下按常量量化一次（2 个 uint8 initializer + 2 个 DQ），
   中间角度计算不再有 Q/DQ。`rotate_half` 用 `x * -1` 避免 ONNX `Neg`。
   Attention 默认 SDPA（`sdpa` 不在 `OPS`，内部无 Q/DQ）；QAT 导出时通过
   `explicit_attention` 切成显式 matmul/softmax，才可注解（见 §3.2.2）。
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
| ONNX 导出（上游） | `tools/deployment/export_onnx.py` |
| QAT 边界（backbone+encoder / frozen decoder） | `qat/boundaries.py` |
| PT2E prepare / convert | `qat/pt2e.py` |
| regional 量化区域自动发现 | `qat/regional.py` |
| QAT 训练入口 | `qat/train_backbone_encoder.py` |
| QAT 图导出 | `qat/export.py` |
| QuantONNX 导出（含 QDQ 合并） | `qat/export_quantonnx.py` |
| FP32 ONNX 导出 | `qat/export_fp32_onnx.py` |
| AXERA PT2E quantizer（vendored） | `utils/ax_quantizer.py`, `utils/ax_quantizer_utils.py`, `utils/ax_quantizer_lsq.py` |

## 11. QAT 工具链与导出改动（2026-11-05）

本轮为路线 B（backbone + encoder QAT、decoder 冻结 FP32）落地的改动。每一项都列出
所在文件和它解决的问题。

### 11.1 模型侧（导出/量化相关）

| 改动 | 文件 | 解决的问题 |
|---|---|---|
| `RopePositionEmbedding.precompute(H, W)`，sin/cos 写入非持久 buffer | `engine/backbone/vit_tiny.py` | 原实现每次前向现算 `arange → meshgrid → angles → Sin/Cos`，ONNX 里保留 1 个 `Sin`、1 个 `Cos` 和整条角度计算链，中间每一步都被插 Q/DQ，量化误差直接进入 RoPE 表；部分 NPU 也不支持三角函数。折叠后表变成常量（640 输入 → 2 个 `[1,1,1600,64]` uint8 + 2 个 DQ），中间链路无 Q/DQ |
| `rotate_half` 用 `x2 * -1` 替代 `-x2` | `engine/backbone/vit_tiny.py` | 原写法导出 24 个 `Neg`（12 层 × q/k），且输入/输出都被量化。改成 `Mul` 后无 `Neg` 节点，`-1` 是 Python 标量不会生成 Q/DQ，数值完全等价 |
| `Attention.explicit_attention`（默认 False，导出时置 True） | `engine/backbone/vit_tiny.py`、`qat/export.py` | `AXQuantizer.OPS` 没有 `sdpa`，12 层 ViT 的 `QKᵀ → softmax → AV` 整段无法注解，ONNX 里是浮点孤岛。导出时切显式 `q@kᵀ → softmax → @v` 后可插 Q/DQ；eager FP32 仍走 SDPA，实测与原基线 diff = 0 |

### 11.2 QAT 工具链（`qat/`）

| 改动 | 文件 | 解决的问题 |
|---|---|---|
| 路线 B 边界：`BackboneEncoderQAT` + `FrozenDecoder` | `qat/boundaries.py` | 只在 backbone+encoder 上 QAT，decoder 保持 FP32；避开 deformable attention、FDR/Integral、decoder TopK 的量化敏感点 |
| PT2E prepare/convert 封装与清理（`_remove_empty_annotations`、`_repair_dangling_shared_qspecs`、`_rank_one_guards`） | `qat/pt2e.py` | DINOv3-S 的 concat/共享 qspec 图在 `prepare_qat_pt2e` 报 `cat_1` 缺 observer、`KeyError: conv2d`；清理悬空共享引用后 S 可正常 prepare |
| `share_qparam`（concat 输入/输出共享 qparams） | `qat/configs/quant/w8a8.json` | 早期 `share_qparam=true` 在 S 的 concat 图报 `cat_1` 缺 observer；`_repair_dangling_shared_qspecs` 加入后可正常 prepare。当前 W8A8 使用 `true`，只影响 concat（S smoke 中 53/56 个 concat 共享同一组 qparams） |
| 自动 regional discovery + 删除 softmax FP32 覆盖 | `qat/regional.py` | 硬编码节点名换模型即失效；原 softmax FP32 策略只生效一半（输入浮点、输出仍 U8），现改为走全局 U8，softmax 输入/输出都有 Q/DQ |
| `--eval-every` / `--eval-interval` + `eval_history.jsonl` | `qat/train_backbone_encoder.py` | 50 epoch 训练过程中看不到精度变化；现在每个 epoch 可做全量 val2017 并逐行记录 |
| `--train-max-batches` / `--eval-max-batches` | `qat/train_backbone_encoder.py` | 冒烟测试跑完整 train2017（118k）和 val2017（5k）太慢；可限制 batch 数做子集验证 |
| 运行元数据（`train_args.json`、`command.txt`、`config_snapshot/`、`environment.json`） | `qat/train_backbone_encoder.py` | 实验不可复现；现在命令、参数、配置快照、环境与 git commit 都随 run 保存 |
| `_share_dyadic_inputs` / `_share_dyadic_input_qspec` / `_shared_qspec_dtype` | `utils/ax_quantizer_utils.py`；在 `utils/ax_quantizer.py`、`utils/ax_quantizer_lsq.py` 的 regional 之后再次调用 | 相邻被注解算子各自建 observer，converted 图里出现 44 处 `Q→DQ→Q→DQ`。把 consumer 输入改成 `SharedQuantizationSpec(producer)` 后，converted 图连续对从 44 降到 0；regional 配置会覆盖 qspec，所以之后要再共享一次 |
| `_annotate_cat` 同时匹配 `torch.cat` / `torch.concat` | `utils/ax_quantizer_utils.py` | `_get_encoder_input` 用的是 `torch.concat`（`aten.concat.default`），原 annotator 只匹配 `torch.cat` / `aten.cat.default`，导致输出 `memory` 前的最后一个 concat 完全没有注解、没有 Q/DQ。现在两种算子都会注解，`memory` 前是 `Concat → Q → DQ` |
| `_annotate_matmul` 增加 `aten.bmm.default` | `utils/ax_quantizer_utils.py` | AIFI 的 `nn.MultiheadAttention` 在 FX 里是 `aten.bmm`，原 annotator 只处理 `aten.matmul`，所以 AIFI 的 QKᵀ/AV 没有 output qspec，ONNX 里 AIFI 的 AV MatMul 后没有 Q/DQ。加上后链路为 `MatMul → Q → DQ → Transpose → Q → DQ → Reshape → Q → DQ → Gemm`，三对 Q/DQ 全部保留，无需导出阶段补回 |

### 11.3 导出代码

| 改动 | 文件 | 解决的问题 |
|---|---|---|
| 导出前调用 `precompute_rope` | `qat/export.py::export_backbone_encoder`、`qat/export_fp32_onnx.py` | 不预热的话 `torch.export` 仍会抓到现算的 `Sin/Cos`；预热后表是 buffer，导出为常量 |
| 导出时切显式 attention 并恢复标志 | `qat/export.py::export_backbone_encoder` | 让 PT2E 能注解 attention；导出结束恢复 `explicit_attention=False`，eager 模型不受影响 |
| `merge_consecutive_qdq`（保留反量化范围较大的一对；支持 `DQ1` 多消费者） | `qat/export_quantonnx.py` | ONNX 导出器会为每个被注解的边物化一对 Q/DQ，即使 qspec 共享也会出现 `Q→DQ→Q→DQ`；`DQ` 有多个消费者时（残差 + Slice）也要能合并。当前 S smoke 导出合并 47 对后连续对为 0，最终 2458 节点 / Q 695 / DQ 819；相同 qparams 下 ORT 输出 diff = 0 |
| `export_report.json` 增加 `merge_qdq` 统计 | `qat/export_quantonnx.py` | 便于确认合并生效与记录 SHA256、Q/DQ 计数、ORT 对比误差 |

### 11.4 当前验证状态

- FP32 全量 val2017：`0.5086`（基线 `0.509`），RoPE 折叠与显式 attention 均无精度损失
- S W8A8 smoke（2 step、1 校准 batch）：图结构验证通过；mAP 无参考意义
- QuantONNX：`Neg/Sin/Cos = 0`，RoPE 表 2 个 uint8 + 2 DQ，连续 QDQ 对 0
- 输出 `memory` 前：`Concat → Q → DQ`（`torch.concat` 注解生效）
- AIFI attention：`MatMul → Q → DQ → Transpose → Q → DQ → Reshape → Q → DQ → Gemm`
  （`aten.bmm` 注解生效，三对 Q/DQ 都保留）
- `export_report` 的 `passed: false` 是 ORT vs converted PT2E 的执行一致性（mean_abs ~0.08），
  不是精度结论；FP32 参考导出的 ORT 验证是 passed


### 11.5 ONNX vs PT2E 差异诊断（2026-11-06）

**现象**：`verify_encoder` 对比 converted PT2E 与 ONNX Runtime，S W8A8 的
`mean_abs` 约 0.069、`max_abs` 约 3.5，远大于 N W8A16 的 2.4e-4。

**不是导出 pass 造成的**：对同一 checkpoint/输入分阶段导出：

```text
raw     mean=8.949e-02
opt     mean=8.949e-02
slim    mean=8.949e-02
merge   mean=8.949e-02
final   mean=8.949e-02
```

`merge_consecutive_qdq`、`insert_upstream_qdq_after_views`、Identity 插入都没有改变误差；
原始 `torch.onnx.export` 输出已经是这个差距。

**定位**：用 epoch 10 checkpoint，对 prepared fake-quant 和 converted Q/DQ 逐边对比
（按 compute 节点名匹配，输入名一致才计入）：

```text
compared edges: 286
mismatched:     285
```

第一个明确的分叉点在 **S8 softmax 输出**：

| | prepared fake-quant | converted Q |
|---|---|---|
| dtype | int8 | int8 |
| qmin/qmax | -127 / 127 | -127 / 127 |
| qscheme 属性 | `per_tensor_symmetric` | `per_tensor_symmetric` |
| scale | **0.0039297** | **0.0078594** |
| zero_point | **-127** | **0** |

softmax 输出范围 [0,1]：

- affine int8：`scale = 1/254 ≈ 0.003937`，`zp = -127`
- symmetric int8：`scale = 1/127 ≈ 0.007874`，`zp = 0`

两者网格差 2 倍。误差从 softmax 传到 attention/FFN，放大为全图差异
（silu/gelu max diff 1.5–7.7，matmul 最高 20）。

**根因**：`utils/ax_quantizer.py::_build_act_fake_quant` 对 `moving_avg` 一律使用
`FusedMovingAvgObsFakeQuantize`。该模块的 forward 走
`torch.fused_moving_avg_obs_fake_quant`，按 min/max 计算 **affine** scale/zp，
不理会 `QuantizationSpec` 的 `per_tensor_symmetric`；而 `convert_pt2e` 按 qspec 的
symmetric qscheme 生成 Q/DQ，于是 prepared 与 converted 不一致。

全局 U8 激活是 `is_symmetric=false`，fused 的 affine 行为与 convert 一致，所以不会
暴露该问题；只有 regional S8（matmul/softmax）这类 symmetric qspec 会触发。

**影响**：当前 50 epoch QAT 训练用 affine int8 softmax fake-quant，导出会变成
symmetric int8 softmax，训练图与导出图不一致；ONNX 与 PT2E 的差距主要来自这里。

**修复方向**（尚未实施）：

1. **symmetric qspec 使用非 fused `FakeQuantize`**（推荐）
   - `FakeQuantize` 按 observer 的 symmetric qscheme 计算 `zp=0`，与 convert 一致；
   - 全局 U8 非对称仍走 fused，不受影响；
   - 需要修改 `_build_act_fake_quant` 按 qscheme 选择 fused/非 fused。
2. S8 regional 改为 affine（`is_symmetric=false`，`qmin=-128/qmax=127`），
   prepared 与 convert 都走 affine；但 int8 matmul 的 zp≠0，需确认后端是否接受。
3. 去掉 softmax 的 S8 区域，回到全局 U8；会破坏 attention 内的 S8 连续性。

修复后需要重新验证：prepared softmax 与 converted softmax 的 scale/zp 完全一致，
再重跑 QAT。

### 11.6 qscheme 修复与 ONNX 残差（2026-11-06）

**修复实施**

- `utils/ax_quantizer.py::_build_act_fake_quant` 增加 `qscheme` 参数：
  `per_tensor_symmetric` 返回非 fused `FakeQuantize`；全局 U8 affine 仍走
  `FusedMovingAvgObsFakeQuantize`。
- `_build_weight_fake_quant` 对 `per_channel_symmetric` 同样改用非 fused
  `FakeQuantize`。实测 fused kernel 的 per-channel symmetric scale 带有约
  `255/254`（0.39%）的偏差，虽然 zero_point 通常仍为 0，但与
  `convert_pt2e` 的 `calculate_qparams()` 不完全一致。
- 调用点传入 `act_qscheme` / `output_qscheme` / `weight_qscheme`。

**验证（fresh prepare + calibrate 2 batch）**

```text
softmax prepared: scale 0.0077223, zp 0, FakeQuantize
softmax converted: scale 0.0077223, zp 0
weight prepared: cached scale == calculate_qparams()，完全一致
```

抽样 286 条激活边，qparams 只有 1 条不一致（修复前是 285 条值不一致）。

**仍未解释的 tensor 差异**

```text
prepared vs converted : mean 0.1185, max 2.80, cos 0.870
converted vs ORT ALL  : mean 0.1335, max 3.08
converted vs ORT BASIC: mean 0.1199
converted vs ORT OFF  : mean 0.1180
```

首个分叉点是 `conv2d` 的 1.3e-5（PyTorch 与 ORT 的卷积实现/累加顺序不同），
随后在 Q/DQ 取整处产生 1 个量化步长的翻转，再经 attention/LayerNorm 逐层放大。
`ORT_ENABLE_ALL` 与 `ORT_DISABLE_ALL` 之间也有 0.125 的差异，说明 ORT 的
QDQ/Conv 融合会额外改变数值。

**结论**：qscheme/qparam 不一致已经修复；剩余差异不是权重加载、量化配置或
导出 pass 错误，而是 1e-5 级浮点差异经过不连续量化后的混沌放大。后续验收应
比较修复后重训模型的全量 COCO mAP（PT2E vs ONNX），不应要求 tensor 逐点相等。

### 11.7 W8A16 U16 fused 问题（待处理，2026-11-06）

N W8A16 smoke（`<smoke-run-dir>/qat_state_epoch00.pt`）：

```text
prepared cached scale:        6.1e-5
observer calculate_qparams:   2.44e-4
194 条激活边中 185 条 qparams 不一致
prepared vs converted:        mean 4.44e-2, max 2.13e-1, cos 0.173
converted vs ONNX:            mean 2.29e-4, max 1.71e-3, passed
```

`FusedMovingAvgObsFakeQuantize` 对 U16 的 cached scale 与
`MovingAverageMinMaxObserver.calculate_qparams()` 不一致（后者被
`eps=2^-12` 截断）。导出 converted PT2E → ONNX 仍然忠实，但 prepared QAT
（训练图）与 converted（部署图）不一致。

**暂不处理**：当前 W8A8 不使用 U16 路径。后续修复方向：U16 激活改用
非 fused `FakeQuantize`，或为 16-bit 设置更小的 `eps`，然后重新验证 W8A16。

### 11.8 关闭 fake quant 的 API 陷阱（2026-11-06）

对比 `prepared`（QAT fake-quant 图）和 FP32 时，误用了：

```python
torch.ao.quantization.disable_fake_quant(prepared)
```

该函数只对传入的 module 本身生效，**不会递归到子模块**。PT2E 的
`prepared` 是一个 GraphModule，fake-quant 都在子模块里，因此这次调用
实际没有关闭任何 fake quant：

```text
before: fake_quant modules 624, enabled 624
after : fake_quant modules 624, enabled 624
```

结果导致 “FQ on” 和 “FQ off” 的输出完全相同（mean 0.1819），容易误判为
observer/权重/图有问题。

正确写法：

```python
prepared.apply(torch.ao.quantization.disable_fake_quant)
```

或显式遍历子模块：

```python
for module in prepared.modules():
    if hasattr(module, "disable_fake_quant"):
        module.disable_fake_quant()
```

修正后的对比（S W8A8 smoke，同一 checkpoint / 输入）：

| 配置 | mean_abs | max_abs | cos |
|---|---:|---:|---:|
| FP32 vs prepared，FQ on，observer off | 0.1819 | 7.08 | 0.6543 |
| FP32 vs prepared，FQ off，observer off | **2.6e-7** | 1.9e-4 | **1.0000** |
| prepared FQ on vs FQ off | 0.1819 | 7.08 | 0.6543 |

同时确认：

```text
fake-quant 模块数: 624
observer_enabled=1: 0
前向前后 scale/zp 不变
```

即 observer 已正确关闭；差距全部来自 W8A8 fake quantization，而不是
observer、权重加载或图结构。`qat/pt2e.py::disable_observers` 是显式遍历
子模块的，所以 observer 的关闭是有效的；只有直接调用
`torch.ao.quantization.disable_fake_quant(prepared)` 这种方式有递归陷阱。
