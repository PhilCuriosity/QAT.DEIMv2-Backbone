# QAT Pipeline 设计

本文设计 DEIMv2 的两种 QAT 形态：

1. **全网络 QAT**：整个检测网络作为一张 PT2E 图做量化感知训练。
2. **部分 QAT**：以 TopK 算子为分割点，把网络拆成多个子图分别 QAT；TopK 本身不参与量化。

本文以 **DEIMv2 HGNetv2-N / COCO** 为主要实例，其他型号按相同原则适配。当前仓库已回退，
本文是设计文档；历史 W16A16 验证结论见 [`../AGENTS.md`](../AGENTS.md)。

## 1. 目标

- 每个型号的量化模型在全量 COCO val2017 上 `mAP50-95` 与 FP32 基线差距 < 1.0。
- 支持两种 QAT 形态，便于对比精度、调试量化误差和适配部署后端：
  - 全网络 QAT：实现简单，端到端误差最小。
  - 部分 QAT：把离散/敏感的 TopK 隔离开，允许分段量化、混合精度和分段部署。
- 评估必须使用同一份预处理、后处理、全量数据；TopK 是否量化、边界张量精度都必须记录。

## 2. DEIMv2-N 的 TopK 分布

DEIMv2-N 的推理路径（640×640 输入）：

- HGNetv2-B0，`return_idx: [2, 3]`
- HybridEncoder，`hidden_dim=128`，两个尺度 `[16, 32]`
- 两个尺度展平后共 **2000 个 token**：`40×40 + 20×20`
- `DEIMTransformer`：`num_queries=300`，3 层 decoder，`reg_max=32`，`nhead=8`
- LQE：`k=4`，在 `reg_max+1=33` 个分布 bin 上做 TopK

TopK 出现位置：

| # | 位置 | 代码 | 输入 | 输出 | 是否在推理图 |
|---|---|---|---|---|---|
| T1 | encoder query 选择 | `DEIMTransformer._select_topk` | `enc_score_head` 的 `[B,2000,80]` logits | 300 个 token 索引 | 是 |
| T2 | LQE 分布 TopK | `LQE.forward` | `[B,300,4,33]` softmax 概率 | 每个 box edge 的 4 个最大概率 | 是 |
| T3 | postprocessor 全局 TopK | `PostProcessor.forward` | `[B,300,80]` sigmoid scores | 300 个 query×class | 是 |

注意：matcher 的 `get_top_k_matches` 是重复 Hungarian 匹配，不是 `torch.topk`；它只在训练中使用，
不进入推理图，因此不作为分割点。

T1 是最关键的离散点：它决定 decoder 使用哪些 query。量化误差导致 T1 选择变化时，query 顺序
会整体改变；历史验证中按输出索引直接蒸馏曾使 N 从 43.0 掉到 6.8。

T2 只影响 LQE 的质量分数，不决定 query 集合，但会影响最终分数排序。
T3 是部署后处理的一部分，不属于 QAT 训练图。

## 3. 形态 A：全网络 QAT

### 3.1 图结构

```text
images
  │
  ▼
┌──────────────────────────────────────────────┐
│ 单个 PT2E QAT 图                              │
│ backbone → encoder → enc heads → TopK(T1)    │
│        → decoder + LQE(T2) → logits/boxes    │
└──────────────────────────────────────────────┘
  │
  ▼
PostProcessor TopK(T3)（浮点，不参与 QAT）
```

- 只 prepare/convert 一张图。
- 训练时通常关闭 denoising，导出 2 输出：`pred_logits [B,300,80]`、`pred_boxes [B,300,4]`。
- T1/T2 的 **indices 永远不量化**；values 和输入建议显式设为 FP32（见第 5 节），
  避免 TopK 选择被 fake-quant 改变。

### 3.2 训练

两种可选损失：

1. **原 criterion**：需要保留 aux/denoising 输出的训练图，按 DEIM 原始 matcher + loss 训练。
2. **蒸馏**：2 输出图 + FP32 teacher。不要按输出索引直接对齐；使用 Hungarian 匹配：
   - teacher top-k 选查询；
   - 与 student 查询做 `L1(box) + (1-IoU)` 匹配；
   - 对匹配对计算 logits MSE、box L1、GIoU loss。

历史实现使用第 2 种，W16A16 结果满足 < 1。

### 3.3 优点与风险

- 优点：单图、端到端、边界少、部署简单。
- 风险：T1/T2 附近仍是量化敏感区；量化配置需要显式把 TopK 设为 FP32。
- 适合作为精度上限和部分 QAT 的对照。

## 4. 形态 B：部分 QAT（TopK 分割）

### 4.1 默认两段式

以 T1 为分割点：

```text
images
  │
  ▼
┌──────────────────────────────────────────────┐
│ Segment A（QAT）                              │
│ backbone → encoder → input_proj              │
│ → enc_score_head                             │
└──────────────────────────────────────────────┘
  │ pre_topk_scores [B,2000,80]
  │ memory          [B,2000,128]
  │ anchors_unact   [B,2000,4]
  ▼
┌──────────────────────────────────────────────┐
│ TopK T1 + gather（FP32，不参与 QAT）           │
│ idx = topk(max_c(scores), 300)               │
│ content = gather(memory, idx)                │
│ ref_anchor = gather(anchors, idx)            │
└──────────────────────────────────────────────┘
  │ content [B,300,128], ref_anchor [B,300,4]
  │ memory 仍作为 decoder cross-attention 的 value
  ▼
┌──────────────────────────────────────────────┐
│ Segment B（QAT）                              │
│ enc_bbox_head → decoder 3 层 + LQE           │
│ → pred_logits / pred_boxes                   │
└──────────────────────────────────────────────┘
  │
  ▼
PostProcessor TopK T3（FP32，不参与 QAT）
```

关键点：

- Segment A 不包含 `enc_bbox_head`，只输出原始 `memory`、`anchors`、`pre_topk_scores`；
  这样 A 的输出与 TopK 选择无关。
- `enc_bbox_head` 放到 Segment B，在 TopK 之后对 `content` 计算 `ref_points_unact`。
- TopK + gather 是纯 FP32 边界；其输入是 A 输出反量化后的浮点张量。
- Segment B 仍需要完整 `memory [B,2000,128]` 做 deformable cross-attention；
  `content` 只是 query 初值。

### 4.2 边界接口

DEIMv2-N 的具体 shape（B=1，640×640）：

| 张量 | Shape | 说明 |
|---|---|---|
| `pre_topk_scores` | `[1, 2000, 80]` | `enc_score_head` 输出，TopK 输入 |
| `memory` | `[1, 2000, 128]` | encoder 输出（已乘 valid mask） |
| `anchors_unact` | `[1, 2000, 4]` | logit 域 anchor |
| `topk_indices` | `[1, 300]` | int64，不量化 |
| `content` | `[1, 300, 128]` | gather 后的 query 初值 |
| `ref_anchor` | `[1, 300, 4]` | gather 后的 anchor |
| `spatial_shapes` | `[[40,40],[20,20]]` | 常量，供 decoder 拆分 memory |
| `pred_logits` | `[1, 300, 80]` | B 输出 |
| `pred_boxes` | `[1, 300, 4]` | B 输出 |

边界张量必须是 **FP32**；如果部署后端要求 FP16，需要单独验证 TopK 选择集合是否变化。

### 4.3 T2：LQE TopK

LQE 位于每个 decoder layer 内：

```text
pred_corners [B,300,4,33]
  → softmax
  → TopK(k=4)          # 不参与 QAT
  → topk_values [B,300,4,4] + mean
  → reg_conf MLP
  → score 修正
```

建议：

- **不要把 LQE TopK 当作子图分割点**。它嵌在每层 decoder 中，拆图会破坏层间数据流。
- 把它作为图内的 **FP32 float island**：TopK 不注解，输入 `prob`、输出 `topk_values` 保持浮点；
  `reg_conf` 可以正常 QAT（输入 Q/DQ）。
- 如果后端无法在量化图中插入 FP32 TopK，则把整个 LQE 分数分支保持浮点，只量化主 decoder。

### 4.4 T3：postprocessor TopK

- 不属于训练图；QAT 不涉及。
- 部署图中位于模型输出之后，保持 FP32。
- 输出 `pred_logits`/`pred_boxes` 可以是 Q/DQ 后的反量化结果。

### 4.5 可选更细粒度分割

如果部署或调试需要，可在 T2 处再拆：

```text
Segment B1：decoder 到 pred_corners / prob
TopK T2（FP32）
Segment B2：reg_conf + 后续层
```

但不推荐作为默认方案：

- 每层都有 T2，会产生大量小图；
- LQE 只修正分数，对 box 主路径影响小；
- 图边界和 float 张量增多，部署收益通常小于复杂度。

默认采用 **两段式（T1 分割）+ T2/T3 float island**。

## 5. 量化配置

### 5.1 全局

首要目标：**W8A16（U16 激活 / S8 权重）**。

```json
{
  "global_config": {
    "is_symmetric": false,
    "act_observer": "moving_avg",
    "weight_observer": "moving_avg_per_channel",
    "input":  {"dtype": "U16", "qmin": 0, "qmax": 65535},
    "weight": {"dtype": "S8", "qmin": -127, "qmax": 127},
    "share_qparam": true
  },
  "regional_configs": []
}
```

W16A16（U16 / S16）作为回退对照：

```json
"weight": {"dtype": "S16", "qmin": -32767, "qmax": 32767}
```

W8A16 的权重只有 8-bit，通常需要 QAT；必要时对权重 scale 使用 LSQ。若 W8A16 在
`backbone+encoder QAT + FP32 decoder` 下差距 >1，先切回 W16A16 验证流程，再决定是否
扩大 QAT 范围或引入混合精度。

### 5.2 TopK 不参与 QAT

QAT.axera 的全局配置默认会注解 `aten.topk` 的 values。必须用 regional config 把 TopK 的
输入和 values 输出设为 FP32：

```json
{
  "regional_configs": [
    {
      "module_names": null,
      "module_type": "topk",
      "module_config": {
        "is_symmetric": true,
        "input":  {"dtype": "FP32"},
        "output": {"dtype": "FP32"}
      }
    }
  ]
}
```

效果：

- producer 仍是量化图，converter 在 TopK 前插入 Dequantize；
- TopK 在 FP32 执行；
- values 输出不插入 Quantize；
- indices 始终是 int，不参与量化。

如果 full QAT 也要求 TopK 绝对不参与，使用同一 regional config；部分 QAT 则必须使用。

### 5.3 其他排除项

建议保持 FP32 或逐例验证：

- TopK indices、argmax、gather indices；
- anchor 网格生成、`log`/`sigmoid`/`inverse_sigmoid` 的边界常量；
- RoPE 的 `sin/cos`；
- LayerNorm / RMSNorm 的统计量；
- postprocessor 的 TopK；
- matcher/criterion（训练用，不导出）。

## 6. 训练策略

### 6.1 全网络 QAT

1. 从 FP32 checkpoint 构建 2 输出 QAT wrapper（关闭 denoising）。
2. PT2E prepare + observer 校准（几百张图）。
3. QAT 蒸馏或原 criterion 训练。
4. fake-quant 全量 val 复评。
5. convert + QuantONNX + ORT 全量复评。

### 6.2 部分 QAT

推荐 **顺序训练 + 可选联合微调**：

#### 阶段 1：Segment A

- 输入：真实图像。
- teacher：FP32 模型的 `pre_topk_scores`、`memory`、`anchors`。
- loss：
  - `L_score = MSE(scores_A, scores_teacher)` 或对 `max_c` 的排序 loss；
  - `L_memory = 1 - cosine(memory_A, memory_teacher)`；
  - `L_anchor = MSE(anchors_A, anchors_teacher)`。
- 可选 TopK 集合保持 loss：对 teacher top-k 边界附近的 token 加大权重。
- 校准数据：真实图像；A 的输出边界用 FP32。

#### 阶段 2：Segment B

- 输入：边界张量。
  - 先用 **teacher 边界**（oracle）训练，隔离 A 的误差；
  - 再用 **A fake-quant 边界** 微调，适配真实推理分布。
- teacher：FP32 模型的最终 logits/boxes。
- loss：Hungarian 匹配后的 logits MSE + box L1 + GIoU（同全网络蒸馏）。
- 校准数据：从 FP32/A 边界张量中采样，不需要重新跑 backbone。

#### 阶段 3（可选）：联合微调

- 组合 `A-QAT → TopK(FP32) → B-QAT`，端到端蒸馏；
- TopK 的 indices 视为常量，gather 对 `memory`/`anchors` 仍可反传，因此 A 能收到 B 的梯度；
- 小学习率、短步数，避免破坏已收敛的分段。

### 6.3 TopK 对齐原则

- **禁止按输出索引直接对齐**：T1 会改变 query 顺序。
- 训练 B 时使用 Hungarian 匹配最终输出。
- 评估 A 时报告 TopK 集合重合率，而不是只看索引数值。
- 如果 A 的 TopK 集合与 FP32 差异大，优先把 `enc_score_head` 保持 FP32，或提高其量化精度。

## 7. 评估与诊断

### 7.1 全量指标

统一使用 [`baseline.md`](baseline.md) 的协议：

- COCO val2017 全量 5000 张；
- 项目 dataloader/transform；
- 项目 `PostProcessor`；
- 项目 `CocoEvaluator` 的 `mAP50-95`。

对比组合：

| 组合 | 目的 |
|---|---|
| FP32 A + FP32 B | 基线 |
| QAT A + FP32 B | 量化 A 的影响 |
| FP32 A + QAT B | 量化 B 的影响 |
| QAT A + QAT B | 实际部署 |
| 全网络 QAT | 精度上限对照 |

### 7.2 边界诊断

- `pre_topk_scores`：MAE、cosine、max-score 分布；
- `memory`：cosine、相对误差；
- `anchors`：MAE；
- TopK 集合：`|S_q ∩ S_fp| / 300`、Jaccard、每图均值；
- query 覆盖率：TopK 集合里 GT 命中率（可用 GT 框中心/匹配 query 统计）。

### 7.3 判定

- 先看 TopK 集合重合率。若明显下降，优先修 A 的分数头量化，而不是继续调 B。
- 边界误差可接受后，再看组合 mAP。
- 只有组合全量 mAP 达标才能作为交付；分段单独指标不能替代。

## 8. 导出与部署

### 8.1 两种导出形态

**单图（推荐先做）**

```text
images → [A-QAT + FP32 TopK + B-QAT] → logits/boxes
```

- 一个 QuantONNX，TopK 作为 FP32 island；
- ORT 验证方便；
- 后端必须支持量化图中夹 FP32 TopK。

**多图（后端不支持混合图时）**

```text
segment_a.onnx  : images → scores, memory, anchors
topk (FP32)     : scores → indices, gather
segment_b.onnx  : content, ref_anchor, memory, shapes → logits, boxes
```

- 每个子图独立 Q/DQ、独立编译；
- 边界张量 FP32，注意带宽（N 的 memory 约 1MB/图）；
- 需要固定 spatial shapes，或把 shapes 作为常量/metadata。

### 8.2 AXERA/Pulsar2 注意

- TopK 的 indices 是 int64；部分工具链要求 int32，导出后需检查。
- TopK 前后不能有隐式 Q/DQ 改变选择；
- 分段编译时，子图输入输出名和 dtype 必须固定；
- 先做 ORT 全量验收，再做 Pulsar2 编译/仿真/板端，不能用仿真结果替代板端结论。

## 9. 实施计划

| 阶段 | 内容 | 产物 |
|---|---|---|
| P0 | 恢复 FP32 基线工具与权重 | `docs/baseline.md` 可复现 |
| P1 | 全网络 QAT 流程基线（W16A16，历史） | 单图 fake-quant / ORT 指标 |
| P1b | **backbone+encoder W8A16 QAT + FP32 decoder（当前首要）** | encoder QAT + FP32 decoder 全量指标 |
| P2 | 两段式 wrapper | `SegmentA`、`TopKBoundary`、`SegmentB` |
| P3 | A/B 独立 prepare + 校准 | 两个 prepared checkpoint |
| P4 | 顺序 QAT + 联合微调 | A/B QAT checkpoint |
| P5 | 组合评估 + 单图/多图导出 | QuantONNX + ORT 全量 |
| P6 | 对比全网络 vs 部分 QAT | N 的对比报告 |
| P7 | 扩展到 Pico / 其他型号 | 各型号结果 |

建议先在 N 上完成 A/B 对比，再决定是否对 Pico 和 DINOv3 系列采用部分 QAT。

## 10. 风险与待定项

- **TopK 集合翻转**：量化 `enc_score_head` 后 TopK 选择变化，是部分 QAT 的主要风险。
  备选：score head 保持 FP32，只量化 backbone/encoder。
- **边界带宽**：`memory [1,2000,128]` FP32 约 1MB/图，多图部署需评估。
- **B 的分布偏移**：用 teacher 边界训练 B 后，需用 A-QAT 边界微调。
- **T2 精度**：LQE TopK 的 FP32 island 是否被后端支持；若不支持，需要整段 LQE 浮点。
- **全网络 QAT 已能满足 <1 时，部分 QAT 的收益是否值得复杂度**：需要量化对比。
- **部署后端对 FP32/量化混合图的支持**：决定单图还是多图导出。

## 11. 形态对比

| 维度 | 全网络 QAT | 部分 QAT（T1 分割） |
|---|---|---|
| 图数量 | 1 | 2 + TopK 边界 |
| TopK 处理 | 可设 FP32 island | 强制 FP32、不参与 QAT |
| 训练复杂度 | 低 | 高（分段校准/训练/联合微调） |
| 精度上限 | 通常最高 | 取决于 A 的 TopK 保持 |
| 调试能力 | 一般 | 可分段定位 |
| 混合精度 | 单图 regional config | 每段独立配置 |
| 部署复杂度 | 低 | 中/高（边界与多图） |
| 推荐场景 | 默认、先做 | TopK 敏感、后端需要分段或混合精度 |

默认建议：**当前首要实验是 backbone+encoder W8A16 QAT + FP32 decoder**（见
[`qat-experiment-plan.md`](qat-experiment-plan.md)）；W16A16 全网络 QAT 作为流程基线和
精度上限对照，TopK 分割部分 QAT 作为后续对照/部署优化。
