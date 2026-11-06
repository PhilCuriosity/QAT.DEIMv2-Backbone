# QAT 实验计划：Backbone + Encoder（W8A16）

本文记录“只对 backbone 和 encoder 做 QAT，decoder 保持 FP32”的实验计划。它是
[`qat-pipeline.md`](qat-pipeline.md) 中部分 QAT 的一个特例：分割点在 decoder 的
`_get_encoder_input` 输出，TopK 完全留在 FP32 decoder 内。

## 1. 目标

- 在 DEIMv2-N 上先验证：只量化并 QAT 训练 backbone + encoder，decoder 冻结 FP32。
- QAT 模型在全量 COCO val2017（5000 张）上的 `mAP50-95` 与 FP32 基线差距 < 1.0。
- 验证通过后扩展到 Pico / S / 其他型号。
- 与全网络 QAT、以 TopK 为分割点的部分 QAT 做精度/复杂度对比。

## 2. 分割点

### 2.1 代码位置

用户提到的 `engine/deim/rtdetrv2_decoder.py:569` 属于 `RTDETRTransformerv2`，用于
`configs/deim_rtdetrv2/*`。DEIMv2-N 实际使用 `DEIMTransformer`，等价位置是：

```text
engine/deim/deim_decoder.py:527
    memory, spatial_shapes = self._get_encoder_input(feats)
```

`DFINETransformer` 的等价位置：

```text
engine/deim/dfine_decoder.py:705
    memory, spatial_shapes = self._get_encoder_input(feats)
```

如果确实要跑 RT-DETRv2 分支，需要改用 `configs/deim_rtdetrv2/*` 和
`rtdetrv2_decoder.py`；本计划默认按 DEIMv2 `DEIMTransformer` 执行。

### 2.2 QAT 图范围

QAT 图包含：

```text
images
  → backbone
  → encoder
  → decoder.input_proj
  → flatten + concat
  → memory [B, N_tokens, C]
```

即 `_get_encoder_input(feats)` **本身**包含在 QAT 图中，输出 `memory` 为边界。

DEIMv2-N @ 640：

```text
memory:         [1, 2000, 128]
spatial_shapes: [[40, 40], [20, 20]]
```

对已发布型号，encoder 的 `hidden_dim` 与 decoder 的 `hidden_dim` 相同，`input_proj`
实际上是 `Identity`；因此 QAT 图等价于 **backbone + encoder**。通道不匹配的配置才会包含
`input_proj` 的 1×1 Conv + BN。

### 2.3 FP32 decoder 范围

以下全部保持 FP32、冻结：

- anchor 生成
- encoder query TopK（T1）
- decoder self-attn / deformable cross-attn / FFN
- FDR / Integral
- LQE（含 T2 TopK）
- 最终 `pred_logits` / `pred_boxes`

因此本方案中 **TopK 不进入 QAT 图**，不会出现 query 重排导致的量化训练问题。

decoder 默认冻结。如需 FP32 微调，可使用 `--train-decoder --decoder-lr 1e-6`；
decoder 仍不进入 PT2E 图，checkpoint 会额外保存 `decoder` state，导出时自动加载。

## 3. 与已有两种形态的关系

| 形态 | QAT 图 | TopK | 训练复杂度 | 说明 |
|---|---|---|---|---|
| 全网络 QAT | backbone + encoder + decoder | 可设为 FP32 island | 低 | 精度上限对照 |
| TopK 分割部分 QAT | encoder 前段 / decoder 后段 | 强制 FP32 边界 | 高 | 需处理 query 集合/接口 |
| **本方案** | backbone + encoder + input_proj | 完全在 FP32 decoder 内 | 中 | 先隔离 TopK，再逐步扩大 QAT 范围 |

本方案是“从 encoder 侧开始”的最安全路径：TopK 和 decoder 都不参与量化，适合先验证
backbone/encoder 的量化误差。

## 4. 量化配置

**首要目标：W8A16（U16 激活 / S8 权重）**。

W8A16 配置示例：

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

W16A16（U16/S16）作为对照和回退方案；如果 W8A16 在冻结 decoder 下差距 >1，先切回
W16A16 验证流程，再决定是否扩大 QAT 范围或引入 LSQ。

量化：

- backbone / encoder / input_proj 中的 Conv、Linear、MatMul、Add、Mul、Concat、
  Sigmoid、Softmax、GridSample 等。
- BN / SyncBN 可折叠；QAT 训练中保留 running stats。

保持浮点：

- LayerNorm / RMSNorm；
- RoPE 的 sin/cos；
- anchor 常量、`inverse_sigmoid` / `log` 边界；
- TopK、indices、postprocessor TopK（不在 QAT 图内）。

校准：

- 使用真实图像（train 或 val），约 200–500 张；
- 只校准到 `memory` 输出；
- 记录校准数据、batch、观察器类型。

W8A16 的权重只有 8-bit，通常需要 QAT；必要时对权重 scale 使用 LSQ（learnable step size）。
后续实验：

- W8A16 + LSQ；
- W16A16 对照；
- 混合精度：backbone 的敏感层（首/末层、depthwise）保持 W16A16，其余 W8A16；
- W8A8 仅作为后续探索。

## 5. 训练设计

### 5.1 模型拆分

建议把 `DEIMTransformer.forward` 重构为：

```python
def encode(self, feats):
    return self._get_encoder_input(feats)

def decode(self, memory, spatial_shapes, targets=None):
    # denoising + _get_decoder_input + decoder + heads
    ...

def forward(self, feats, targets=None):
    memory, shapes = self.encode(feats)
    return self.decode(memory, shapes, targets)
```

QAT 图只包 `backbone + encoder + decoder.input_proj + encode`，输出 `memory`。

`FrozenDecoder` 调用 `decode(memory, spatial_shapes, targets)`，参数冻结、eval 模式。

### 5.2 损失

推荐主方案：

```text
memory_s = student_qat(images)
outputs = frozen_decoder.decode(memory_s, shapes, targets)
loss = DEIMCriterion(outputs, targets)
```

- decoder 参数 `requires_grad=False`；
- 梯度通过 FP32 decoder 反传到 `memory_s`，再反传到 backbone/encoder；
- TopK 在 decoder 内，按原逻辑执行；
- 这是最接近“量化 encoder + FP32 decoder”部署形态的训练。

辅助 loss（可选）：

```text
L = L_criterion
  + λ_mem * (1 - cosine(memory_s, memory_teacher))
```

- memory 蒸馏只作为正则，权重不宜过大；
- 不使用 query 索引对齐的 logits/boxes 蒸馏。

备选方案：只做 memory MSE/cosine 蒸馏，不调用 decoder/criterion；实现简单，但 task-aware
程度较低，作为对照。

### 5.3 超参起点

- epoch：1–5
- batch：16–32
- optimizer：AdamW
- lr：1e-5 ~ 5e-6（backbone 可更低）
- weight decay：1e-4
- observer warmup：50–100 step 后冻结
- clip grad norm：0.1
- EMA：可选

### 5.4 冻结项

- decoder 全部参数；
- decoder 的 BN/LayerNorm 统计量（eval）；
- 不冻结 backbone/encoder 的 BN running stats（按 QAT 正常更新）；
- criterion 不参与导出，只训练用。

## 6. 评估

全量 COCO val2017，统一协议：

| 组合 | 目的 |
|---|---|
| FP32 encoder + FP32 decoder | 基线 |
| QAT encoder + FP32 decoder | 本方案 |
| FP32 encoder + FP32 decoder（teacher memory） | 上界/隔离 |
| 全网络 QAT | 精度对照 |

指标：

- mAP50-95（主指标）；
- `memory` cosine / MAE；
- decoder TopK query 集合重合率（teacher memory vs student memory）；
- 最终 top-k 检测框/分数差异。

验收：

```text
|QAT backbone+encoder - FP32| < 1.0
```

且必须全量 val2017，不能用子集替代。

## 7. 导出

QAT 完成后建议导出两个图：

1. `backbone_encoder_qat.onnx`：`images → memory`
2. `decoder_fp32.onnx`：`memory (+ spatial_shapes) → pred_logits / pred_boxes`

也可以尝试单图（量化 encoder + FP32 decoder），但需要后端支持混合量化/浮点图。

验证：

- PyTorch **CPU** vs ONNX Runtime **CPU**；
- 两个子图都逐输出直接对比（不含 TopK，可以直接要求数值一致）；
- 记录 SHA256、输入输出 shape、最大绝对误差。

## 8. 实验矩阵

| ID | 内容 | 量化 | 训练 | 状态 |
|---|---|---|---|---|
| E0 | N/Pico/S FP32 基线复测 | - | - | N/Pico/S 已完成 |
| E1 | N backbone+encoder PTQ（仅校准，不训练） | W8A16 | 否 | 计划 |
| E2 | N backbone+encoder QAT（criterion） | W8A16 | 是 | 计划 |
| E3 | N backbone+encoder QAT + memory 蒸馏 | W8A16 | 是 | 计划 |
| E4 | E2 导出 backbone/decoder ONNX + ORT 验证 | W8A16 | - | 计划 |
| E5 | N W8A16 + LSQ 对比 | W8A16+LSQ | 是 | 计划 |
| E6 | N W16A16 回退对照 | W16A16 | 是 | 计划 |
| E7 | Pico backbone+encoder QAT | W8A16 | 是 | 计划 |
| E8 | S backbone+encoder QAT | W8A16 | 是 | 计划 |
| E9 | 全网络 QAT 对照 | W16A16 | 是 | 历史结果，产物已回退 |

## 9. 风险与备选

- **decoder 冻结导致精度上限受限**：如果 W8A16 下差距 >1，先尝试 W16A16 回退对照，
  再尝试 LSQ、解冻 decoder 的 LayerNorm/最后一层，或改用全网络 QAT。
- **8-bit 权重误差**：W8A16 的权重量化误差比 W16A16 大，可能需要 LSQ、逐层混合精度
  或更长的 QAT 训练。
- **BN 统计漂移**：QAT 训练中 backbone/encoder BN 更新可能影响 FP32 decoder；需要监控
  BN running stats 和校准数据分布。
- **input_proj**：已发布型号多为 Identity；若后续配置包含 Conv+BN，需要单独校准。
- **memory 边界**：N 的 `[1,2000,128]` FP32 约 1MB/图；部署时注意带宽。
- **导出混合图**：后端可能不支持量化 encoder + FP32 decoder 的单图，需要双图。
- **评估差异**：TopK 在 FP32 decoder 内，query 顺序仍可能因 encoder 数值差异变化；
  以最终 mAP 和 query 集合重合率为准。

## 10. 实施步骤

1. 恢复 QAT 基础设施（PT2E / QAT.axera / 数据加载）。
2. 重构 `DEIMTransformer` 为 `encode` / `decode`（或用 wrapper 复现）。
3. 实现 `BackboneEncoderQAT`、`FrozenDecoder`。
4. N + W8A16：prepare、校准、E1 PTQ 评估。
5. E2 QAT 训练 + 全量评估；必要时 E5 加入 LSQ。
6. E4 导出 + CPU/ORT 直接验证。
7. 扩展 W16A16 回退对照（E6）到 Pico、S。
8. 与全网络 QAT、TopK 分割方案对比。

## 11. 待确认

1. 确认使用 DEIMv2 `DEIMTransformer`（`deim_decoder.py:527`），不是 RT-DETRv2 的
   `rtdetrv2_decoder.py:569`。
2. 第一版使用 **W8A16**；W16A16 作为回退对照。
3. decoder 冻结 FP32，只训练 backbone + encoder。
4. 先做 N，还是 N + Pico 并行。
5. 是否接受“量化 encoder + FP32 decoder”双图导出。
6. W8A16 是否需要首轮就启用 LSQ。

## 12. 实现状态（2026-11-04）

路线 B 的代码骨架已落地：

- `qat/boundaries.py`：`BackboneEncoderQAT`、`FrozenDecoder`
- `qat/pt2e.py`：W8A16 prepare/校准/convert
- `qat/regional.py`：从 FX 图自动发现 softmax / depthwise / 首尾 conv / attention
- `qat/train_backbone_encoder.py`：W8A16 QAT 训练 CLI
- `qat/tests/test_model_change.py`：模型变更 + regional discovery 回归测试

测试结果：

```text
qat/tests/test_model_change.py ....  [100%]
4 passed
```

已覆盖：

- encoder 后插入新的 Conv+BN+SiLU block 后，export/prepare 仍成功；
- 新节点被 regional discovery 发现并写入配置；
- 生成的 regional config 可被 `ax_load_config` 加载；
- 修改结构后仍可完成 QAT forward/backward/optimizer step。

路线 A（全网络 QAT）只在 `qat-pipeline.md` 中保留设计，不实现。

### 初步精度结果

- EXP-001（W8A16 PTQ，16 校准 batch）：mAP50-95 = **21.8150**
- EXP-002（W8A16 QAT，200 step，lr 1e-5）：mAP50-95 = **31.9316**
- FP32：42.9824

结论：W8A16 纯 PTQ 误差很大，QAT 有恢复趋势，但 200 step 仍差 11.05；仅约 3200 张图，远未完成 1 epoch。后续优先：

1. 增加 QAT step / epoch；
2. 调整 lr（backbone vs encoder）；
3. 尝试 LSQ；
4. 对敏感层做混合精度（例如 depthwise/首尾 conv 保持 W16A16）；
5. 再评估 decoder 是否需要解冻。

## 13. LSQ 与蒸馏支持（2026-11-05）

已接入路线 B 训练/导出流程：

- `--lsq`：切换到 `utils/ax_quantizer_lsq.py`
- checkpoint 写入 `use_lsq`，导出自动识别并重建 LSQ 图
- `--teacher-weights`：启用 FP32 teacher 蒸馏
- KD loss：
  - memory cosine
  - Hungarian 匹配后的 logits MSE / box L1 / GIoU
- checkpoint 写入 `distill` 字段

已验证：

- LSQ + KD 1-step smoke 正常
- LSQ checkpoint → QuantONNX 导出 `passed=true`
- `qat/tests/test_distill.py` 3 passed

尚未完成：正式 LSQ/KD 全量训练和精度验收。

## 14. 本轮执行计划：W8A8 纯 QAT（不启用 KD/LSQ）

本轮只验证 DEIMv2-S 的单纯 QAT：

- 模型：DEIMv2-S（DINOv3-S）
- config：`configs/deimv2/deimv2_dinov3_s_coco.yml`
- 权重：`weights/deimv2_dinov3_s_coco.pth`
  - SHA256：`fa07a020c8c7474dc20c70a34f2b16a6a6920b9d2d421146dd79a5d9d72fb2da`
- 范围：backbone + encoder QAT，decoder 冻结 FP32
- 量化：W8A8（U8 激活 / S8 权重）
- 训练：50 epoch，batch 32
- 不启用 LSQ
- 不启用蒸馏

命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m qat.train_backbone_encoder \
  --config configs/deimv2/deimv2_dinov3_s_coco.yml \
  --weights weights/deimv2_dinov3_s_coco.pth \
  --quant-config qat/configs/quant/w8a8.json \
  --output-dir runs/qat/deimv2_s_w8a8_qat_$(date +%Y%m%d_%H%M%S) \
  --device cuda:0 --batch-size 32 --workers 8 \
  --epochs 50 --lr 1e-5 --calibration-batches 32 --eval-every 1
```

`calibration-batches 32` 的作用：

- 在 QAT 前，用 val loader 的 32 个 batch 运行 encoder；
- observer 统计每层激活 min/max，初始化 fake-quant 的 scale/zero_point；
- 只更新量化参数，不更新网络权重，不使用标签；
- 默认校准后关闭 observer，后续 50 epoch 使用固定 qparams；
- 如需训练中继续更新 observer，加 `--keep-observers`。
- `--eval-every 1`：每个 epoch 跑一次全量 val2017，写入 `eval_history.jsonl`；
  最终结果写入 `eval.json`。50 次全量评估会增加明显耗时。

产物：

```text
<output-dir>/
├── qat_state.pt
├── quant_config_discovered.json
├── regional_discovery.json
├── train_history.json
└── eval.json
```

本次不使用 `--teacher-weights` / `--kd-*`，KD 代码保留供后续对照实验。
