# QAT 实验记录（W8A16 优先）

本文记录 DEIMv2 QAT 实验。计划见
[`qat-experiment-plan.md`](qat-experiment-plan.md)，评估协议见
[`baseline.md`](baseline.md)。

## 1. 记录规范

每次实验必须记录：

- 实验 ID、日期、负责人
- 模型 / config / 权重路径与 SHA256
- 量化配置（W8A16 / W16A16 / 混合精度）
- 分割点（全网络 / TopK 分割 / decoder 前分割）
- 训练数据、batch、步数、lr、是否蒸馏、是否 LSQ、decoder 是否冻结
- FP32 mAP50-95、QAT mAP50-95、差值
- 评估命令、评估设备、ORT provider
- 产物目录与 SHA256
- 结论、失败原因或待办

精度结论必须基于全量 COCO val2017（5000 张），不能使用子集或不同后处理口径。

## 2. 汇总

| ID | 日期 | 模型 | 形态 | 量化 | 训练 | FP32 | QAT | Δ | 状态 |
|---|---|---|---|---|---:|---:|---:|---|
| RUN-001 | 2026-11-03 | N | FP32 基线 | - | - | 42.9824 | - | - | 完成 |
| RUN-002 | 2026-11-03 | Pico | FP32 基线 | - | - | 38.4593 | - | - | 完成 |
| RUN-003 | 2026-11-04 | S | FP32 基线 | - | - | 50.9000 | - | - | 完成 |
| HIST-001 | 2026-11-03 | N | 全网络 QAT | W16A16 | 是 | 42.9824 | 42.9678 | -0.0146 | 历史，产物已回退 |
| HIST-002 | 2026-11-03 | N | 全网络 QAT → ORT | W16A16 | 是 | 42.9824 | 42.4903 | -0.4921 | 历史，产物已回退 |
| HIST-003 | 2026-11-03 | Pico | 全网络 QAT | W16A16 | 是 | 38.4593 | 38.4756 | +0.0163 | 历史，产物已回退 |
| HIST-004 | 2026-11-03 | Pico | 全网络 QAT → ORT | W16A16 | 是 | 38.4593 | 38.2732 | -0.1861 | 历史，产物已回退 |
| EXP-001 | 待定 | N | backbone+encoder PTQ | W8A16 | 否 | 42.9824 | 待测 | - | 计划 |
| EXP-002 | 待定 | N | backbone+encoder QAT | W8A16 | 是 | 42.9824 | 待测 | - | 计划 |
| EXP-003 | 待定 | N | backbone+encoder QAT + memory 蒸馏 | W8A16 | 是 | 42.9824 | 待测 | - | 计划 |
| EXP-004 | 待定 | N | backbone+encoder 导出/ORT | W8A16 | - | 42.9824 | 待测 | - | 计划 |
| EXP-005 | 待定 | N | backbone+encoder QAT + LSQ | W8A16+LSQ | 是 | 42.9824 | 待测 | - | 计划 |
| EXP-006 | 待定 | N | backbone+encoder W16A16 回退对照 | W16A16 | 是 | 42.9824 | 待测 | - | 计划 |
| EXP-007 | 待定 | Pico | backbone+encoder QAT | W8A16 | 是 | 38.4593 | 待测 | - | 计划 |
| EXP-008 | 待定 | S | backbone+encoder QAT | W8A16 | 是 | 50.9000 | 待测 | - | 计划 |

## 3. 详细记录

### RUN-001：DEIMv2-N FP32 基线

- 日期：2026-11-03
- 模型：DEIMv2 HGNetv2-N
- config：`configs/deimv2/deimv2_hgnetv2_n_coco.yml`
- 权重：`deimv2_hgnetv2_n_coco.pth`
  - SHA256：`3bb740e77c23c62b924f0078ee0973fc2e07d1438113421fe09877e1a422b820`
- 评估：COCO val2017 全量 5000 张，项目 `CocoEvaluator` + `PostProcessor`
- 环境：torch 2.6.0+cu124，A100 80GB
- 结果：**mAP50-95 = 42.9824**
- 产物：已回退，无 ONNX/checkpoint
- 备注：与 README 的 43.0 一致

### RUN-002：DEIMv2-Pico FP32 基线

- 日期：2026-11-03
- 模型：DEIMv2 HGNetv2-Pico
- config：`configs/deimv2/deimv2_hgnetv2_pico_coco.yml`
- 权重：`deimv2_hgnetv2_pico_coco.pth`
  - SHA256：`ff545f3a269fef28842c8e314ff9457950bbf9a1a62a4beb6bd0d20089481409`
- 结果：**mAP50-95 = 38.4593**
- 产物：已回退
- 备注：与 README 的 38.5 一致

### RUN-003：DEIMv2-S FP32 基线

- 日期：2026-11-04
- 模型：DEIMv2 DINOv3-S
- config：`configs/deimv2/deimv2_dinov3_s_coco.yml`
- 权重：`weights/deimv2_dinov3_s_coco.pth`
  - SHA256：`fa07a020c8c7474dc20c70a34f2b16a6a6920b9d2d421146dd79a5d9d72fb2da`
- 结果：**mAP50-95 = 0.509**
- 备注：必须使用 `interaction_indexes: [3, 7, 11]`；HF config 的 `[5, 8, 11]` 只有 47.0

### HIST-001 / HIST-002：N 全网络 W16A16 QAT

- 日期：2026-11-03
- 形态：全网络 QAT（backbone + encoder + decoder）
- 量化：W16A16（U16/S16）
- 训练：200 step，lr 1e-6，FP32 teacher + Hungarian 匹配蒸馏
- FP32：42.9824
- fake-quant：42.9678（Δ -0.0146）
- QuantONNX ORT：42.4903（Δ -0.4921）
- 产物：已按要求回退删除
- 经验：不能按输出索引直接蒸馏；encoder TopK 会重排 query，曾导致 N 掉到 6.8

### HIST-003 / HIST-004：Pico 全网络 W16A16 QAT

- 日期：2026-11-03
- 形态：全网络 QAT
- 量化：W16A16
- 训练：200 step，lr 1e-6，Hungarian 匹配蒸馏
- FP32：38.4593
- fake-quant：38.4756（Δ +0.0163）
- QuantONNX ORT：38.2732（Δ -0.1861）
- 产物：已回退删除

### EXP-001：N backbone+encoder W8A16 PTQ

- 状态：计划
- 目标：只校准 backbone+encoder，不训练，测量纯 PTQ 误差
- 量化：W8A16（U16 激活 / S8 权重）
- 分割：`DEIMTransformer._get_encoder_input` 输出 `memory`
- decoder：FP32 冻结
- 评估：全量 val2017
- 待填：mAP、memory cosine/MAE、TopK 重合率

### EXP-002：N backbone+encoder W8A16 QAT

- 状态：计划
- 目标：验证 QAT 训练能否恢复 PTQ 误差
- 量化：W8A16
- 损失：FP32 decoder + `DEIMCriterion`
- decoder：冻结 FP32
- 训练：1–5 epoch，lr 1e-5 ~ 5e-6
- 待填：训练曲线、mAP、Δ

### EXP-003：N backbone+encoder W8A16 QAT + memory 蒸馏

- 状态：计划
- 目标：在 EXP-002 基础上加入 `1 - cosine(memory_s, memory_teacher)`
- 量化：W8A16
- 待填：辅助 loss 权重、mAP、稳定性

### EXP-004：N backbone+encoder 导出/ORT

- 状态：计划
- 目标：导出 `backbone_encoder_qat.onnx` 与 `decoder_fp32.onnx`
- 验证：PyTorch CPU vs ORT CPU，逐输出直接对比
- 待填：SHA256、max_abs、mAP

### EXP-005：N backbone+encoder W8A16 + LSQ

- 状态：计划
- 目标：8-bit 权重量化误差较大时，对 weight scale 使用 LSQ
- 待填：LSQ 配置、mAP、Δ、训练稳定性

### EXP-006：N backbone+encoder W16A16 回退对照

- 状态：计划
- 目标：如果 W8A16 差距 >1，切回 W16A16 验证流程与精度上限
- 量化：W16A16
- 待填：mAP、Δ

### EXP-007 / EXP-008：Pico / S backbone+encoder QAT

- 状态：计划
- 目标：验证 W8A16 方案是否可扩展到其他型号
- 待填：FP32、PTQ、QAT、Δ

## 4. 单次实验模板

```markdown
### EXP-XXX：<一句话目标>

- 日期：
- 模型 / config：
- 权重 / SHA256：
- 量化配置：
- 分割点：
- decoder 状态：冻结 / 训练
- 训练数据 / batch / steps / lr：
- 蒸馏 / 辅助 loss：
- 评估命令：
- 评估设备 / provider：
- FP32 mAP50-95：
- QAT mAP50-95：
- Δ：
- memory cosine / MAE：
- TopK 集合重合率：
- 产物目录 / SHA256：
- 结论：
- 问题 / 下一步：
```

## 5. 当前结论

- 全网络 W16A16 QAT 已在 N/Pico 上满足 < 1（历史结果，产物已回退）。
- 当前首要目标改为 **W8A16**，验证更保守的 **backbone+encoder QAT + FP32 decoder**：
  - TopK 完全在 FP32 decoder 内，量化风险最低；
  - 8-bit 权重误差更大，需要 QAT，必要时加 LSQ；
  - 如果 W8A16 差距 >1，先用 W16A16 回退对照，再决定是否扩大 QAT 范围。
