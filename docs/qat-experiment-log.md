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
| EXP-000 | 2026-11-04 | N | 路线B实现 + 回归测试 | W8A16 | - | - | - | - | 完成 |
| EXP-001 | 2026-11-04 | N | backbone+encoder PTQ | W8A16 | 否 | 42.9824 | 21.8150 | -21.1674 | 完成，PTQ 误差大 |
| EXP-002 | 2026-11-04 | N | backbone+encoder QAT（200 step） | W8A16 | 是 | 42.9824 | 31.9316 | -11.0508 | 部分完成，需更多训练 |
| EXP-003 | 待定 | N | backbone+encoder QAT + memory 蒸馏 | W8A16 | 是 | 42.9824 | 待测 | - | 计划 |
| EXP-004 | 待定 | N | backbone+encoder 导出/ORT | W8A16 | - | 42.9824 | 待测 | - | 计划 |
| EXP-005 | 2026-11-05 | N | backbone+encoder LSQ + KD smoke | W8A16+LSQ | 是 | 42.9824 | 待测 | - | 链路 smoke 通过，精度待测 |
| EXP-006 | 待定 | N | backbone+encoder W16A16 回退对照 | W16A16 | 是 | 42.9824 | 待测 | - | 计划 |
| EXP-007 | 待定 | Pico | backbone+encoder QAT | W8A16 | 是 | 38.4593 | 待测 | - | 计划 |
| EXP-008 | 待定 | S | backbone+encoder QAT | W8A16 | 是 | 50.9000 | 待测 | - | 计划 |
| EXP-009 | 2026-11-05 | S | backbone+encoder W8A8 纯 QAT（50 epoch） | W8A8 | 是 | 50.9000 | 未测 | - | 已启动，epoch 0/50 手动终止，产物保留 |

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

### EXP-000：路线 B 实现 + 模型变更/自动发现回归

- 日期：2026-11-04
- 形态：backbone+encoder QAT，FP32 decoder 冻结
- 量化：W8A16
- 代码：
  - `qat/boundaries.py`
  - `qat/pt2e.py`
  - `qat/regional.py`
  - `qat/train_backbone_encoder.py`
  - `qat/tests/test_model_change.py`
- 测试：
  - 在 encoder 后插入额外 Conv+BN+SiLU block；
  - FX export + PT2E prepare 成功；
  - regional discovery 识别新 block 节点；
  - 生成的 regional config 可加载；
  - 完成一次 QAT forward/backward/optimizer step。
- 结果：`pytest qat/tests/test_model_change.py` 4 passed。
- 产物：临时 smoke 目录，未作为正式精度实验。
- 备注：路线 A（全网络 QAT）只记录在文档，不实现。

### EXP-001：N backbone+encoder W8A16 PTQ

- 日期：2026-11-04
- 状态：完成
- 量化：W8A16（U16 激活 / S8 权重），auto regional config
- 分割：`DEIMTransformer._get_encoder_input` 输出 `memory`
- decoder：FP32 冻结
- 校准：val2017，16 batch × 16
- 评估：全量 val2017
- 结果：**mAP50-95 = 21.8150**（Δ -21.1674）
- 产物：`runs/qat/deimv2_n_w8a16_ptq_20261104_184900`（已删除，指标保留在本文档）
- 结论：W8A16 纯 PTQ 在 backbone+encoder 上误差很大，不能直接交付。

### EXP-002：N backbone+encoder W8A16 QAT（200 step）

- 日期：2026-11-04
- 状态：部分完成
- 量化：W8A16 + auto regional config
- 损失：FP32 decoder + `DEIMCriterion`
- decoder：冻结 FP32
- 训练：200 step，batch 16，lr 1e-5
- 评估：全量 val2017
- 结果：**mAP50-95 = 31.9316**（Δ -11.0508）
- 结论：QAT 从 21.8 恢复到 31.9，但距离 FP32 仍差 11.05；200 step 仅约 3200 张图，远未完成 1 epoch，需要更多 step、调 lr、LSQ 或混合精度。
- 产物：`runs/qat/deimv2_n_w8a16_qat200_lr1e-5_20261104_184700`（已删除，指标保留在本文档）
- 备注：早期 38.65 的结果是验证时 observer 仍在更新导致的验证集泄漏，已废弃；当前结果在评估前显式冻结 observer。

### EXP-003：N backbone+encoder W8A16 QAT + memory 蒸馏

- 状态：计划
- 目标：在 EXP-002 基础上加入 `1 - cosine(memory_s, memory_teacher)`
- 量化：W8A16
- 待填：辅助 loss 权重、mAP、稳定性

### EXP-004：N backbone+encoder 导出/ORT

- 状态：QAT 图导出完成
- 目标：导出 `backbone_encoder_qat.onnx` 与 `decoder_fp32.onnx`
- 产物：`runs/qat/deimv2_n_w8a16_qat200_lr1e-5_20261104_184700/quantonnx/`（已删除，指标保留在本文档）
- 验证：PyTorch CPU vs ORT CPU
  - backbone+encoder `memory`：mean_abs 2.39e-4，max_abs 0.035
  - decoder（FP32，含 TopK）mean_abs 1.08e-3 / 6.8e-5
- 简化：`program.optimize()` + `onnxslim.slim`
  - encoder：900 → 899 节点，Constant 0，functions 0（优化前 692 Constant / 3 functions）
  - decoder：545 → 548 节点，Constant 0，functions 0
- 联合推理：`qat/infer_quantonnx.py`，ORT 输出与 converted PT2E 参考匹配
  （matched logits mean 1.35e-3，boxes mean 2.0e-5），`joint_infer_report.json` passed
- 备注：QAT 图 233 QuantizeLinear / 376 DequantizeLinear；decoder 原始输出可能因 TopK 查询顺序变化，按 mean 或后处理比较
- 待填：组合全量 mAP

### EXP-005：N backbone+encoder W8A16 + LSQ + KD smoke

- 状态：链路 smoke 通过，精度待测
- 目标：验证 LSQ prepared 图、KD loss、checkpoint 元数据和 QuantONNX 导出闭环
- 量化：W8A16 + LSQ
- 训练：1 step，batch 2，teacher=FP32 N，KD memory/logits/boxes/GIoU
- 结果：训练正常，loss 有限；checkpoint 含 `use_lsq=true`、`distill=true`；
  LSQ checkpoint 可导出 QuantONNX，`export_report.json passed=true`
- 产物：`runs/smoke/deimv2_n_w8a16_lsq_kd_smoke_20261105_094307`（已删除）
- 待填：正式 step/epoch、全量 mAP、LSQ lr 调参
- 单元测试：`qat/tests/test_distill.py` 3 passed

### EXP-009：S backbone+encoder W8A8 纯 QAT（50 epoch）

- 状态：**已启动并手动终止**（2026-11-05，用户要求停止；非崩溃/非 OOM）
- 模型：DEIMv2-S，`configs/deimv2/deimv2_dinov3_s_coco.yml`
- 权重：`weights/deimv2_dinov3_s_coco.pth`
- 量化：W8A8（U8/S8）
- 训练：50 epoch，batch 32，lr 1e-5，不使用 max-steps
- 校准：32 个 val batch（约 1024 张图）
- 评估：每个 epoch 全量 val2017，记录 `eval_history.jsonl`
- 蒸馏：不启用
- LSQ：不启用
- tmux session：`deimv2_s_w8a8_20261105_104136`（已 kill）
- 产物目录：`runs/qat/deimv2_s_w8a8_qat_20261105_104136/`
  - 含 `launch.sh` / `command.txt` / `train_args.json` / `environment.json` /
    `config_snapshot/` / `quant_config_discovered.json` / `regional_discovery.json` /
    `train.log` / `STATUS.md`
- 终止时进度：epoch 0 未完成，最后 step 300/3696（约 9600 张图），最后 loss 28.16273
- 未产生 checkpoint（`qat_state.pt` 按 epoch 落盘）和评估结果
- 精度：未测（无有效 epoch 产物）
- 恢复方式：`bash runs/qat/deimv2_s_w8a8_qat_20261105_104136/launch.sh`，
  建议换新的时间戳目录避免覆盖日志

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
