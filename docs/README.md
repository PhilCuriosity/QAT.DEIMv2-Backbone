# DEIMv2 QAT 文档

本目录用于存放 DEIMv2 QAT 的设计、实验和部署说明。

## 现有文档

- [`architecture.md`](architecture.md)：Backbone / Encoder / Decoder / PostProcessor 结构、型号矩阵和 TopK 位置。
- [`baseline.md`](baseline.md)：FP32 基线、评估协议、复现步骤和已复现指标。
- [`qat-pipeline.md`](qat-pipeline.md)：全网络 QAT 与以 TopK 为分割点的部分 QAT 设计。
- [`qat-experiment-plan.md`](qat-experiment-plan.md)：backbone+encoder W8A16 QAT + FP32 decoder 的实验计划。
- [`qat-experiment-log.md`](qat-experiment-log.md)：QAT 实验记录、历史结果和单次实验模板。
- [`../AGENTS.md`](../AGENTS.md)：项目工作约定、模型结构、FP32 基线、量化敏感点、
  历史 W16A16 验证结论和当前状态。

## 计划中的文档

- `deployment.md`：QuantONNX、AXERA Pulsar2 和板端验收

新增文档时请同步更新本索引。
