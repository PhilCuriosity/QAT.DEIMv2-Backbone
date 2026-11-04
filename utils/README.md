# utils

QAT 流程的公共工具代码，避免脚本之间复制粘贴。

当前：

- `ax_quantizer.py`、`ax_quantizer_lsq.py`、`ax_quantizer_utils.py`：
  AXERA `QAT.Ultralytics` 的 PT2E 量化器源码（含 LSQ），来源和许可证见下文。
- `export_fp32_onnx.py`：导出 DEIMv2 浮点 ONNX：
  - `model_fp32.onnx`：`images → pred_logits, pred_boxes`
  - `backbone_encoder_fp32.onnx`：`images → feat_0, feat_1, ...`
  - 导出后使用 ONNX Runtime CPU 与 PyTorch CPU 逐输出对比，结果写入 `export_report.json`。

建议放置：

- 权重/配置/ONNX 的 SHA256 与 manifest
- COCO 数据路径检查与 dataloader 覆盖
- FP32 / fake-quant / ORT 的统一评估入口
- ONNX checker 与 ORT session 探测
- 日志、随机种子、设备选择

保持这里的函数无副作用；具体实验入口放到 `tools/` 或独立脚本中。

## AXERA quantizer 来源与许可证

`ax_quantizer.py`、`ax_quantizer_lsq.py`、`ax_quantizer_utils.py` 来自：

- Repository：`https://github.com/AXERA-TECH/QAT.Ultralytics`
- Commit：`3da74ff56df605b96ec0f249f299044a1fa2c925`
- Path：`ultralytics/utils/ax_*.py`

本地修改：

- `from ultralytics.utils.ax_quantizer_utils import ...` 改为
  `from .ax_quantizer_utils import ...`，使文件可作为本项目的 `utils` 包导入。

QAT.Ultralytics 以 AGPL-3.0 分发。将这些文件与 DEIMv2 一起再分发前，需要确认许可证
条款与预期用途兼容。
