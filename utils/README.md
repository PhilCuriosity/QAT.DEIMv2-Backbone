# utils

QAT 流程的公共工具代码，避免脚本之间复制粘贴。

当前：

- `ax_quantizer.py`、`ax_quantizer_lsq.py`、`ax_quantizer_utils.py`：
  AXERA `QAT.Ultralytics` 的 PT2E 量化器源码（含 LSQ），来源和许可证见下文。
- `quantized_decomposed_dequantize_per_channel.py`：
  PyTorch `quantized_decomposed` 算子的 ONNX decomposition，供 QuantONNX 导出使用。


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
