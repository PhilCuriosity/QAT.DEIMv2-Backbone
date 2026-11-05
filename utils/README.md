# utils

QAT 流程的公共工具代码，避免脚本之间复制粘贴。

当前：

- `ax_quantizer.py`、`ax_quantizer_lsq.py`、`ax_quantizer_utils.py`：
  AXERA `QAT.Ultralytics` 的 PT2E 量化器源码（含 LSQ），来源和许可证见下文。
- `quantized_decomposed_dequantize_per_channel.py`：
  PyTorch `quantized_decomposed` 算子的 ONNX decomposition，供 QuantONNX 导出使用。
- `infer_quantonnx.py`：联合运行 QAT backbone+encoder QuantONNX 与 decoder FP32 ONNX：
  - 默认只使用 ONNX Runtime；
  - 可选 `--checkpoint` + `--weights` 与 converted PT2E 参考输出对比；
  - 保存 memory / pred_logits / pred_boxes 到 npz。
- `export_fp32_onnx.py`：导出 DEIMv2 浮点 ONNX，支持 `--mode {both,full,backbone_encoder,decoder,all}`：
  - `model_fp32.onnx`：`images → pred_logits, pred_boxes`
  - `backbone_encoder_fp32.onnx`：`images → feat_0, feat_1, ...`
  - `decoder_fp32.onnx`：`memory → pred_logits, pred_boxes`
  - 默认 `both`（full + backbone_encoder）；`all` 额外导出 decoder
  - 导出后使用 ONNX Runtime CPU 与 PyTorch CPU 逐输出对比，结果写入 `export_report.json`

## export_fp32_onnx.py 使用样例

导出 FP32 全模型 + backbone/encoder（默认 `both`）：

```bash
cd ~/project/DEIMv2

python utils/export_fp32_onnx.py \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --weights ~/project/qat.harness/taskDir/DEIMv2/weights/deimv2_hgnetv2_n_coco.pth \
  --output-dir runs/export/deimv2_n_fp32_onnx_$(date +%Y%m%d_%H%M%S) \
  --device cpu
```

同时导出 full / backbone_encoder / decoder：

```bash
python utils/export_fp32_onnx.py \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --weights ~/project/qat.harness/taskDir/DEIMv2/weights/deimv2_hgnetv2_n_coco.pth \
  --output-dir runs/export/deimv2_n_fp32_onnx_all \
  --mode all --device cpu
```

只导出 decoder：

```bash
python utils/export_fp32_onnx.py \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --weights ~/project/qat.harness/taskDir/DEIMv2/weights/deimv2_hgnetv2_n_coco.pth \
  --output-dir runs/export/deimv2_n_decoder_fp32 \
  --mode decoder --device cpu
```

输出：

```text
<output-dir>/
├── model_fp32.onnx               # --mode full/both/all
├── backbone_encoder_fp32.onnx    # --mode backbone_encoder/both/all
├── decoder_fp32.onnx             # --mode decoder/all
└── export_report.json
```

decoder 输入输出：

```text
memory [1,2000,128] → pred_logits [1,300,80], pred_boxes [1,300,4]
```

## infer_quantonnx.py 使用样例

仅使用 ONNX Runtime 联合推理：

```bash
cd ~/project/DEIMv2

python utils/infer_quantonnx.py \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --onnx-dir runs/qat/deimv2_n_w8a16_qat200_lr1e-5_20261104_184700/quantonnx \
  --image example.jpg \
  --save-npz runs/qat/deimv2_n_w8a16_qat200_lr1e-5_20261104_184700/quantonnx/joint_output.npz \
  --output runs/qat/deimv2_n_w8a16_qat200_lr1e-5_20261104_184700/quantonnx/joint_infer_report.json
```

输出：

- `--save-npz`：`memory`、`pred_logits`、`pred_boxes`
- `--output`：推理与验证 JSON

与 converted PT2E 参考输出对比（需要 FP32 权重构建 decoder）：

```bash
python utils/infer_quantonnx.py \
  --config configs/deimv2/deimv2_hgnetv2_n_coco.yml \
  --checkpoint runs/qat/deimv2_n_w8a16_qat200_lr1e-5_20261104_184700/qat_state.pt \
  --weights ~/project/qat.harness/taskDir/DEIMv2/weights/deimv2_hgnetv2_n_coco.pth \
  --onnx-dir runs/qat/deimv2_n_w8a16_qat200_lr1e-5_20261104_184700/quantonnx \
  --reference-mode converted \
  --image example.jpg \
  --atol 5e-3
```

`--checkpoint` 只包含 QAT backbone+encoder 图状态，不包含 decoder 权重，因此同时需要 `--weights` 构建 FP32 decoder 参考。

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
