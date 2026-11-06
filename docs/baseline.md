# FP32 基线

本文记录 DEIMv2 各型号在 COCO val2017 上的 FP32 基线、评估协议和复现方式。QAT 验收以本文的
FP32 数字为对照，不使用 README 表格或历史报告中的数字直接替代。

## 1. 结论

当前环境已复现：

| 型号 | 权重 | FP32 mAP50-95 | 与 README 对比 |
|---|---|---:|---|
| HGNetv2-N | `deimv2_hgnetv2_n_coco.pth` | **42.9824** | 43.0，一致 |
| HGNetv2-Pico | `deimv2_hgnetv2_pico_coco.pth` | **38.4593** | 38.5，一致 |
| DINOv3-S | `deimv2_dinov3_s_coco.pth` | **50.9** | 50.9，一致 |

其余型号（Atto / Femto / M / L / X）尚未在当前环境复测。

## 2. 环境

```bash
# 已激活项目 Python 环境时可直接使用 python；也可用 PY 覆盖解释器路径。
PY=${PY:-python}
```

- Python 3.10.19
- torch 2.6.0+cu124
- torchvision 0.21.0+cu124
- faster-coco-eval 1.8.0
- onnx 1.17.0 / onnxruntime 1.21.0
- GPU：NVIDIA A100-SXM4-80GB
- COCO 根目录：`$COCO_ROOT`（本地 COCO 数据根目录）

torchvision 0.21 与项目自带的 v2 transform 存在 API 差异（`_transform` → `transform`）。
复现前需要兼容层，或在 torchvision 0.20.1 环境运行。

## 3. 评估协议

- 数据：COCO **val2017 全量 5000 张**，不使用子集。
- 输入：各型号 YAML 的 `eval_spatial_size`。
- 预处理：
  - HGNetv2 型号：`Resize [H,W]` → `float32 / 255`，**不做 ImageNet Normalize**。
  - DINOv3 型号：`Resize [640,640]` → `/255` →
    `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`。
- 后处理：项目 `PostProcessor`，focal 模式下对 `sigmoid(logits)` 在 **query × class** 上做
  全局 TopK300，同一个 query 可对应多个类别。
- 指标：项目 `CocoEvaluator`（faster-coco-eval）的 `coco_eval_bbox`，取第 0 项
  `mAP50-95`。
- 结果记录：完整 12 项 COCO 指标、权重 SHA256、config、输入尺寸、GPU、命令。

## 4. 已复现的完整指标

### HGNetv2-N

```
mAP50-95 = 0.42982352856640155   (42.9824)
mAP50    = 0.6012689168981549
mAP75    = 0.4616168147705317
mAP_small  = 0.2485166343158803
mAP_medium = 0.4728472229780308
mAP_large  = 0.6139622972546678
AR1      = 0.347239816123394
AR10     = 0.5719515572164774
AR100    = 0.6365412838231055
AR_small = 0.41253205532897097
AR_medium= 0.6991543332477235
AR_large = 0.842892697128089
```

- 权重 SHA256：`3bb740e77c23c62b924f0078ee0973fc2e07d1438113421fe09877e1a422b820`
- 评估耗时：约 64 秒（A100，batch 64）

### HGNetv2-Pico

```
mAP50-95 = 0.38459307009587224   (38.4593)
mAP50    = 0.5527392060109105
mAP75    = 0.41149648255730376
mAP_small  = 0.2099967005622642
mAP_medium = 0.42383644069598253
mAP_large  = 0.548650447610879
AR1      = 0.32512200023496596
AR10     = 0.544896116272676
AR100    = 0.6028411025307916
AR_small = 0.3680238878376768
AR_medium= 0.6659076805235209
AR_large = 0.8308553236901648
```

- 权重 SHA256：`ff545f3a269fef28842c8e314ff9457950bbf9a1a62a4beb6bd0d20089481409`
- 评估耗时：约 61 秒（A100，batch 64）

### DINOv3-S

- `mAP50-95 = 0.509`
- 权重：`weights/deimv2_dinov3_s_coco.pth`
  - `.pth` SHA256：`fa07a020c8c7474dc20c70a34f2b16a6a6920b9d2d421146dd79a5d9d72fb2da`
  - HF `model.safetensors` SHA256：`fe545b150766b8761a696f7b1d92ea138fafe8e02a7398823e0c62d2a5867956`
  - 来源：`Intellindust/DEIMv2_DINOv3_S_COCO`
- 必须使用项目 YAML 的 `interaction_indexes: [3, 7, 11]`。
- 使用 HuggingFace `config.json` 的 `[5, 8, 11]` 只有 `0.470`；HF config 与发布权重不一致，
  不要用 HF config 构建推理或训练图。

## 5. 权重准备

仓库不提交权重。需要准备：

```text
weights/
├── deimv2_hgnetv2_n_coco.pth
├── deimv2_hgnetv2_pico_coco.pth
└── deimv2_dinov3_s_coco.pth
```

格式为：

```python
{"model": state_dict}
```

来源：

- HuggingFace：`Intellindust/DEIMv2_HGNetv2_N_COCO`、`..._PICO_COCO`、`..._DINOv3_S_COCO`
- README 中的 Google Drive / Quark 官方 checkpoint

从 HF safetensors 转换时，**按项目 YAML 建图**，不要直接用 HF `config.json`。N / Pico 的 HF
config 与 YAML 一致，S 的 `interaction_indexes` 不一致。

## 6. 复现命令

项目 YAML 中的 COCO 路径可能是占位路径。在仓库根目录执行以下命令，用临时 YAML
覆盖数据路径：

```bash
export COCO_ROOT=/path/to/coco
PY=${PY:-python}

mkdir -p runs/baseline
cat > runs/baseline/deimv2_baseline_n.yml <<EOF
__include__: ['$PWD/configs/deimv2/deimv2_hgnetv2_n_coco.yml']

output_dir: runs/baseline/deimv2_baseline_n
train_dataloader:
  dataset:
    img_folder: $COCO_ROOT/train2017/
    ann_file: $COCO_ROOT/annotations/instances_train2017.json
val_dataloader:
  dataset:
    img_folder: $COCO_ROOT/val2017/
    ann_file: $COCO_ROOT/annotations/instances_val2017.json
  num_workers: 8
EOF

CUDA_VISIBLE_DEVICES=0 $PY train.py \
  -c runs/baseline/deimv2_baseline_n.yml \
  --test-only \
  -r weights/deimv2_hgnetv2_n_coco.pth \
  -d cuda:0
```

Pico 只需替换 YAML 和权重路径；S 替换为 `deimv2_dinov3_s_coco.yml` 和对应权重。

输出中的：

```text
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.430
```

即 mAP50-95。完整 12 项指标会打印在随后几行。

### torchvision 0.21 兼容写法

当前 `torch2.6` 环境是 torchvision 0.21，而项目 transform 使用 0.20 的 `_transform` hook，
直接运行会报 `NotImplementedError`。仓库回退后没有内置兼容层；可以用下面的自包含脚本运行
（与直接调用 `train.py` 等价）：

```bash
PY=${PY:-python}
CUDA_VISIBLE_DEVICES=0 $PY - <<'EOF'
import sys
sys.path.insert(0, '.')
from engine.data.transforms._transforms import ConvertBoxes, ConvertPILImage, PadToSize

for cls in (PadToSize, ConvertBoxes, ConvertPILImage):
    if '_transform' in cls.__dict__:
        cls.transform = cls._transform
    if '_get_params' in cls.__dict__:
        cls.make_params = cls._get_params

import runpy
sys.argv = [
    'train.py',
    '-c', 'runs/baseline/deimv2_baseline_n.yml',
    '--test-only',
    '-r', 'weights/deimv2_hgnetv2_n_coco.pth',
    '-d', 'cuda:0',
]
runpy.run_path('train.py', run_name='__main__')
EOF
```

如果切换到 torchvision 0.20.1 环境，可直接使用上一节的 `train.py` 命令。

## 7. 与量化结果的关系

W16A16 的量化验证结果不在本文重复，见 `AGENTS.md` 的“历史验证结论”一节。历史 W16A16
产物已按要求回退；当前 QAT 工具链见 `qat/`。重新开始 QAT 时，必须先按本文复现 FP32
基线，再与量化结果比较。

验收标准：

```text
| QAT mAP50-95 - FP32 mAP50-95 | < 1.0
```

且必须使用同一份预处理、后处理、评估脚本和全量 val2017。
