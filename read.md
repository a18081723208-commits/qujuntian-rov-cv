# 水下桥梁病害视觉模块

部署位置：`D:\曲钧天ROV-CV`。这是一个等待真实数据接入的 PyTorch 项目骨架，包含可独立训练的病害检测、裂缝二值分割和可选图像增强模型，以及一个组合推理接口。

**交付状态：仅创建代码和文档，未安装依赖、未执行/测试项目代码、未启动 conda、未训练或下载权重。目录中不包含甲方图像、虚构训练数据或已训练模型。** 以下命令供用户之后在自己指定的环境中手动执行。没有权重时不能进行有效推理，也没有经过验证的病害识别精度。

## 1. 模型职责与数据流

| 模块 | 当前基线 | 监督数据 | 输出 | 独立入口 |
| --- | --- | --- | --- | --- |
| 病害检测 | torchvision Faster R-CNN / ResNet-50 FPN | RGB 图像 + COCO 边界框 | 病害类别、置信度、边界框 | `rov_cv.train_detection` |
| 裂缝分割 | 小型两级 U-Net | RGB 图像 + 同尺寸二值裂缝掩膜 | 每像素裂缝概率及二值掩膜 | `rov_cv.train_segmentation` |
| 可选图像增强 | 小型残差 CNN | 几何对齐的水下输入/参考图像对 | RGB 增强图 | `rov_cv.train_enhancement` |

三个训练入口各自创建模型、优化器和数据加载器，不会调用另一个模型，也不共享权重。检测是目标检测，不额外提供整图分类头；默认类别为裂缝、剥落、露筋、锈蚀，属于可调整的初始标签表，须与甲方标注规范确认。裂缝分割是前景/背景二分类，不是全部病害的多类别分割。

推理数据流：

```text
原始图像 -> EXIF 方向校正 -> RGB
                            |
                   可选：图像增强权重
                            |
                    当前 RGB 图像
                     /            \
           检测模型（可选）      裂缝分割模型（可选）
                     \            /
                  原尺寸结果 + JSON + 叠加图
```

检测与分割默认针对整张图像独立执行，分割不依赖检测框，也没有检测框裁剪或拼接逻辑。未提供增强权重时，检测和分割直接使用原始 RGB 图像。增强只在显式传入 `--enhancement` 时启用。训练阶段不会自动增强输入；如要在增强图上训练检测/分割，应提前准备一致的图像版本，并保持几何坐标和标注对齐。

## 2. 目录架构

```text
D:\曲钧天ROV-CV\
├── requirements.txt
├── read.md
├── .gitignore
├── configs/
│   ├── detection.yaml           # 检测类别、图像缩放、训练参数
│   ├── segmentation.yaml        # 分割尺寸、阈值、训练参数
│   └── enhancement.yaml         # 增强尺寸、配对数据、训练参数
├── rov_cv/
│   ├── __init__.py
│   ├── common.py               # 路径、RGB 读取、张量转换、设备选择
│   ├── datasets.py             # COCO 与图像/掩膜、图像/图像配对读取
│   ├── models.py               # 模型构建与可替换的网络定义
│   ├── training.py             # 训练循环、验证指标、权重保存
│   ├── train_detection.py      # 仅训练检测
│   ├── train_segmentation.py   # 仅训练分割
│   ├── train_enhancement.py    # 仅训练增强
│   ├── inference.py            # Python 组合推理接口 VisionPipeline
│   └── predict.py              # 命令行单图推理
├── data/
│   └── README.md               # 用户另行放置各任务的数据集
├── weights/                    # 训练完成后生成各任务权重
└── outputs/                    # 用户指定的单图推理输出
```

已建立下一节使用的数据空目录，图像和标注需由用户另行放入。所有配置中的相对路径、命令行图像/权重/输出相对路径都以项目根目录为基准，也支持 Windows 绝对路径。为确保能导入本地包，下述 `python -m` 命令应在项目根目录执行。Windows YAML 绝对路径建议使用 `D:/datasets/...` 或单引号 `'D:\datasets\...'`，避免双引号中的反斜杠转义。

## 3. 依赖与环境

建议使用 Python 3.10 或 3.11，在用户自行选定并已激活的环境中安装依赖；本项目不创建或激活 conda 环境。

```powershell
Set-Location -LiteralPath 'D:\曲钧天ROV-CV'
python -m pip install -r requirements.txt
```

`torch` 和 `torchvision` 必须是相互兼容的一对。GPU 版应先依据本机驱动和安装渠道选择对应的 PyTorch/torchvision 安装包，再安装其余依赖；`requirements.txt` 的版本范围不保证任意组合均兼容，也不保证自动安装到需要的 CUDA 版本。代码不会自动安装依赖。

配置 `device: auto` 会在用户运行时选择可用的 CUDA，否则选择 CPU；可改成 `cpu` 或 `cuda:0`。Windows 下默认 `workers: 0`，确认环境后再调整。显存不足时先减小 `batch_size` 和图像尺寸。`requirements.txt` 给出基础依赖范围，尚未在用户机器验证或生成精确版本锁定文件。

所有模型默认从随机初始化开始，检测同时设置 `weights=None` 与 `weights_backbone=None`，不会自动下载预训练权重。小规模样本通常不足以支撑从零训练检测模型；后续可按实际数据量，在 `models.py` 中增加显式的预训练权重接入，类别头仍需匹配。

## 4. 训练数据接口

通用图像格式：`.jpg/.jpeg/.png/.bmp/.tif/.tiff`，推荐 8 位三通道 JPEG/PNG。程序读取 EXIF 校正后的 RGB 图像；不接收视频流、深度图、声呐数据、URL 或 base64。多帧文件只使用 Pillow 默认打开的一帧；RGBA 转 RGB 不使用透明度。标注必须对应方向校正后的像素布局，最好在标注前统一图像方向。

按桥梁、采集批次或视频序列划分训练集和验证集，避免相邻帧分到不同集合。三个任务都要求有非空 train 与 val 数据集，未实现自动随机划分或合成数据。

### 4.1 检测：COCO 边界框

```text
data/detection/
├── train/
│   ├── images/scene_001.jpg
│   └── annotations.json
└── val/
    ├── images/scene_101.jpg
    └── annotations.json
```

每份标注的最小结构如下，文件名、图像尺寸与边界框请替换为真实值；示例本身不是交付数据：

```json
{
  "images": [
    {"id": 1, "file_name": "scene_001.jpg", "width": 1920, "height": 1080}
  ],
  "categories": [
    {"id": 1, "name": "crack"},
    {"id": 2, "name": "spalling"},
    {"id": 3, "name": "exposed_rebar"},
    {"id": 4, "name": "corrosion"}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [100, 200, 300, 80], "area": 24000, "iscrowd": 0}
  ]
}
```

- COCO 输入框为 `[x, y, width, height]`，左上角原点，单位为原图像素。框必须有限、宽高为正且完全位于图像范围内；不会自动裁剪无效框。
- `file_name` 必须是 `images` 目录内的相对路径，可含子目录。`image_id` 必须引用存在的图像 ID，图像 ID 唯一。
- `categories.name` 必须与 `configs/detection.yaml` 中的前景类别名完全一致。COCO category ID 可以不连续，程序按名称映射到连续训练 ID；`background` 固定为配置索引 0，不写入 COCO 前景类别。
- 支持无目标图像：保留 `images` 条目，不为该图添加 annotations。当前不支持 `iscrowd=1`。
- 检测读取后变为 RGB `float32 [3,H,W]`、范围 `[0,1]`；torchvision 内部负责归一化及保持宽高比的缩放。target 包含 `boxes: float32 [N,4]`（xyxy）、`labels: int64 [N]` 等字段。不要在数据集外再执行 torchvision 均值方差归一化。

### 4.2 分割：图像与二值掩膜

```text
data/segmentation/
├── train/
│   ├── images/scene_001.jpg
│   └── masks/scene_001.png
└── val/
    ├── images/scene_101.jpg
    └── masks/scene_101.png
```

目录为扁平结构，图像与 mask 文件同名（扩展名可不同），mask 固定为 `.png`。同一图像目录不能出现相同主文件名的多种扩展名。mask 原尺寸必须与图像一致，使用单通道或索引 PNG，像素 0=背景，1 或 255=裂缝；不接收 RGB 彩色标注或含其他像素值的多类别掩膜。无裂缝图像对应全 0 mask。

读取后图像是 `[3,H,W]` RGB `[0,1]`，mask 是 `[1,H,W]` float32 0/1。训练与推理将图像双线性缩放到 `model.image_size: [height,width]`；训练 mask 使用最近邻缩放。尺寸必须为不小于 4 的 4 的倍数。固定尺寸缩放会改变非正方形图像的宽高比，细裂缝可能丢失；高分辨率切片和重叠融合留作后续扩展。

### 4.3 增强：几何对齐的配对图像

```text
data/enhancement/
├── train/
│   ├── input/scene_001.jpg
│   └── target/scene_001.png
└── val/
    ├── input/scene_101.jpg
    └── target/scene_101.png
```

采用扁平目录，同名主文件名配对，目标可用不同图像扩展名。输入和参考图必须具有相同原尺寸、视角、几何位置及对应内容；不能用不配对的“清晰水下图”作为 target。模型输入输出均为 RGB `[B,3,H,W]`、范围 `[0,1]`。没有可靠配对参考图时可完全跳过该模块，当前没有无监督增强训练器。

## 5. 独立训练方法

先准备对应任务的 train/val 数据，再修改该任务配置中的路径、类别和超参数。以下三条命令独立选择执行，没有顺序依赖：

```powershell
python -m rov_cv.train_detection --config configs/detection.yaml
python -m rov_cv.train_segmentation --config configs/segmentation.yaml
python -m rov_cv.train_enhancement --config configs/enhancement.yaml
```

| 任务 | 损失 | 每轮验证 | best.pt 选择依据 |
| --- | --- | --- | --- |
| 检测 | torchvision 返回的分类、回归、RPN 损失之和 | 按类别匹配、IoU≥0.5、置信度阈值筛选后的总体 precision/recall/F1 | F1 最大 |
| 分割 | BCEWithLogits + soft Dice loss | 全部验证像素汇总的 Dice、IoU | Dice 最大 |
| 增强 | RGB L1 | `[0,1]` RGB 的逐元素平均绝对误差 MAE | MAE 最小 |

检测验证使用按置信度排序的一对一贪心匹配；它是骨架级检查指标，不是 COCO mAP，也不包含完整的多 IoU/多尺度评估。分割的预测和真值全部为空时 Dice/IoU 记为 1；检测完全没有 TP、FP、FN 时 F1 记为 0。薄裂缝、类别不均衡和空白样本较多时需结合可视化与每类别指标进一步评估。

训练后对应目录产生：

```text
weights/detection/       # 或 segmentation/、enhancement/
├── config.json         # 本次配置快照
├── history.json        # 每轮训练平均批次损失、验证指标
├── last.pt             # 最近一轮权重
└── best.pt             # 当前最优验证指标权重
```

checkpoint 包含 `format_version/task/epoch/config/model_state/optimizer_state/metrics`。当前训练入口不提供断点续训参数；虽然保留优化器状态，恢复训练需另行实现。输出目录已有 `last.pt`、`best.pt` 或 `history.json` 时会拒绝开始，下一次训练请使用新的 `train.output_dir`。推理以 checkpoint 内嵌的类别和模型配置构建网络，训练后修改 YAML 不会改变既有权重的含义。

## 6. 单图推理

至少指定一项已训练的权重；不能用缺失权重或随机模型替代。

仅检测：

```powershell
python -m rov_cv.predict --image 'D:\inspection\frame001.jpg' --detection weights/detection/best.pt --output outputs/frame001_detection
```

仅裂缝分割：

```powershell
python -m rov_cv.predict --image 'D:\inspection\frame001.jpg' --segmentation weights/segmentation/best.pt --output outputs/frame001_segmentation
```

检测和分割组合：

```powershell
python -m rov_cv.predict --image 'D:\inspection\frame001.jpg' --detection weights/detection/best.pt --segmentation weights/segmentation/best.pt --score-threshold 0.5 --mask-threshold 0.5 --device auto --output outputs/frame001_combined
```

显式启用增强：

```powershell
python -m rov_cv.predict --image 'D:\inspection\frame001.jpg' --enhancement weights/enhancement/best.pt --detection weights/detection/best.pt --segmentation weights/segmentation/best.pt --output outputs/frame001_enhanced
```

也允许只指定增强权重导出增强图。阈值参数范围为 `[0,1]`，未指定时使用对应 checkpoint 内的配置（默认 0.5）。使用不同的输出目录保存每张图像或每次实验；目录中已有本接口的结果文件时会拒绝覆盖。

## 7. 图像输出与 JSON 接口

| 文件 | 条件 | 内容 |
| --- | --- | --- |
| `result.json` | 总是 | 输入尺寸、执行模块、检测列表、分割统计和产物相对路径 |
| `overlay.png` | 总是 | 当前 RGB 图像，绿色检测框、红色裂缝叠加；框标签使用数字类别 ID 与分数 |
| `crack_mask.png` | 运行分割 | 原尺寸单通道 PNG，0=背景，255=裂缝 |
| `enhanced.png` | 运行增强 | 恢复到原尺寸的 RGB 增强图 |

所有输出图像均与 **EXIF 校正后的原图** 同宽高。检测框由 torchvision 恢复到该尺寸；分割先对模型输出执行 sigmoid，再将概率双线性插值回原尺寸，最后阈值化。增强先按配置缩放输入，再将输出双线性插值回原尺寸。因此增强尺寸恢复不代表保留了全部原始细节。

下面是字段结构示意，数值仅为说明接口，不是运行结果：

```json
{
  "schema_version": "1.0",
  "image_path": "D:\\inspection\\frame001.jpg",
  "width": 1920,
  "height": 1080,
  "coordinate_space": "exif_corrected_original_pixels",
  "bbox_format": "xyxy",
  "enhancement_applied": false,
  "detection_executed": true,
  "segmentation_executed": true,
  "detection_score_threshold": 0.5,
  "detections": [
    {"class_id": 1, "class_name": "crack", "score": 0.91,
     "bbox_xyxy": [100.0, 200.0, 400.0, 280.0]}
  ],
  "segmentation": {
    "class_name": "crack",
    "mask_path": "crack_mask.png",
    "threshold": 0.5,
    "crack_pixels": 1200,
    "crack_area_ratio": 0.0005787037037037037
  },
  "artifacts": {"crack_mask": "crack_mask.png", "overlay": "overlay.png"}
}
```

- 坐标原点为左上角，x 向右、y 向下；`bbox_xyxy=[xmin,ymin,xmax,ymax]` 为浮点像素边界，范围为 x∈[0,width]、y∈[0,height]，不是归一化坐标，也不是 COCO 输入的 xywh。
- `class_id` 是 checkpoint 中 `classes` 列表的索引，背景为 0，输出前景从 1 开始；不一定等于原始 COCO category ID。`score` 是模型置信度，不是已校准的工程可靠性。
- `detections=[]` 可能表示未执行检测，也可能表示无框通过阈值，使用 `detection_executed` 区分。
- 未运行分割时 `segmentation=null`；`crack_area_ratio` 为全图裂缝像素占比，不是检测框内占比，也不表示实际裂缝面积。
- JSON 中所有产物路径以本次输出目录为基准，`image_path` 为绝对路径。增强启用时，artifacts 还包含 `enhanced_image: "enhanced.png"`。
- 当前不输出裂缝实例 ID、骨架、真实长度/宽度或物理面积。工程尺度测量需另加相机标定、距离与尺度信息。

## 8. Python 集成接口

供后续 ROV 上位机或离线处理程序调用；以下示例需用户自行在环境中运行：

```python
from rov_cv.inference import VisionPipeline

pipeline = VisionPipeline(
    detection="weights/detection/best.pt",
    segmentation="weights/segmentation/best.pt",
    enhancement=None,
    device="auto",
)
result = pipeline.predict(
    image_path=r"D:\inspection\frame001.jpg",
    output_dir="outputs/frame001_python",
    score_threshold=0.5,
    mask_threshold=0.5,
)
```

`VisionPipeline(...)` 只在构造时加载指定权重，可复用同一实例依次处理多张图像，但每张图须使用独立输出目录。`predict(...)` 接受本地图像文件路径，返回与 `result.json` 相同的 Python dict，并写入图像和 JSON。当前没有直接接收 NumPy/OpenCV 数组的公开入口；内部张量采用 RGB/CHW，与 OpenCV 默认 BGR/HWC 不同。

本版本为同步单图接口，未实现摄像头采集、视频解码、批量目录命令、REST 服务或并发调度。调用失败时直接抛出 Python 异常，不将文件缺失、坏图、错误标注或权重不匹配伪装成“未发现病害”。异常中断可能留下部分产物，重新运行时请使用新的输出目录。只加载可信来源、与本代码架构匹配的 checkpoint；代码使用 `torch.load(..., weights_only=True)`。

## 9. 后续替换和接入建议

1. 甲方图像到位后，先统一标签定义、图像方向、分辨率、桥梁/批次划分和质量筛选，再制作独立验证集。
2. 保持 `datasets.py` 的 target 协议与 `inference.py` 的结果协议，在 `models.py` 替换检测器、U-Net 或增强网络；网络改变后需重新训练并使用匹配权重。
3. 检测可补充完整 COCO mAP、类别召回和漏检分析；分割可增加细裂缝切片、边界指标和误差复核；增强需比较原图/增强图对下游结果的实际影响。
4. 当前没有数据增强、学习率调度、混合精度、早停、TensorBoard、模型导出或性能基准。基础实现用于明确模块边界和数据接口，部署速度与识别效果需在后续真实数据和指定环境中验证。

本次交付没有执行本文件中的任何安装、训练或推理命令。
