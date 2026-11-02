# 模型结果数据与后续处理指南

适用项目：`D:\曲钧天ROV-CV`。本文依据当前 `training.py` 与 `inference.py` 的实际输出编写。输入准备见 [image_init.md](image_init.md)，完整运行方法见 [read.md](read.md)。所有示例值仅说明格式，不是模型实际运行或精度验证结果。

## 1. 区分训练产物与图像推理结果

| 阶段 | 产物位置示例 | 内容 | 用途 |
| --- | --- | --- | --- |
| 检测训练 | `weights/detection/` | 配置、指标、检测 checkpoint | 检查训练过程、选取检测模型 |
| 分割训练 | `weights/segmentation/` | 配置、指标、分割 checkpoint | 选取裂缝分割模型 |
| 可选增强训练 | `weights/enhancement/` | 配置、指标、增强 checkpoint | 选取图像增强模型 |
| 单图推理 | 用户指定的 `outputs/<图像或实验编号>/` | JSON、叠加图、可选 mask/增强图 | 上位机显示、统计、人工复核与归档 |

`.pt` 是模型参数文件，不是某张图的病害识别报告。对实际图像进行推理后，才会得到该图的 `result.json` 等文件。当前没有提供已训练权重，必须先用真实数据训练或接入格式匹配的可信权重。

## 2. 训练结束后得到什么

每个训练任务独立保存：

```text
weights/<task>/
├── config.json
├── history.json
├── last.pt
└── best.pt
```

- `config.json`：本次训练配置快照，包含任务、类别、数据路径、网络结构和参数。
- `history.json`：按 epoch 记录平均批次训练损失及该轮验证指标的 JSON 数组。
- `last.pt`：最近完成一轮的 checkpoint。
- `best.pt`：按该任务验证指标选择的 checkpoint；不一定是最后一轮。

checkpoint 包含 `format_version/task/epoch/config/model_state/optimizer_state/metrics`。推理使用其中的模型配置和权重，类别由 checkpoint 内的 `classes` 决定；仅修改外部 YAML 不会改变已有模型的类别含义。保留了优化器状态，但当前训练入口尚未实现断点续训。

### 验证指标怎样理解

| 任务 | history.json 指标 | best.pt 规则 | 阅读限制 |
| --- | --- | --- | --- |
| 检测 | `train_loss`、`precision_at_iou_0.5`、`recall_at_iou_0.5`、`f1_at_iou_0.5`、`tp/fp/fn` | F1 最大 | 使用配置的分数阈值，按类别及 IoU≥0.5 贪心匹配，是总体指标，不是 COCO mAP |
| 分割 | `train_loss`、`pixel_dice`、`pixel_iou` | Dice 最大 | 全部验证图像像素汇总；不能单独证明细裂缝、边界或每张图均识别良好 |
| 增强 | `train_loss`、`mae_rgb_0_1` | MAE 最小 | 对 RGB 配对参考图的平均绝对误差，不直接衡量病害检测是否改善 |

损失值在不同任务之间不可直接比较。分割验证预测与真值均无任何前景时 Dice/IoU 为 1；检测 TP、FP、FN 均为 0 时 F1 为 0。训练损失下降后仍应检查独立场景上的漏检、误检和掩膜质量。当前没有自动生成评价曲线或测试集报告。

## 3. 单图推理会输出哪些文件

例如同时加载检测和分割权重：

```powershell
Set-Location -LiteralPath 'D:\曲钧天ROV-CV'
python -m rov_cv.predict --image 'D:\inspection\frame001.jpg' --detection weights/detection/best.pt --segmentation weights/segmentation/best.pt --output outputs/frame001
```

启用增强时另外提供 `--enhancement weights/enhancement/best.pt`。至少提供一个任务权重。下列文件按执行模块生成：

| 文件 | 格式、尺寸 | 产生条件 | 推荐处理方法 |
| --- | --- | --- | --- |
| `result.json` | UTF-8 JSON | 成功完成推理时总是生成 | 程序读取结构化结果、保存图像关联信息 |
| `overlay.png` | RGB、原图尺寸 | 总是生成 | 显示和人工复核，不用来计算裂缝像素 |
| `crack_mask.png` | 单通道 PNG、原图尺寸，0/255 | 执行分割 | 像素统计、连通域、骨架分析的输入 |
| `enhanced.png` | RGB PNG、原图尺寸 | 执行增强 | 原图/增强图比较、增强处理记录 |

“原图尺寸”指 **EXIF 方向校正后** 的宽高；若原图仅依赖 EXIF 旋转显示，不能把坐标直接套到未校正的原始像素阵列上。增强启用时，检测、分割和叠加图使用增强后的图像；未启用时使用原始 RGB 图像。

每次推理应使用新的独立输出目录。目录中已有 `result.json/enhanced.png/crack_mask.png/overlay.png` 任一文件时，当前接口拒绝覆盖。异常可能留下部分产物，结果接收端应确认 `result.json` 及其引用文件完整后再入库。

## 4. result.json 字段解释

格式示例：

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
    {
      "class_id": 1,
      "class_name": "crack",
      "score": 0.91,
      "bbox_xyxy": [100.0, 200.0, 400.0, 280.0]
    }
  ],
  "segmentation": {
    "class_name": "crack",
    "mask_path": "crack_mask.png",
    "threshold": 0.5,
    "crack_pixels": 1200,
    "crack_area_ratio": 0.0005787037037037037
  },
  "artifacts": {
    "crack_mask": "crack_mask.png",
    "overlay": "overlay.png"
  }
}
```

| 字段 | 含义与单位 |
| --- | --- |
| `schema_version` | 结果结构版本，当前为字符串 `1.0` |
| `image_path` | 输入文件绝对路径，跨机器归档时还应另存稳定图像 ID |
| `width/height` | 方向校正后的原图宽/高，单位像素 |
| `coordinate_space` | 当前固定为 `exif_corrected_original_pixels` |
| `bbox_format` | 当前固定为 `xyxy` |
| `enhancement_applied` | 是否使用了增强模型 |
| `detection_executed/segmentation_executed` | 是否执行了对应模块 |
| `detection_score_threshold` | 检测输出筛选阈值，仅执行检测时出现 |
| `detections` | 通过分数阈值的检测列表；没有框时为 `[]` |
| `segmentation` | 分割统计对象；未执行分割时为 `null` |
| `artifacts` | 生成文件的相对路径，以本次输出目录为基准 |

Python 接口 `VisionPipeline.predict(...)` 返回与该 JSON 相同结构的 dict，并同步写入文件。不能只用空列表判断“无病害”：必须先检查模块是否执行，以及结果文件是否完整。

## 5. 检测结果怎样处理

每个目标包含 `class_id/class_name/score/bbox_xyxy`：

- `class_id` 为 checkpoint 类别列表索引，0 是背景，前景从 1 起；它不保证等于原始 COCO category ID。
- `class_name` 为该模型类别名称，例如 `crack`，当前标签表见 [image_init.md](image_init.md)。
- `score` 为模型置信度，不是工程风险概率或已经校准的可靠性。
- 框为 `[xmin,ymin,xmax,ymax]`，左上角为原点，x 向右、y 向下，单位为浮点原图像素边界，范围 x∈[0,width]、y∈[0,height]。

输入 COCO 使用 xywh，输出使用 xyxy。例如 `[100,200,400,280]` 对应宽 300、高 80；转回 xywh 时计算 `[xmin,ymin,xmax-xmin,ymax-ymin]`，不能直接照抄四个数。

建议的下游处理顺序：

1. 根据 `detection_executed` 确认模块已运行，再按 `class_name` 分组统计目标数。
2. 根据验证集漏检/误检表现选择阈值，通过 `--score-threshold` 在推理时设置。Faster R-CNN 内部已有检测后处理/NMS，现有推理入口又执行分数筛选；不要默认把同一病害多框直接累加为物理面积。
3. 需要裁剪病害小图时，将左上角向下取整、右下角向上取整，裁到图像合法范围；使用相同 EXIF 校正视图，保存原图 ID 与原始浮点框。
4. 在界面中显示原图/叠加图、类别和置信度，记录人工复核结论。`overlay.png` 上数字标签对应 `class_id`，完整名称在 JSON 内。

降低阈值需要重新推理：已经被过滤掉的框不会保存在 JSON 中。`detections=[]` 只表示没有保留下来的候选框，不能证明图像绝对无病害。

## 6. 分割结果怎样处理

模型内部输出 `[B,1,H,W]` logits；推理将其 sigmoid 转成概率，双线性插值到原图尺寸，再按阈值生成二值掩膜。当前仅保存最终 `crack_mask.png`，**不保存概率图或 logits**。

mask 为 0=背景、255=裂缝，应按“像素值大于 0”生成布尔数组，再统计像素数量，不能把 255 的总和当成裂缝面积。

```text
crack_pixels = mask 中非零像素的数量
crack_area_ratio = crack_pixels / (width × height)
裂缝像素占比百分数 = crack_area_ratio × 100%
```

占比范围为 0～1，统计整个图像，不限于检测框内部。检测与分割是两个独立任务，没有逐实例对应关系：一个检测框可包含多段掩膜，框外也可能出现分割前景。框与 mask 不一致时应保留两者并复核，不应自动宣布某一方正确。

### 可另行实现的后处理

以下操作当前尚未写入项目，仅作为下游处理设计：

| 操作 | 用途 | 需要注意 |
| --- | --- | --- |
| 连通域分析 | 提取候选裂缝区域、像素面积、外接框 | 相交裂缝可能连成一个区域，一个裂缝也可能被预测成多段，连通域数量不等于真实裂缝条数 |
| 小区域过滤 | 去除孤立噪声 | 阈值必须依据分辨率和验证数据确定，过大可能删除细裂缝 |
| 开/闭运算 | 平滑噪声、连接小间隙 | 会改变裂缝宽度、连通性和像素面积，应保留原始 mask 与参数 |
| 骨架化、距离变换 | 估计中心线和像素级宽度 | 需要处理分叉、断裂、边缘偏差，不能直接当成真实长度或宽度 |
| 检测框与 mask 相交 | 形成候选区域内的像素统计 | 属于新增融合规则，可能舍弃框外真实裂缝，须单独验证 |

后处理后另存例如 `crack_mask_cleaned.png` 和独立统计文件，保留当前模型原始输出，方便追踪差异。若要改变分割阈值，应通过 `--mask-threshold` 重新推理；仅凭现有 0/255 PNG 无法恢复原始概率并重新阈值化。

## 7. 增强结果怎样处理

增强模型输出 RGB `[B,3,H,W]`，裁到 `[0,1]`，恢复到原图尺寸后量化为 8 位并保存为 `enhanced.png`；随后检测/分割读取该增强 RGB 图像。输出尺寸恢复不表示原始细节已恢复，网络也可能改变色彩或纹理表现。

建议保留原图及增强图进行并排复核，并用同一独立数据集比较“原图检测/分割”与“增强后检测/分割”的表现。肉眼更亮或更鲜艳不代表下游更准确。没有增强权重时，`enhancement_applied=false`，也不会生成 `enhanced.png`。

## 8. 上位机接收、坐标显示与归档

建议按以下顺序处理每张图像：

1. 创建唯一输出目录，调用推理；只有成功返回且产物完整时才登记为已完成。当前异常会直接抛出，不返回伪造的空检测报告。
2. 读取 `result.json`，检查结构版本、图像尺寸、模块执行标志及 artifacts 路径；加载原图、overlay 和可选 mask。
3. 上位机显示图像缩放为 `display_width × display_height` 时，同步按 `sx=display_width/width`、`sy=display_height/height` 缩放框坐标。若显示控件使用留白或 letterbox，还应加上图像在控件内的偏移；mask 显示缩放使用最近邻。
4. 归档图像 ID、原图、JSON、mask、overlay、所用权重版本、阈值、增强开关以及人工复核结论。当前 JSON 不自动包含时间戳、权重路径/hash、桥梁编号或相机参数，这些需由调用方在独立记录中补充。
5. 若用于 CSV/数据库，一行可存一个检测目标；图像级表保存处理状态和裂缝占比。无检测目标图像也应保留图像级记录，区分未运行、运行失败、无框通过阈值。

当前无视频多帧去重、目标跟踪或跨帧同一病害关联。逐帧累加检测数会重复计算同一病害，汇总报告应另加时空关联及人工复核流程。

## 9. 像素结果与工程尺度的边界

当前框、mask、`crack_pixels` 和 `crack_area_ratio` 都属于图像空间结果，没有毫米/厘米的真实量纲。实际裂缝长度、宽度和面积需要相机标定、畸变校正、尺度或距离信息，以及适合水下拍摄几何的换算与验证。不同距离和视角的像素面积不宜直接比较病害严重程度。

本次交付仅增加说明文档，没有执行上述推理命令、后处理操作或训练，也没有生成实际识别结果。
