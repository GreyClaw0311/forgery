# 图像篡改检测系统

> 基于 OpenCV 特征提取和机器学习的图像篡改检测服务

---

## 项目简介

本系统提供图像篡改检测能力，采用**两阶段串联架构**，能够精准判断图片是否篡改并定位篡改区域。

### 核心特点

- ✅ **两阶段检测**: GB分类器前置判断 + 区域定位算法
- ✅ **多算法支持**: ELA、DCT、Fusion、ML 四种检测算法
- ✅ **像素级检测**: ML 算法支持像素级篡改区域定位
- ✅ **篡改区域坐标**: 输出篡改区域矩形坐标
- ✅ **篡改区域标注**: 所有算法在原图上绘制矩形框标注篡改区域
- ✅ **GPU 加速**: 支持 PyTorch GPU 加速推理
- ✅ **REST API**: FastAPI 服务，易于集成
- ✅ **完整日志**: 自动保存到 ./logs 目录，每30天清理
- ✅ **高并发支持**: 服务端默认8个 Worker，测试脚本支持并发测试
- ✅ **一键启停**: start.sh / stop.sh 脚本，支持前台/后台运行
- ✅ **测试工具**: 完整的单张/批量测试脚本，支持并发加速

---

## 检测流程

```
┌─────────────────────────────────────────────────────────────┐
│                      输入图片                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  第一阶段: GB分类器 (前置检测)                                │
│  - 判断图片是否篡改                                          │
│  - 输出篡改置信度                                            │
│  - 如果判断为正常，直接返回，跳过后续检测                       │
└─────────────────────────────────────────────────────────────┘
                              │
                    置信度 > 阈值?
                    /          \
                  否            是
                  │             │
                  ▼             ▼
┌──────────────────┐   ┌───────────────────────────────────────┐
│  返回: 正常图片   │   │  第二阶段: 区域定位算法                 │
│  无需后续处理     │   │  - ELA: JPEG压缩误差分析              │
└──────────────────┘   │  - DCT: DCT块效应检测                 │
                       │  - Fusion: 多特征融合                 │
                       │  - ML: 像素级机器学习                 │
                       └───────────────────────────────────────┘
                                        │
                                        ▼
                       ┌───────────────────────────────────────┐
                       │  输出: 篡改区域掩码 + 坐标              │
                       └───────────────────────────────────────┘
```

---

## 目录结构

```
forgery/
├── README.md                    # 项目说明
├── requirements.txt             # 服务依赖
│
├── data/                        # 测试数据
│   └── tamper_data/            # 测试图片
│       ├── easy/               # 简单篡改
│       ├── difficult/          # 困难篡改
│       └── good/               # 正常图片
│
├── docs/                        # 文档
│   ├── api_document.md         # 接口文档
│   └── experiment_report_*.md  # 实验报告
│
├── train/                       # 训练模块 (离线使用)
│   ├── gb_classifier/          # GB 分类器训练
│   └── pixel_segmentation/     # 像素级模型训练
│
└── release/                     # 推理服务 (在线部署)
    ├── server_forgrey.py       # FastAPI 主程序
    ├── test_service.py         # 测试脚本 ⭐
    ├── start.sh                # 服务启动脚本 ⭐
    ├── stop.sh                 # 服务停止脚本 ⭐
    ├── requirements.txt        # 服务依赖
    │
    ├── algorithms/             # 算法实现
    │   ├── ela_detector.py     # ELA 检测器
    │   ├── dct_detector.py     # DCT 检测器
    │   ├── fusion_detector.py  # 融合检测器
    │   ├── ml_detector.py      # ML 检测器
    │   └── features.py         # 特征提取模块
    │
    ├── models/                 # 模型文件 (需上传)
    │   ├── gb_classifier/      # GB 分类器
    │   │   ├── model.pkl
    │   │   ├── scaler.pkl
    │   │   └── config.json
    │   └── pixel_segmentation/ # 像素级模型
    │       └── model.pkl
    │
    └── utils/                  # 工具函数
        └── postprocess.py      # 后处理
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型文件

模型文件需要放在 `release/models/` 目录：

```
release/models/
├── gb_classifier/           # GradientBoost 分类器 (前置检测)
│   ├── model.pkl
│   ├── scaler.pkl
│   └── config.json
└── pixel_segmentation/      # 像素级分割模型 (区域定位)
    └── model.pkl
```

### 3. 启动服务

**方式一：使用启动脚本 (推荐)**

```bash
cd release

# 前台启动 (默认端口 8000)
./start.sh

# 指定端口启动
./start.sh -p 8080

# 后台启动 (守护进程模式)
./start.sh -d

# 后台启动并指定端口
./start.sh -d -p 8080
```

**方式二：直接运行**

```bash
cd release
python server_forgrey.py          # 默认端口 8000
python server_forgrey.py --port 8080  # 指定端口
```

**服务特性:**
- 默认监听 `0.0.0.0:8000`
- 默认 **8个 Worker** 并发处理
- 日志自动保存到 `./logs` 目录
- 日志每 **30天** 自动清理一次
- 日志轮转：每个文件最大 10MB，保留5个备份

**停止服务:**

```bash
cd release
./stop.sh           # 停止服务
./stop.sh -f        # 强制停止
```

服务启动后访问: http://localhost:8000

### 4. 测试服务

```bash
# 单张测试
python test_service.py --mode single --image test.jpg --algorithm ml

# 批量测试 (默认保存图片，默认8并发)
python test_service.py --mode batch --data_dir ./data/tamper_data --algorithm ml

# 批量测试 (指定并发数)
python test_service.py --mode batch --data_dir ./data/tamper_data --algorithm ml --workers 16

# 批量测试 (不保存图片)
python test_service.py --mode batch --data_dir ./data/tamper_data --algorithm ml --no_save_images

# 全算法对比测试
python test_service.py --mode compare --data_dir ./data/tamper_data

# 采集发版数据
python test_service.py --mode release --data_dir ./test --dataset_name TestSet
```

**测试脚本参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --mode | single | 测试模式 (single/batch/compare/release) |
| --image | - | 单张测试图片路径 |
| --data_dir | ./test | 数据目录 |
| --algorithm | ml | 算法 (ela/dct/fusion/ml/all) |
| --server | http://localhost:8000 | 服务地址 |
| --output | ./test_results | 输出目录 |
| --workers | 8 | 并发线程数 |
| --save_images | True | 保存结果图片 (默认开启) |
| --no_save_images | - | 不保存结果图片 |

**结果图片格式:**
- 左侧：原图
- 右侧：篡改区域标注（红色轮廓 + 绿色矩形框）

---

## API 使用方法

### 健康检查

```bash
curl http://localhost:8000/health
```

**响应:**

```json
{
  "status": "0001:服务运行正常.",
  "timestamp": "2026-03-24T10:00:00",
  "algorithms": ["ela", "dct", "fusion", "ml"],
  "gb_classifier_loaded": true
}
```

### 检测图片

```bash
curl -X POST "http://localhost:8000/tamper_detection/v1/tamper_detect_img" \
  -d "image_base64=$(base64 -w 0 test.jpg)" \
  -d "algorithm=ml"
```

**响应 (正常图片):**

```json
{
  "status": "0001:解析成功.",
  "is_tampered": false,
  "confidence": 0.15,
  "gb_confidence": 0.15,
  "mask_base64": null,
  "marked_image_base64": null,
  "tamper_regions": null,
  "algorithm": "ml",
  "processing_time": 0.05
}
```

**响应 (篡改图片):**

```json
{
  "status": "0001:解析成功.",
  "is_tampered": true,
  "confidence": 0.92,
  "gb_confidence": 0.95,
  "mask_base64": "iVBORw0KGgoAAAANS...",
  "marked_image_base64": "/9j/4AAQSkZJRgABAQ...",
  "tamper_regions": [
    {
      "left_top": [120, 85],
      "right_bottom": [276, 227]
    }
  ],
  "algorithm": "ml",
  "processing_time": 0.35
}
```

### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image_base64 | string | 是 | 图片 Base64 编码 |
| algorithm | string | 否 | 算法名称: ela/dct/fusion/ml (默认 ml) |
| skip_gb | boolean | 否 | 跳过 GB 前置检测 (默认 false) |

### 返回字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| is_tampered | boolean | 是否篡改 |
| confidence | float | 最终置信度 |
| gb_confidence | float | GB 分类器置信度 |
| mask_base64 | string | 篡改区域掩码 Base64 (二值图) |
| marked_image_base64 | string | 标记后的图片 Base64 (原图+篡改矩形框) |
| tamper_regions | array | 篡改区域坐标列表 |
| processing_time | float | 处理时间 (秒) |

---

## 测试脚本使用

### 单张图片测试

```bash
python test_service.py --mode single \
    --image ./data/tamper_data/difficult/images/cpmv_0001.jpg \
    --algorithm ml \
    --save_images
```

### 批量测试

```bash
# 单算法测试
python test_service.py --mode batch \
    --data_dir ./data/tamper_data \
    --algorithm ml \
    --save_images

# 全算法测试
python test_service.py --mode batch \
    --data_dir ./data/tamper_data \
    --algorithm all \
    --save_images
```

### 多算法对比

```bash
python test_service.py --mode compare \
    --data_dir ./data/tamper_data
```

### 测试报告示例

```
============================================================
测试报告
============================================================

【测试信息】
  数据目录: ./data/tamper_data
  测试算法: ml
  测试时间: 2026-03-26T10:30:00
  测试图片: 100/100

【分类指标】(二分类)
  Accuracy:  0.9200 (92/100)
  Precision: 0.9452
  Recall:    0.8936
  F1-score:  0.9187
  FPR (假阳性率): 0.0548
  FNR (假阴性率): 0.1064

  混淆矩阵:
              预测篡改  预测正常
  实际篡改        42        5
  实际正常         3       50

【像素级指标】(分割)
  评估图片数: 42
  平均 IoU:  0.7523
  平均 Dice: 0.8234
  平均 Pixel-F1: 0.8012
  平均 Precision: 0.8456
  平均 Recall: 0.7623

  整体像素级指标 (汇总):
  Overall IoU:  0.7845
  Overall Dice: 0.8512

  像素混淆矩阵:
  TP: 1,234,567  TN: 45,678,901
  FP: 234,567    FN: 345,678

【检测框准确率】(定位) ⭐新增
  评估图片数: 42
  平均 Precision: 0.8523 (预测框正确匹配比例)
  平均 Recall:    0.7845 (真实框被匹配比例)
  平均 F1:        0.8167
  平均 IoU:       0.7234 (匹配成功的平均 IoU)

  整体检测框指标 (汇总):
  Overall Precision: 0.8623
  Overall Recall:    0.7956
  Overall F1:        0.8278

  真实检测框: 50 个
  预测检测框: 45 个
  匹配成功: 真实=38 | 预测=39

【性能指标】
  总耗时: 35.25s
  平均耗时: 352.50ms
  最小耗时: 120.30ms
  最大耗时: 890.50ms
  处理速度: 2.84 FPS

【置信度统计】
  篡改图片平均置信度: 0.8542 (47张)
  正常图片平均置信度: 0.1823 (53张)
```

### 像素级指标说明

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| IoU | Intersection over Union | TP / (TP + FP + FN) |
| Dice | Dice 系数 | 2×TP / (2×TP + FP + FN) |
| Pixel-F1 | 像素级 F1 | 2×Precision×Recall / (Precision+Recall) |
| Pixel-Precision | 像素级精确率 | TP / (TP + FP) |
| Pixel-Recall | 像素级召回率 | TP / (TP + FN) |
| Overall | 汇总指标 | 基于所有图片的总 TP/FP/FN 计算 |

### 检测框准确率说明 ⭐新增

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| B-Precision | 检测框精确率 | 匹配成功的预测框数 / 总预测框数 |
| B-Recall | 棡测框召回率 | 匹配成功的真实框数 / 总真实框数 |
| B-F1 | 棡测框 F1 | 2×Precision×Recall / (Precision+Recall) |
| B-IoU | 棡测框平均 IoU | 匹配成功的检测框平均 IoU |
| Overall | 汇总指标 | 基于所有图片的总匹配数计算 |

**检测框匹配策略**:
1. 从真实 mask 提取检测框 (轮廓外接矩形)
2. 从预测结果 `tamper_regions` 提取检测框
3. 对于每个真实框，找预测框中 IoU 最大的
4. IoU > 0.5 记为正确匹配

**结果图片标注**:
- 绿色实线框: 预测检测框
- 蓝色虚线框: 真实检测框 (Ground Truth)
- 红色轮廓: 篡改区域边界

---

## 算法说明

| 算法 | 速度 | 精度 | 适用场景 | 说明 |
|------|------|------|----------|------|
| ELA | 快 | 中 | JPEG 篡改 | JPEG压缩误差分析 |
| DCT | 快 | 中 | 块效应检测 | DCT块效应不一致性 |
| Fusion | 中 | 高 | 综合检测 | ELA+DCT融合 |
| ML | **快 (GPU)** | 最高 | 精准定位 | 像素级机器学习 ⭐推荐 |

### ML 算法优化

**v3.0 GPU 加速版本** 支持 XGBoost GPU 推理，性能提升 **5-10x**:

| 阶段 | 优化前 (CPU) | 优化后 (GPU) | 提升 |
|------|--------------|--------------|------|
| DCT 预计算 | ~200ms (双重循环) | ~50ms (向量化) | **4x** |
| percentile | ~500ms (排序) | ~100ms (partition) | **5x** |
| 窗口级 DCT | ~360ms | 0ms (已移除) | **移除** |
| 模型推理 | ~200ms | ~50ms (GPU) | **4x** |
| **总计** | **~1.5-2s** | **~0.3-0.5s** | **5-10x** |

**关键优化点**:
- `GlobalFeatureCache`: 预计算 ELA/DCT/Noise 等全局特征图
- `FastPixelFeatureExtractor`: 从缓存快速提取窗口特征
- **DCT 向量化**: 移除双重循环，批量处理 8x8 块
- **percentile 优化**: 使用 `np.partition` 替代排序
- **GPU 推理**: XGBoost 自动检测并启用 CUDA

**GPU 配置要求**:
- NVIDIA GPU + CUDA 12.x
- XGBoost 3.0+
- PyTorch (可选，用于检测 GPU)

```bash
# 验证 GPU 推理已启用
python -c "
import xgboost as xgb
model = xgb.XGBClassifier(tree_method='hist', device='cuda:0')
print('GPU 推理: ✓')
"

---

## 训练模型

如需重新训练模型，请参考 [train/README.md](train/README.md)。

### 训练 GB 分类器

```bash
python train/gb_classifier/train_gb.py \
    --data_dir ./data/tamper_data \
    --output_dir ./release/models/gb_classifier
```

### 训练像素级模型

提供两个训练脚本：

#### 1. `train_pixel.py` - 基础版

适用于一般场景，功能完整：

```bash
# GPU 加速训练 (推荐)
python train/pixel_segmentation/train_pixel.py \
    --data_dir /path/to/data \
    --model_type xgb \
    --preset balanced \
    --num_workers 16

# 保存数据集缓存
python train/pixel_segmentation/train_pixel.py \
    --data_dir /path/to/data \
    --cache_dataset ./cache/dataset.npz

# 从缓存加载
python train/pixel_segmentation/train_pixel.py \
    --data_dir /path/to/data \
    --load_cache ./cache/dataset.npz
```

#### 2. `train_pixel_bbox.py` - 检测框优化版 ⭐推荐

针对**检测框准确率**优化，新增：
- **边界加权采样**: 篡改区域边界样本权重更高
- **连通域分析**: 训练时监控检测框数量匹配度
- **后处理参数优化**: 自动搜索最佳阈值

```bash
# 检测框优化训练 (推荐)
python train/pixel_segmentation/train_pixel_bbox.py \
    --data_dir /path/to/data \
    --preset bbox_optimized \
    --num_workers 16

# 高召回配置 (优先召回率)
python train/pixel_segmentation/train_pixel_bbox.py \
    --data_dir /path/to/data \
    --preset high_recall
```

#### 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 数据目录 (必需) | - |
| `--output_dir` | 模型输出目录 | `./release/models/pixel_segmentation` |
| `--preset` | 预设配置 | `balanced` / `bbox_optimized` |
| `--model_type` | 模型类型 | `xgb` |
| `--window_size` | 滑动窗口大小 | 32 |
| `--stride` | 滑动步长 | 16 |
| `--num_workers` | 并行进程数 | 8 |
| `--skip_solid` | 跳过纯色块 | True |
| `--cache_dataset` | 保存缓存路径 | - |
| `--load_cache` | 加载缓存路径 | - |

#### 预设配置对比

| 参数 | default | balanced | bbox_optimized ⭐ | high_recall |
|------|---------|----------|-------------------|-------------|
| 篡改过采样 | 3x | 5x | **6x** | 8x |
| 类别权重 | 10 | 20 | **25** | 30 |
| **边界权重** | 2.0 | 3.0 | **4.0** | 3.0 |
| **最小面积** | 100 | 100 | **150** | 80 |
| 树数量 | 300 | 500 | **600** | 600 |
| 适用场景 | 基础 | 平衡 | **检测框优化** | 高召回 |

#### 模型类型对比

| 模型 | 训练 | 推理 | GPU 支持 | 推荐 |
|------|------|------|----------|------|
| **xgb** | GPU | GPU | ✅ 原生支持 | ⭐⭐⭐⭐⭐ |
| lgb | CPU | CPU | ❌ | ⭐⭐⭐ |
| rf | CPU | CPU | ❌ | ⭐⭐ |

#### 检测框优化原理

```
原方案:
像素分类 → 阈值 → 连通域 → 检测框
         ↑ 问题：边界模糊

优化方案:
像素分类 + 边界加权 → 阈值优化 → 连通域过滤 → 检测框
    ↑ 边界样本权重更高    ↑ 搜索最佳参数  ↑ 去噪
```

**关键参数**:
- `BORDER_WEIGHT`: 边界样本权重，越大边界越清晰
- `MIN_AREA_THRESHOLD`: 最小连通域面积，过滤噪点
- `MORPH_KERNEL_SIZE`: 形态学核大小，平滑边界

#### 训练输出示例

```
处理 train 集: 10000 张图片 (16 进程)
  [████████████████████] 10000/10000 | 速度: 125.3 张/s

数据集统计:
  处理时间: 800.5s
  总样本数: 25,231,892
  篡改像素: 2,523,189 (10.0%)
  边界样本: 504,638 (2.0%)

连通域分析:
  真实: 8423 个
  预测: 8102 个
  平均大小: 1256
  噪点数: 23

训练完成:
  F1: 0.8523
  Precision: 0.8712
  Recall: 0.8342
  阈值: 0.42
```

---

## 接口文档

详细接口文档请参考: [docs/api_document.md](docs/api_document.md)

### 状态码说明

| 状态码 | 描述 | 说明 |
|--------|------|------|
| 0000 | 未知异常 | 服务发生未捕获的全局异常 |
| 0001 | 解析成功/服务运行正常 | 业务处理成功 |
| 0002 | base64解码异常 | image_base64参数解码失败 |
| 0004 | 参数格式错误 | 参数缺失、类型错误或格式不符合要求 |
| 0007 | 请求参数内容错误或为空 | 参数为空或无法解析图片 |

---

## 注意事项

1. **模型文件**: 首次使用需要上传模型文件到 `release/models/` 目录
2. **GPU 支持**: ML 算法支持 GPU 加速，需安装 PyTorch + CUDA
3. **内存消耗**: 像素级检测需要较大内存，建议 8GB+
4. **GB 分类器**: 如果 GB 分类器模型未加载，会跳过前置检测直接进行区域定位

---

## 更新日志

### 2026-03-30 v5 ⭐检测框优化训练

- ✅ **新增检测框优化训练脚本**
  - `train_pixel_bbox.py` 专门针对检测框准确率优化
  - 边界加权采样: 篡改区域边界样本权重更高
  - 连通域分析: 训练时监控检测框数量匹配度
  - 后处理参数优化: 自动搜索最佳阈值
- ✅ **新增预设配置**
  - `bbox_optimized`: 检测框优化 (推荐)
  - `high_recall`: 高召回场景
- ✅ **详细文档更新**
  - 检测框优化原理说明
  - 关键参数解释
  - 使用示例

### 2026-03-30 v4 ⭐训练脚本合并

- ✅ **合并三个训练脚本为一个**
  - 保留 `train_pixel.py`，删除 `train_pixel_fast.py` 和 `train_pixel_optimized.py`
  - 统一使用 49 维特征 (移除无效 LBP)
- ✅ **新增功能**
  - 数据集缓存 (`--cache_dataset` / `--load_cache`)
  - 连通域分析 (与检测框准确率相关)
  - 4 种预设配置 (`default`/`balanced`/`high_recall`/`aggressive`)
  - GPU 加速参数 (`device='cuda:0'`)
- ✅ **检测框准确率相关优化**
  - 训练时分析连通域数量和大小
  - 帮助评估模型对检测框准确率的影响

### 2026-03-30 v3 ⭐GPU 加速

- ✅ **XGBoost GPU 推理**
  - 训练脚本新增 `tree_method='hist'` + `device='cuda:0'`
  - 推理时自动检测并启用 GPU
  - 模型推理加速 5-10x
- ✅ **特征提取优化**
  - DCT 预计算向量化 (移除双重循环，4x 加速)
  - `np.partition` 替代 `np.percentile` (3x 加速)
  - 移除窗口级重复 DCT 计算
- ✅ **性能提升**
  - 单图推理: 1.5-2s → 0.3-0.5s (5-10x)
  - GPU 利用率: 0% → 70-90%

### 2026-03-27 v3

- ✅ **检测框准确率指标**
  - 新增检测框级别的 Precision/Recall/F1/IoU 统计
  - 使用最佳匹配策略：IoU > 0.5 记为正确匹配
  - 支持多检测框匹配和汇总指标计算
- ✅ **结果图片检测框绘制优化**
  - 绿色实线框：预测检测框
  - 蓝色虚线框：真实检测框 (Ground Truth)
  - 红色轮廓：篡改区域边界
  - 修复 ML 算法结果图片无检测框的问题

### 2026-03-27 v2

- ✅ **测试脚本并发支持**
  - 支持 `--workers` 参数指定并发线程数
  - 默认 8 线程并发测试
  - ThreadPoolExecutor 实现，大幅提升测试速度
- ✅ **服务启动/停止脚本**
  - `start.sh`: 一键启动服务
  - `stop.sh`: 一键停止服务
  - 支持前台/后台运行
  - 支持指定端口
  - 自动检查依赖和端口占用

### 2026-03-27 v1

- ✅ **篡改区域标注优化**
  - 所有算法在原图上绘制矩形框标注篡改区域
  - 红色轮廓 + 绿色矩形框，清晰直观
- ✅ **测试脚本优化**
  - 结果图片只保存一张（左侧原图，右侧标注）
  - 批量测试默认保存图片
- ✅ **日志管理自动化**
  - 日志自动保存到 `./logs` 目录
  - 每30天自动清理旧日志
  - 日志轮转：10MB × 5个文件
- ✅ **并发支持**
  - 默认8个 Worker 并发处理
  - 无需参数配置

### 2026-03-26

- ✅ **ML 算法推理优化** (5-10x 加速)
  - 全局特征预计算 `GlobalFeatureCache`
  - 避免每窗口重复 JPEG 编解码
  - 滑动窗口从缓存快速取值
- ✅ 新增极速优化版训练脚本 `train_pixel_fast.py`
- ✅ 实时进度显示 (进度条/速度/预计时间)
- ✅ LBP 向量化加速 (skimage, 50x)
- ✅ 纯色块智能跳过
- ✅ 测试脚本新增像素级指标 (IoU/Dice/Pixel-F1)
- ✅ 特征提取器支持多维度 (57/35)
- ✅ ML 检测器自动适配特征维度

### 2026-03-24

- ✅ GB分类器作为所有算法的前置检测器
- ✅ 新增 test_service.py 测试脚本
- ✅ 支持单张/批量/对比测试
- ✅ 测试报告包含效果指标和性能指标
- ✅ 保存推理结果图片
- ✅ 修复 numpy 类型序列化问题

### 2026-03-23

- ✅ 重构目录结构，train 与 release 平级
- ✅ 增加状态码支持 (0000/0001/0002/0004/0007)
- ✅ 增加 tamper_regions 篡改区域坐标输出
- ✅ 增加 marked_image_base64 标记图片输出

### 2026-03-20

- ✅ 整理项目结构
- ✅ 创建 FastAPI 服务
- ✅ 整合四种检测算法
- ✅ 支持 GPU 推理

---

**项目负责人**: 灰 (上坤商业帝国首席 CTO)  
**直属上级**: CEO 上坤