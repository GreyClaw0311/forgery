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
- ✅ **GPU 加速**: 支持 PyTorch GPU 加速推理
- ✅ **REST API**: FastAPI 服务，易于集成
- ✅ **测试工具**: 完整的单张/批量测试脚本

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
    ├── main.py                 # FastAPI 主程序
    ├── test_service.py         # 测试脚本 ⭐
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

```bash
cd release
python main.py
```

服务启动后访问: http://localhost:8000

### 4. 测试服务

```bash
# 检查服务状态
python test_service.py --mode single --image test.jpg --algorithm ml

# 批量测试
python test_service.py --mode batch --data_dir ./data/tamper_data --algorithm ml --save_images

# 多算法对比
python test_service.py --mode compare --data_dir ./data/tamper_data
```

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
| mask_base64 | string | 篡改区域掩码 Base64 |
| marked_image_base64 | string | 标记后的图片 Base64 |
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
  测试时间: 2026-03-24T10:30:00
  测试图片: 100/100

【效果指标】
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

【性能指标】
  总耗时: 35.25s
  平均耗时: 352.50ms
  最小耗时: 120.30ms
  最大耗时: 890.50ms
  处理速度: 2.84 FPS

【置信度统计】
  篡改图片平均置信度: 0.8542 (47张)
  正常图片平均置信度: 0.1823 (53张)

【分类统计】
  easy        :  20张, Acc=0.9500, Conf=0.8934, Time=280.5ms
  difficult   :  27张, Acc=0.8519, Conf=0.8215, Time=420.3ms
  good        :  53张, Acc=0.9434, Conf=0.1823, Time=310.2ms
```

---

## 算法说明

| 算法 | 速度 | 精度 | 适用场景 | 说明 |
|------|------|------|----------|------|
| ELA | 快 | 中 | JPEG 篡改 | JPEG压缩误差分析 |
| DCT | 快 | 中 | 块效应检测 | DCT块效应不一致性 |
| Fusion | 中 | 高 | 综合检测 | ELA+DCT融合 |
| ML | 较慢 | 最高 | 精准定位 | 像素级机器学习 ⭐推荐 |

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

```bash
python train/pixel_segmentation/train_pixel.py \
    --data_dir /path/to/processed_data \
    --output_dir ./release/models/pixel_segmentation
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