# 图像篡改检测系统

> 基于 OpenCV 特征提取和机器学习的图像篡改检测服务

---

## 项目简介

本系统提供图像篡改检测能力，支持**单特征检测**、**融合检测**和**机器学习检测**三种模式，能够精准定位图像中的篡改区域。

### 核心特点

- ✅ **多算法支持**: ELA、DCT、Fusion、ML 四种检测算法
- ✅ **像素级检测**: ML 算法支持像素级篡改区域定位
- ✅ **篡改区域坐标**: 输出篡改区域矩形坐标
- ✅ **GPU 加速**: 支持 PyTorch GPU 加速推理
- ✅ **REST API**: FastAPI 服务，易于集成

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
├── gb_classifier/           # GradientBoost 分类器
│   ├── model.pkl
│   └── scaler.pkl
└── pixel_segmentation/      # 像素级分割模型
    └── model.pkl
```

### 3. 启动服务

```bash
cd release
python main.py
```

服务启动后访问: http://localhost:8000

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
  "timestamp": "2026-03-20T10:00:00",
  "algorithms": ["ela", "dct", "fusion", "ml"]
}
```

### 检测图片

```bash
curl -X POST "http://localhost:8000/tamper_detection/v1/tamper_detect_img" \
  -d "image_base64=$(base64 -w 0 test.jpg)" \
  -d "algorithm=ml"
```

**响应:**

```json
{
  "status": "0001:解析成功.",
  "is_tampered": true,
  "confidence": 0.92,
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

---

## 算法说明

### 1. ELA (Error Level Analysis)

**原理**: 分析 JPEG 压缩误差，篡改区域通常有不同的压缩痕迹。

**特点**:
- 单特征检测，速度快
- 适合检测 JPEG 篡改

### 2. DCT (Discrete Cosine Transform)

**原理**: 分析 DCT 系数分布，篡改区域 DCT 特征异常。

**特点**:
- 单特征检测，速度快
- 适合检测块效应

### 3. Fusion (多特征融合)

**原理**: 融合 ELA、DCT 等多特征，通过自适应权重融合。

**特点**:
- 多特征综合判断
- 精度高于单特征

### 4. ML (机器学习串联) ⭐ 推荐

**原理**: 
```
图片 → GradientBoost 判断是否篡改 → 像素级 ML 定位篡改区域
```

**特点**:
- 两阶段检测，精度最高
- 支持 GPU 加速
- 像素级定位
- 输出篡改区域坐标

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

### 出参说明

| 参数 | 说明 |
|------|------|
| is_tampered | 是否检测到篡改 |
| confidence | 置信度 (0-1) |
| mask_base64 | 篡改区域掩码图片 (PNG) |
| marked_image_base64 | 标记篡改区域的原图片 (JPEG) |
| tamper_regions | 篡改区域矩形坐标列表 |

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
│   ├── final_report.md         # 最终报告
│   └── experiment_report_*.md  # 实验报告
│
└── release/                     # 服务代码 (部署用)
    ├── main.py                 # FastAPI 主程序
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
    │   └── pixel_segmentation/ # 像素级模型
    │
    └── utils/                  # 工具函数
        └── postprocess.py      # 后处理
```

---

## 性能指标

| 算法 | 速度 | 精度 | 适用场景 |
|------|------|------|----------|
| ELA | 快 | 中 | JPEG 篡改 |
| DCT | 快 | 中 | 块效应检测 |
| Fusion | 中 | 高 | 综合检测 |
| ML | 较慢 | 最高 | 精准定位 |

---

## 注意事项

1. **模型文件**: 首次使用需要上传模型文件到 `release/models/` 目录
2. **GPU 支持**: ML 算法支持 GPU 加速，需安装 PyTorch + CUDA
3. **内存消耗**: 像素级检测需要较大内存，建议 8GB+

---

## 更新日志

### 2026-03-23

- ✅ 重构目录结构，移除训练代码
- ✅ 增加状态码支持 (0000/0001/0002/0004/0007)
- ✅ 增加 tamper_regions 篡改区域坐标输出
- ✅ 增加 marked_image_base64 标记图片输出
- ✅ 接口路由改为 /tamper_detection/v1/tamper_detect_img

### 2026-03-20

- ✅ 整理项目结构
- ✅ 创建 FastAPI 服务
- ✅ 整合四种检测算法
- ✅ 支持 GPU 推理

---

**项目负责人**: 灰 (上坤商业帝国首席 CTO)  
**直属上级**: CEO 上坤