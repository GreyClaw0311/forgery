# Forgery Detection System

基于传统图像处理方法和机器学习的图像篡改检测系统。

## 项目简介

本系统通过提取图像特征（JPEG块效应、频域分析、颜色一致性等），结合 Gradient Boosting 机器学习模型，实现图像篡改检测。

### 核心指标

| 指标 | 值 |
|------|------|
| F1-score | **0.9302** |
| Precision | **97.87%** |
| 误报率(FPR) | 36.84% |

## 目录结构

```
forgery/
├── train/                     # 训练相关代码
│   ├── __init__.py
│   ├── features.py           # 特征提取模块 (Top 10特征)
│   └── train.py              # 训练脚本
├── release/                   # 发布相关代码
│   ├── __init__.py
│   ├── pipeline.py           # 检测Pipeline
│   └── models/               # 模型文件目录
├── data/                      # 测试数据
│   ├── README.md             # 数据说明
│   └── tamper_data/          # 测试样本
│       ├── easy/             # 简单篡改 (20张)
│       │   ├── images/
│       │   └── masks/
│       ├── difficult/        # 复杂篡改 (17张)
│       │   ├── images/
│       │   └── masks/
│       └── good/             # 正常图片 (10张)
├── final_report.md            # 全流程实验报告
├── requirements.txt           # Python依赖
├── Dockerfile                # Docker配置
└── README.md                 # 项目说明
```

## 快速开始

### 环境要求

- Python 3.9+
- OpenCV 4.8+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

```python
from release.pipeline import ForgeryDetector

# 初始化检测器
detector = ForgeryDetector()

# 预测单张图片
result = detector.predict('test_image.jpg')

print(f"是否篡改: {result['is_tampered']}")
print(f"置信度: {result['confidence']:.4f}")
```

### 命令行使用

```bash
python -m release.pipeline image.jpg
```

## 训练模型

```bash
python train/train.py --data_dir ./data --output_dir ./release/models
```

## 特征说明

系统使用 **Top 10** 关键特征：

| 特征 | 描述 | 重要性 |
|------|------|--------|
| fft | 傅里叶频域分析 | 14.23% |
| resampling | 重采样检测 | 12.09% |
| color | 颜色一致性 | 11.45% |
| jpeg_block | JPEG块效应 | 10.48% |
| splicing | 拼接检测 | 9.94% |
| cfa | CFA插值检测 | 9.53% |
| contrast | 对比度一致性 | 8.98% |
| edge | 边缘一致性 | 8.48% |
| jpeg_ghost | JPEG伪影检测 | 7.62% |
| saturation | 饱和度一致性 | 7.20% |

## Docker 部署

```bash
# 构建镜像
docker build -t forgery-detector:1.0 .

# 运行检测
docker run --rm -v /path/to/images:/images forgery-detector:1.0 \
    python -m release.pipeline /images/test.jpg
```

## 实验报告

详见 [final_report.md](./final_report.md)

## License

MIT License

## 作者

灰 - 上坤商业帝国首席CTO

---

**GitHub**: https://github.com/GreyClaw0311/forgery
