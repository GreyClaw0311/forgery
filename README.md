# Forgery Detection System

基于传统图像处理方法和机器学习的图像篡改检测系统。

## 项目简介

本系统通过提取图像特征（JPEG块效应、频域分析、颜色一致性等），结合Gradient Boosting机器学习模型，实现图像篡改检测。主要检测类型包括：

- **拼接篡改** (Splicing): 将其他图像的部分拼接到目标图像
- **复制粘贴** (Copy-Move): 图像内部区域的复制移动
- **修饰篡改** (Retouching): 通过修复工具等进行的修饰

### 核心指标

| 指标 | 值 |
|------|------|
| F1-score | **0.9302** |
| Precision | **97.87%** |
| Recall | 88.62% |
| 误报率(FPR) | 36.84% |

## 目录结构

```
forgery/
├── train/                 # 训练相关代码
│   └── train.py          # 训练入口脚本
├── release/              # 发布相关代码
│   ├── pipeline.py       # 检测Pipeline
│   └── models/           # 模型文件
│       ├── model.pkl     # 训练好的模型
│       ├── scaler.pkl    # 标准化器
│       └── config.json   # 配置文件
├── data/                 # 测试数据
│   ├── easy/             # 简单篡改样本
│   ├── difficult/        # 复杂篡改样本
│   └── good/             # 正常样本
├── src/                  # 源代码
│   ├── features/         # 特征提取模块 (24个特征)
│   └── config.py         # 配置文件
├── reports/              # 实验报告
│   └── final_report.md   # 全流程报告
├── requirements.txt      # Python依赖
├── Dockerfile           # Docker配置
└── README.md            # 项目说明
```

## 快速开始

### 环境要求

- Python 3.9+
- OpenCV 4.8+
- scikit-learn 1.3+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用方法

#### 1. 预测单张图片

```python
from release.pipeline import ForgeryDetector

# 初始化检测器
detector = ForgeryDetector()

# 预测图片
result = detector.predict('test_image.jpg')

print(f"是否篡改: {result['is_tampered']}")
print(f"置信度: {result['confidence']:.4f}")
```

#### 2. 批量预测

```python
# 批量预测多张图片
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.predict_batch(image_paths)

for r in results:
    print(f"{r['image_path']}: {'篡改' if r['is_tampered'] else '正常'}")
```

#### 3. 命令行使用

```bash
python -m release.pipeline image.jpg
```

### Docker 部署

```bash
# 构建镜像
docker build -t forgery-detector:1.0 .

# 运行检测
docker run --rm -v /path/to/images:/images forgery-detector:1.0 \
    python -m release.pipeline /images/test.jpg
```

## 训练模型

### 准备数据

将数据放置在 `data/` 目录下：

```
data/
├── easy/images/      # 简单篡改图片
├── difficult/images/ # 复杂篡改图片
└── good/             # 正常图片
```

### 执行训练

```bash
python train/train.py --data_dir ./data --output_dir ./release/models
```

## 特征说明

系统使用 **Top 10** 关键特征进行检测：

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

## 模型说明

- **模型类型**: `sklearn.ensemble.GradientBoostingClassifier`
- **参数**: max_depth=5, learning_rate=0.1, n_estimators=200
- **最优阈值**: 0.85

## 性能详情

### 测试集分布

| 类别 | 数量 | 说明 |
|------|------|------|
| Easy | ~529 | 简单篡改 |
| Difficult | ~561 | 复杂篡改 |
| Good | 57 | 正常图片 |
| **总计** | **1147** | 测试集20% |

### 误报分析

误报率 **36.84%** 100% 来自 Good 类别（正常图片被误判为篡改）。

## 注意事项

1. **数据不平衡**: 正常样本仅占4.9%，建议收集更多正常样本
2. **阈值调整**: 可根据业务需求调整阈值（阈值↑ → 误报率↓，漏报率↑）
3. **格式支持**: 支持 JPG、PNG、BMP 格式

## License

MIT License

## 作者

灰 - 上坤商业帝国首席CTO

---

**GitHub**: https://github.com/GreyClaw0311/forgery