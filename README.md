# 图像篡改检测系统

基于传统OpenCV方法实现的图像篡改检测算法，通过机器学习融合多个手工特征实现高精度篡改检测。

## 项目概述

### 目标
检测图像是否经过篡改（如拼接、复制粘贴、修饰等操作）

### 方法
- 提取10个关键图像特征
- 使用Gradient Boosting分类器进行融合判断
- 优化阈值降低误报率

### 效果
| 指标 | 值 |
|------|------|
| CV F1-score | **0.9742** |
| 测试集 F1 | **0.9302** |
| Precision | **97.87%** |
| Recall | 88.62% |
| 误报率(FPR) | 36.84% |
| 漏报率(FNR) | 11.38% |

---

## 目录结构

```
forgery/
├── src/
│   ├── features/              # 特征提取模块
│   │   ├── feature_ela.py     # 错误级别分析
│   │   ├── feature_dct.py     # DCT域分析
│   │   ├── feature_cfa.py     # CFA插值检测
│   │   ├── feature_noise.py   # 噪声一致性
│   │   ├── feature_edge.py    # 边缘一致性
│   │   ├── feature_lbp.py     # 局部二值模式
│   │   ├── feature_histogram.py
│   │   ├── feature_sift.py
│   │   ├── feature_fft.py     # 傅里叶频域
│   │   ├── feature_metadata.py
│   │   ├── feature_hog.py     # 方向梯度直方图
│   │   ├── feature_color.py   # 颜色一致性
│   │   ├── feature_wavelet.py # 小波分析
│   │   └── ... (共24个特征)
│   ├── forgery_pipeline.py    # 完整预测Pipeline
│   ├── pipeline.py            # Pipeline框架
│   └── config.py              # 配置文件
├── scripts/
│   ├── process_full_data.py   # 数据处理脚本
│   ├── step1_build_matrix_optimized.py  # 特征提取
│   ├── step2_train_xgboost.py # 模型训练
│   └── step3_optimize_model.py # 模型优化
├── reports/
│   ├── feature_experiment_report.md  # 特征实验报告
│   ├── feature_fusion_research.md    # 融合方法调研
│   ├── variant_experiment_report.md  # 变体特征报告
│   └── gb_model_training_report.md   # 模型训练报告
├── results/
│   ├── full/
│   │   ├── feature_matrix.csv   # 特征矩阵
│   │   ├── optimized_model.pkl  # 训练好的模型
│   │   ├── optimized_scaler.pkl # 标准化器
│   │   └── pipeline_info.json   # Pipeline信息
│   └── *.json                   # 各阶段结果
├── tamper_data_full/            # 全量数据集
│   └── processed/
│       ├── easy/images/         # 简单篡改图片
│       ├── difficult/images/    # 复杂篡改图片
│       └── good/                # 正常图片
└── README.md
```

---

## 快速开始

### 环境要求

- Python 3.12
- OpenCV 4.13.0
- scikit-learn
- numpy
- pandas

### 安装依赖

```bash
pip install opencv-python scikit-learn numpy pandas
```

### 使用方法

#### 1. 预测单张图片

```python
from forgery_pipeline import ForgeryDetectionPipeline

# 初始化Pipeline
pipeline = ForgeryDetectionPipeline()

# 预测图片
result = pipeline.predict('path/to/image.jpg')

print(f"是否篡改: {result['is_tampered']}")
print(f"置信度: {result['confidence']:.4f}")
print(f"篡改概率: {result['probability']:.4f}")
```

#### 2. 批量预测

```python
# 批量预测多张图片
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = pipeline.predict_batch(image_paths)

for r in results:
    print(f"{r['image_path']}: {'篡改' if r['is_tampered'] else '正常'}")
```

---

## 特征说明

项目使用Top 10关键特征：

| 特征 | 说明 | 重要性 |
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

---

## 模型性能

### 训练数据

| 类别 | 数量 |
|------|------|
| Easy (简单篡改) | 2,648 |
| Difficult (复杂篡改) | 2,800 |
| Good (正常) | 283 |
| **总计** | **5,731** |

### 混淆矩阵

```
              预测正常  预测篡改
  实际正常        26        31
  实际篡改        39      1051
```

---

## 项目运行

### 1. 数据处理

```bash
python scripts/process_full_data.py
```

### 2. 特征提取

```bash
python scripts/step1_build_matrix_optimized.py
```

### 3. 模型训练

```bash
python scripts/step2_train_xgboost.py
```

### 4. 模型优化

```bash
python scripts/step3_optimize_model.py
```

---

## 技术亮点

1. **特征工程**: 实现了24个图像篡改检测特征
2. **特征选择**: 基于重要性选择Top 10特征
3. **数据不平衡处理**: 类别权重调整
4. **阈值优化**: 最优阈值0.85
5. **Pipeline封装**: 一键预测接口

---

## 注意事项

- 当前数据集中正常样本仅占4.9%，导致误报率较高
- 阈值可根据实际需求调整（降低阈值→降低误报率，但会增加漏报率）
- 建议收集更多正常样本以改善模型性能

---

## 作者

灰 - 上坤商业帝国首席CTO

## License

MIT