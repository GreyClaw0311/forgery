# 训练模块使用说明

本目录包含两种机器学习模型的训练脚本，与 `release/` 目录平级，用于模型训练，不参与在线推理服务。

---

## 目录结构

```
train/
├── gb_classifier/              # GradientBoost 分类器训练
│   ├── __init__.py
│   └── train_gb.py            # 训练脚本
├── pixel_segmentation/         # 像素级分割模型训练
│   ├── __init__.py
│   ├── train_pixel.py         # 训练脚本
│   └── detect_pixel.py        # 检测脚本
├── README.md                   # 本文档
└── requirements.txt            # 训练依赖
```

---

## 1. GradientBoost 分类器训练

### 用途
判断图片是否被篡改（二分类）

### 脚本位置
`train/gb_classifier/train_gb.py`

### 使用方法

```bash
cd /path/to/forgery

# 训练模型
python train/gb_classifier/train_gb.py \
    --data_dir ./data/tamper_data \
    --output_dir ./release/models/gb_classifier
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --data_dir | ./data | 数据目录 (包含 easy/difficult/good) |
| --output_dir | ./release/models | 模型输出目录 |

### 输出文件

```
release/models/gb_classifier/
├── model.pkl      # 训练好的模型
├── scaler.pkl     # 特征标准化器
└── config.json    # 配置信息
```

### 特征说明

模型使用 10 个特征，基于 `release/algorithms/features.py`：

1. FFT - 频域分析
2. Resampling - 重采样检测
3. Color - 颜色一致性
4. JPEG Block - JPEG块效应
5. Splicing - 拼接检测
6. CFA - CFA插值检测
7. Contrast - 对比度一致性
8. Edge - 边缘一致性
9. JPEG Ghost - JPEG伪影
10. Saturation - 饱和度一致性

---

## 2. 像素级分割模型训练

### 用途
像素级定位篡改区域

### 脚本位置
`train/pixel_segmentation/train_pixel.py`

### 使用方法

```bash
cd /path/to/forgery

# 使用 LightGBM 训练 (推荐)
python train/pixel_segmentation/train_pixel.py \
    --data_dir /path/to/processed_data \
    --output_dir ./release/models/pixel_segmentation \
    --model-type lgb \
    --num-workers 8

# 使用 XGBoost 训练
python train/pixel_segmentation/train_pixel.py \
    --data_dir /path/to/processed_data \
    --model-type xgb

# 使用集成模型
python train/pixel_segmentation/train_pixel.py \
    --data_dir /path/to/processed_data \
    --model-type ensemble
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --data_dir | 必填 | 处理后的数据目录 |
| --output_dir | ./results | 模型输出目录 |
| --model-type | lgb | 模型类型: lgb/xgb/rf/ensemble |
| --window-size | 32 | 滑动窗口大小 |
| --stride | 16 | 滑动步长 |
| --num-workers | 8 | 多进程数量 |

### 数据格式

需要先使用数据处理脚本：

```
processed_data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 输出文件

```
release/models/pixel_segmentation/
├── model.pkl                # 训练好的模型
├── results.json             # 训练结果
└── feature_importance.txt   # 特征重要性报告
```

---

## 3. 环境依赖

训练环境需要安装以下依赖：

```bash
pip install -r train/requirements.txt
```

主要依赖：
- numpy>=1.24.0
- opencv-python>=4.8.0
- scikit-learn>=1.3.0
- lightgbm>=4.0.0
- xgboost>=2.0.0
- torch>=2.0.0 (GPU 训练)

---

## 4. 性能优化建议

1. **使用 GPU**: 像素级训练支持 GPU 加速
2. **多进程**: 增大 `--num-workers` 加速特征提取
3. **LightGBM**: 比 Random Forest 快 10x
4. **采样策略**: 智能采样解决数据不平衡

---

## 5. 常见问题

### Q: 训练内存不足？
A: 减小 `--num-workers` 或减小每张图片的采样数

### Q: F1 Score 太低？
A: 检查数据质量，尝试不同的 `--model-type`

### Q: 训练太慢？
A: 使用 LightGBM，增加多进程数量

---

## 6. 与 release 目录的关系

- **train/** - 模型训练代码，离线使用
- **release/** - 推理服务代码，在线部署
- **release/models/** - 存放训练好的模型文件

训练完成后，将模型文件保存到 `release/models/` 目录，即可被推理服务加载使用。