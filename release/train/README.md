# 训练模块使用说明

本目录包含两种机器学习模型的训练脚本：

---

## 1. GradientBoost 分类器训练

### 用途
判断图片是否被篡改（二分类）

### 脚本位置
`release/train/gb_classifier/train_gb.py`

### 使用方法

```bash
cd release/train/gb_classifier

# 训练模型
python train_gb.py \
    --data_dir /path/to/tamper_data \
    --output_dir ../../models/gb_classifier
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --data_dir | ./data | 数据目录 (包含 easy/difficult/good) |
| --output_dir | ./models | 模型输出目录 |

### 输出文件

```
models/gb_classifier/
├── model.pkl      # 训练好的模型
├── scaler.pkl     # 特征标准化器
└── config.json    # 配置信息
```

### 特征说明

模型使用 24 个特征，Top 10 最重要特征：

1. FFT (14.23%)
2. Resampling (12.09%)
3. Color (11.45%)
4. JPEG Block (10.87%)
5. Splicing (9.23%)
6. CFA (8.56%)
7. Contrast (7.34%)
8. Edge (6.21%)
9. JPEG Ghost (5.67%)
10. Saturation (4.32%)

---

## 2. 像素级分割模型训练

### 用途
像素级定位篡改区域

### 脚本位置
`release/train/pixel_segmentation/train_pixel.py`

### 使用方法

```bash
cd release/train/pixel_segmentation

# 使用 LightGBM 训练 (推荐)
python train_pixel.py \
    --data_dir /path/to/processed_data \
    --output_dir ../../models/pixel_segmentation \
    --model-type lgb \
    --num-workers 8

# 使用 XGBoost 训练
python train_pixel.py \
    --data_dir /path/to/processed_data \
    --model-type xgb

# 使用集成模型
python train_pixel.py \
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

需要先使用 `process_data.py` 处理数据：

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
models/pixel_segmentation/
├── model.pkl                # 训练好的模型
├── results.json             # 训练结果
└── feature_importance.txt   # 特征重要性报告
```

---

## 3. 数据处理脚本

### 用途
将原始数据转换为训练格式

### 脚本位置
`data/` 目录下的 `process_data.py`

### 使用方法

```bash
python process_data.py \
    --source /path/to/raw/data \
    --output /path/to/processed/data \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

### 输入数据格式

```
raw_data/
├── easy/
│   ├── images/
│   └── masks/
├── difficult/
│   ├── images/
│   └── masks/
└── good/               # 正常图片，无 mask
    └── *.jpg
```

---

## 环境依赖

训练环境需要安装 `requirements.txt` 中的依赖：

```bash
pip install -r requirements.txt
```

主要依赖：
- numpy>=1.24.0
- opencv-python>=4.8.0
- scikit-learn>=1.3.0
- lightgbm>=4.0.0
- xgboost>=2.0.0
- torch>=2.0.0 (GPU 训练)

---

## 性能优化建议

1. **使用 GPU**: 像素级训练支持 GPU 加速
2. **多进程**: 增大 `--num-workers` 加速特征提取
3. **LightGBM**: 比 Random Forest 快 10x
4. **采样策略**: 智能采样解决数据不平衡

---

## 常见问题

### Q: 训练内存不足？
A: 减小 `--num-workers` 或减小每张图片的采样数

### Q: F1 Score 太低？
A: 检查数据质量，尝试不同的 `--model-type`

### Q: 训练太慢？
A: 使用 LightGBM，增加多进程数量