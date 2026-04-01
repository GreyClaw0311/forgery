# 模型文件目录

本目录存放预训练模型文件，需要手动上传。

---

## GB 分类器模型

位置: `gb_classifier/`

需要文件:
- `model.pkl` - GradientBoost 分类器模型
- `scaler.pkl` - 特征标准化器
- `config.json` - 配置文件 (阈值等)

训练方法:
```bash
cd train/gb_classifier
python train_gb.py --data_dir /path/to/data --output_dir ../../release/models/gb_classifier
```

---

## 像素级分割模型

位置: `pixel_segmentation/`

需要文件:
- `model.pkl` - 像素级分割模型 (包含 model, scaler, threshold, config)

训练方法:
```bash
cd train/pixel_segmentation
python train_pixel_bbox.py --data_dir /path/to/data --preset bbox_optimized --output_dir ../../release/models/pixel_segmentation
```

---

## 模型性能参考 (2026-03-31 测试)

### 2K 数据集 (平衡数据)

| 指标 | 数值 | 说明 |
|------|------|------|
| 分类准确率 | **99.75%** | 极高 |
| FPR | **0.5%** | 极低误报率 |
| Overall IoU | **22.09%** | 中等定位精度 |
| 检测框 F1 | 13.02% | 整体偏低 |

### 6K 数据集 (不平衡数据)

| 指标 | 数值 | 说明 |
|------|------|------|
| 分类准确率 | 85.04% | 较高 |
| FPR | 52.28% | 误报率高 (正常样本不足) |
| Overall IoU | 8.85% | 定位精度较低 |
| 检测框 F1 | 4.13% | 整体偏低 |

### 性能分析

| 数据集 | 特点 | 模型表现 |
|--------|------|----------|
| 2K | 平衡 (50%篡改) | 分类优秀，定位待优化 |
| 6K | 不平衡 (83.6%篡改) | 误报率高，需增加正常样本 |

---

## 环境依赖

模型推理需要以下环境:

```bash
# 安装依赖
cd release
pip install -r requirements.txt
```

核心依赖:
- scikit-learn 1.7.2
- xgboost 3.2.0 (GPU)
- torch 2.11.0 (CUDA 13.0)
- opencv-python 4.13.0.92

详见 `release/requirements.txt`

---

**注意**: 模型文件较大，建议通过其他方式传输（如云存储下载）。