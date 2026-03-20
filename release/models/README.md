# 模型文件目录

本目录存放预训练模型文件，需要手动上传。

---

## GB 分类器模型

位置: `gb_classifier/`

需要文件:
- `model.pkl` - GradientBoost 分类器模型
- `scaler.pkl` - 特征标准化器

训练方法:
```bash
cd release/train/gb_classifier
python train_gb.py --data_dir /path/to/data --output_dir ../../models/gb_classifier
```

---

## 像素级分割模型

位置: `pixel_segmentation/`

需要文件:
- `model.pkl` - 像素级分割模型 (包含 model, scaler, threshold)

训练方法:
```bash
cd release/train/pixel_segmentation
python train_pixel.py --data_dir /path/to/data --output_dir ../../models/pixel_segmentation
```

---

## 模型性能参考

| 模型 | 数据集 | F1 Score | 说明 |
|------|--------|----------|------|
| GB 分类器 | 小批量 | 0.97 | CV F1 |
| 像素级 ML | 小批量 | 0.90 | 像素级 F1 |
| 像素级 ML | 全量数据 | 0.46 | 需优化 |

---

**注意**: 模型文件较大，建议通过其他方式传输（如云存储下载）。