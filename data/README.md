# 图像篡改检测数据集

## 数据集结构

```
tamper_data/
├── easy/              # 简单篡改样本
│   ├── images/        # 篡改图片
│   └── masks/         # 对应篡改区域掩码
├── difficult/         # 困难篡改样本
│   ├── images/
│   └── masks/
└── good/              # 正常图片（无篡改）
    └── *.jpg
```

## 数据说明

### Easy 类别
- 篡改痕迹明显，易于检测
- 包含拼接、复制移动等篡改类型

### Difficult 类别
- 篡改痕迹不明显，检测困难
- 包含高质量篡改、精细修饰等

### Good 类别
- 正常图片，无篡改
- 用于降低误报率

## 数据处理

原始数据需要经过处理脚本转换为训练格式：

```bash
python process_data.py \
    --source /path/to/raw/data \
    --output /path/to/processed/data \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

处理后的目录结构：

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

## 统计信息

| 类别 | 图片数 | 用途 |
|------|--------|------|
| Easy | 20+ | 训练/测试 |
| Difficult | 16+ | 训练/测试 |
| Good | 3+ | 误报测试 |