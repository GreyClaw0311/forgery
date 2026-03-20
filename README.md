# 图像篡改像素级分割系统

> 像素级图像篡改检测，精准定位篡改区域

---

## 项目简介

本项目实现了一个基于机器学习的图像篡改检测系统，能够**像素级**定位图像中的篡改区域。系统通过滑动窗口提取多维度特征，使用 LightGBM/XGBoost 模型进行分类，最终生成篡改区域掩码。

### 核心特点

- ✅ **像素级检测**: 精确定位篡改区域边界
- ✅ **多维度特征**: 57维特征综合判断（DCT、ELA、Noise、LBP、频域等）
- ✅ **GPU加速**: 支持 PyTorch GPU 加速特征提取
- ✅ **多进程并行**: 大幅提升训练和推理速度
- ✅ **智能采样**: 解决篡改像素稀疏问题

---

## 快速开始

### 1. 环境安装

```bash
pip install numpy opencv-python scikit-learn tqdm lightgbm xgboost torch
```

### 2. 数据准备

数据目录结构：

```
data/
├── train/
│   ├── images/    # 训练图片 (.jpg)
│   └── masks/     # 对应Mask (.png), 篡改区域为白色(>127)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 3. 训练模型

**优化版训练脚本 (推荐):**

```bash
python scripts/train_fast.py \
    --data-dir ./data \
    --output-dir ./results/model_v2 \
    --model-type lgb \
    --num-workers 8
```

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | 必填 | 数据目录 |
| `--output-dir` | ./results/model_v2 | 模型输出目录 |
| `--model-type` | lgb | 模型类型: lgb/xgb/rf/ensemble |
| `--window-size` | 32 | 滑动窗口大小 |
| `--stride` | 16 | 滑动步长 |
| `--num-workers` | 8 | 多进程数量 |

### 4. 检测/测试

**优化版检测 (GPU加速):**

```bash
python scripts/detect_fast.py \
    --model ./results/model_v2 \
    --data-dir ./data \
    --split test \
    --device cuda:0 \
    --num-workers 4
```

**参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | 模型目录 |
| `--data-dir` | 必填 | 数据目录 |
| `--split` | test | 数据集划分 |
| `--device` | cuda:0 | GPU 设备 (单卡) |
| `--num-workers` | 4 | 多进程数量 |

**单张图片检测:**

```bash
python scripts/detect.py \
    --model ./results/model_v2 \
    --image test.jpg \
    --output ./results/output
```

---

## 方法说明

### 核心流程

```
输入图像
    ↓
滑动窗口 (32×32, stride=16)
    ↓
57维特征提取 (DCT/ELA/Noise/LBP/频域/对比度)
    ↓
LightGBM/XGBoost 分类
    ↓
篡改概率热力图
    ↓
阈值分割 + 后处理
    ↓
篡改区域 Mask
```

### 特征说明

| 特征类别 | 特征数 | 说明 |
|----------|--------|------|
| **DCT** | 8 | JPEG压缩痕迹检测 |
| **ELA** | 4 | 错误级别分析 |
| **Noise** | 6 | 噪声一致性分析 |
| **Edge** | 6 | 边缘特征 (Sobel/Canny) |
| **Texture** | 8 | 纹理统计特征 |
| **Color** | 5 | 颜色/HSV特征 |
| **LBP** | 8 | 局部二值模式 (新增) |
| **Frequency** | 6 | 频域特征 (新增) |
| **Contrast** | 6 | 局部对比度 (新增) |
| **总计** | **57** | |

### 模型对比

| 模型 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| **LightGBM** | 快 | 高 | 默认推荐 |
| **XGBoost** | 中 | 高 | 备选 |
| **Random Forest** | 慢 | 中 | 兼容性场景 |
| **Ensemble** | 慢 | 最高 | 追求精度 |

### 智能采样策略

针对数据不平衡问题（篡改像素仅占 2.7%），采用智能采样：

```
篡改像素: 过采样 3x
正常像素: 欠采样至篡改像素的 2x
目标比例: 篡改:正常 ≈ 1:2
```

---

## 性能指标

### 实验结果

| 实验 | 数据规模 | F1 | Precision | Recall |
|------|----------|------|-----------|--------|
| 小数据集 | 200张 | 0.898 | 0.91 | 0.88 |
| 全量数据 (原版) | 30,644张 | 0.4587 | 0.5117 | 0.4156 |
| 全量数据 (优化版) | 30,644张 | 待测试 | - | - |

### 性能优化

| 指标 | 原版 | 优化版 |
|------|------|--------|
| 训练时间 | 49小时 | ~2-5小时 |
| 推理速度 | 2.96s/张 | ~0.06s/张 |
| 测试集耗时 (6559张) | 5.4小时 | ~7分钟 |

---

## 文件结构

```
forgery/
├── README.md                   # 项目说明
├── docs/
│   ├── experiment_report_20260320.md   # 全量数据实验报告
│   └── optimization_guide.md           # 性能优化指南
├── scripts/
│   ├── train.py               # 原始训练脚本
│   ├── train_fast.py          # 优化版训练脚本 (推荐)
│   ├── detect.py              # 原始检测脚本
│   └── detect_fast.py         # 优化版检测脚本 (推荐)
├── results/
│   └── model/                 # 模型文件
│       ├── model.pkl          # 训练好的模型
│       ├── results.json       # 训练结果
│       └── feature_importance.txt  # 特征重要性
└── data/                      # 数据目录 (不提交)
```

---

## 使用示例

### Python API

```python
from scripts.detect import ForgeryDetector

# 加载模型
detector = ForgeryDetector('./results/model_v2')

# 检测图片
result = detector.detect_from_file('image.jpg')

# 获取结果
mask = result['mask']              # 篡改掩码 (0-255)
heatmap = result['heatmap']        # 置信度热力图 (0-1)
confidence = result['confidence']  # 整体篡改置信度

# 可视化
import cv2
cv2.imwrite('mask.png', mask)
```

### 批量处理

```python
import os
from scripts.detect import ForgeryDetector

detector = ForgeryDetector('./results/model_v2')

image_dir = './images'
for img_name in os.listdir(image_dir):
    if img_name.endswith('.jpg'):
        result = detector.detect_from_file(os.path.join(image_dir, img_name))
        print(f"{img_name}: confidence={result['confidence']:.2%}")
```

---

## 注意事项

### 硬件要求

| 任务 | CPU | 内存 | GPU |
|------|-----|------|-----|
| 训练 | 8核+ | 16GB+ | 可选 |
| 推理 | 4核+ | 8GB+ | 推荐 |

### GPU 使用

- **默认单卡**: `--device cuda:0`
- **多卡需手动指定**: 如需使用多卡，可启动多个进程，分别指定 `cuda:0`、`cuda:1`
- **建议最多2卡**: 不一定所有卡都空闲，根据实际情况选择

### 常见问题

1. **内存不足**: 减小 `--num-workers` 或减小 `MAX_SAMPLES_PER_IMAGE`
2. **训练慢**: 使用 `train_fast.py` + LightGBM
3. **推理慢**: 使用 `detect_fast.py` + GPU
4. **F1低**: 检查数据质量，尝试不同的 `--model-type`

---

## 更新日志

### 2026-03-20

- ✅ 添加 `train_fast.py` 优化版训练脚本
- ✅ 添加 `detect_fast.py` 优化版检测脚本
- ✅ 增强特征集：LBP、频域、局部对比度 (35→57维)
- ✅ 智能采样策略解决数据不平衡
- ✅ 支持 LightGBM/XGBoost 模型
- ✅ 多进程并行处理
- ✅ GPU 加速特征提取

### 2026-03-16

- ✅ 像素级分割模型 (F1=0.898)
- ✅ 35维特征提取
- ✅ Random Forest 基准模型

---

## 相关链接

- **GitHub**: https://github.com/GreyClaw0311/forgery
- **实验报告**: `docs/experiment_report_20260320.md`
- **优化指南**: `docs/optimization_guide.md`

---

**项目负责人**: 灰 (上坤商业帝国首席CTO)  
**直属上级**: CEO 上坤  
**更新时间**: 2026-03-20