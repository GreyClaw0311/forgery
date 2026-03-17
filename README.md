# 图像篡改像素级分割系统

像素级图像篡改检测，精准定位篡改区域。

## 快速开始

### 1. 环境安装

```bash
pip install numpy opencv-python scikit-learn tqdm
```

### 2. 数据准备

数据目录结构：
```
data/
├── train/
│   ├── images/    # 训练图片 (.jpg)
│   └── masks/     # 对应Mask (.png)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 3. 训练模型

```bash
python scripts/train.py --data-dir ./data --output-dir ./results/model
```

参数说明：
- `--data-dir`: 数据目录
- `--output-dir`: 模型输出目录
- `--window-size`: 滑动窗口大小 (默认32)
- `--stride`: 滑动步长 (默认16)
- `--n-estimators`: 随机森林树数量 (默认200)

### 4. 检测/测试

**单张图片检测：**
```bash
python scripts/detect.py --model ./results/model --image test.jpg --output ./results/output
```

**批量测试评估：**
```bash
python scripts/detect.py --model ./results/model --data-dir ./data --split test
```

## 方法说明

### 核心思路

```
输入图像 → 滑动窗口(32×32) → 35维特征提取 → Random Forest → 像素级预测 → 篡改Mask
```

### 特征说明

| 特征类别 | 特征数 | 说明 |
|----------|--------|------|
| DCT | 8 | JPEG压缩痕迹检测 |
| ELA | 4 | 错误级别分析 |
| Noise | 4 | 噪声一致性分析 |
| Edge | 6 | 边缘特征 |
| Texture | 8 | 纹理统计特征 |
| Color | 5 | 颜色/HSV特征 |
| **总计** | **35** | |

### 模型配置

- **算法**: Random Forest
- **树数量**: 200
- **最大深度**: 30
- **类别权重**: balanced

## 性能指标

| 指标 | 数值 |
|------|------|
| Precision | ~91% |
| Recall | ~88% |
| F1 | ~0.90 |

## 文件结构

```
forgery/
├── scripts/
│   ├── process_data.py   # 数据处理脚本
│   ├── train.py          # 训练脚本
│   └── detect.py         # 检测/测试脚本
├── results/
│   └── model/            # 模型文件
│       ├── model.pkl     # 训练好的模型
│       └── results.json  # 训练结果
└── README.md
```

## 使用示例

```python
from scripts.detect import ForgeryDetector

# 加载模型
detector = ForgeryDetector('./results/model')

# 检测图片
result = detector.detect_from_file('image.jpg')

# 结果
mask = result['mask']          # 篡改掩码 (0-255)
heatmap = result['heatmap']    # 置信度热力图 (0-1)
confidence = result['confidence']  # 整体篡改置信度
```

## 数据处理

如果需要从原始数据整理：

```bash
python scripts/process_data.py --source /path/to/raw/data --output /path/to/processed/data
```

参数说明：
- `--source`: 原始数据目录
- `--output`: 输出目录
- `--train-ratio`: 训练集比例 (默认0.7)
- `--val-ratio`: 验证集比例 (默认0.15)
- `--test-ratio`: 测试集比例 (默认0.15)

## 注意事项

1. **数据质量**: 确保Mask与图片对应，且Mask中篡改区域为白色(>127)
2. **内存消耗**: 训练时特征提取会占用较多内存，建议分批处理大数据集
3. **检测速度**: 滑动窗口方式较慢，可通过增大stride加速

---

**项目负责人**: 灰 (上坤商业帝国首席CTO)  
**更新时间**: 2026-03-16