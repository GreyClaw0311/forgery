# 像素级分割模型调参指南

## 当前问题诊断

### 训练结果分析

```
篡改像素: 1,204,978 (7.9%)
正常像素: 14,015,842 (92.1%)

训练集 F1: 0.5179
验证集 F1: 0.5042
最佳阈值: 0.72
Precision: 0.5967
Recall: 0.5637
F1: 0.5797
```

### 核心问题

| 问题 | 表现 | 原因 |
|------|------|------|
| **数据严重不平衡** | 篡改仅 7.9% | 模型偏向预测正常 |
| **采样不足** | MAX=3000 | 丢失大量篡改像素 |
| **类别权重不够** | scale_pos_weight=10 | 对少数类惩罚不足 |

---

## 调参策略

### 策略 1: 预设配置 (推荐)

```bash
# 平衡配置 (推荐首选)
python train_tuned.py --data_dir /path/to/data --preset balanced

# 激进配置 (数据不平衡严重时)
python train_tuned.py --data_dir /path/to/data --preset aggressive
```

### 预设对比

| 参数 | default | balanced | aggressive |
|------|---------|----------|------------|
| MAX_SAMPLES | 3000 | **5000** | **8000** |
| 篡改过采样 | 3x | **5x** | **10x** |
| 正常欠采样 | 2x | **1.5x** | **1x** |
| 类别权重 | 10 | **20** | **50** |
| 树数量 | 300 | **500** | **800** |
| 学习率 | 0.05 | **0.03** | **0.02** |
| 预期F1 | ~0.58 | **~0.70** | **~0.75** |

---

## 手动调参

### 1. 采样策略调整

```python
# 针对数据不平衡
MAX_SAMPLES_PER_IMAGE = 5000      # 增加总采样量
TAMPER_OVERSAMPLE_RATIO = 5       # 篡改样本过采样 5 倍
NORMAL_UNDERSAMPLE_RATIO = 1.5    # 正常样本欠采样到 1.5 倍篡改量
```

**目标**: 让篡改像素占比提升到 20-30%

### 2. 类别权重调整

```python
# LightGBM
'scale_pos_weight': 20,  # 增大到 20
'is_unbalance': True,    # 自动平衡

# XGBoost
'scale_pos_weight': 20,

# Random Forest
class_weight='balanced'  # 自动平衡
```

**公式**: `scale_pos_weight = 负样本数 / 正样本数 ≈ 92.1 / 7.9 ≈ 12`  
**建议**: 设置为 15-30，给少数类更大权重

### 3. 模型复杂度调整

```python
# 更复杂的模型
N_ESTIMATORS = 500          # 更多树
NUM_LEAVES = 127            # 更复杂叶子
MAX_DEPTH = 12              # 限制深度防止过拟合
LEARNING_RATE = 0.03        # 更小学习率
```

### 4. 特征工程

当前 57 维特征，可以尝试：

```python
# 移除低重要性特征
# 根据之前实验，最有效的特征:
# Noise, Edge, DCT, ELA, 纹理

# 或者尝试更少特征
FEATURE_DIM = 35  # 精简版
```

---

## 训练建议流程

### 第一步: 使用 balanced 预设

```bash
python train_tuned.py \
    --data_dir /ai_paas/limk/workspace/image_tamper/pixel_classify_processed \
    --preset balanced \
    --num_workers 16
```

**预期效果**: F1 从 0.58 提升到 0.65-0.70

### 第二步: 如果效果仍不理想，使用 aggressive

```bash
python train_tuned.py \
    --data_dir /ai_paas/limk/workspace/image_tamper/pixel_classify_processed \
    --preset aggressive \
    --num_workers 16
```

**预期效果**: F1 提升到 0.70-0.75

### 第三步: 根据结果微调

如果 Recall 低 (漏报多):
- 增加 `TAMPER_OVERSAMPLE_RATIO`
- 增加 `scale_pos_weight`
- 降低阈值 (0.5 → 0.4)

如果 Precision 低 (误报多):
- 增加 `NORMAL_UNDERSAMPLE_RATIO`
- 增加模型复杂度
- 提高阈值 (0.5 → 0.6)

---

## 高级优化

### 1. 数据增强

```python
# 对篡改区域做增强
- 翻转 (水平/垂直)
- 旋转 (90°/180°/270°)
- 颜色抖动
```

### 2. 分层采样

```python
# 按篡改面积分层采样
# 大面积篡改 → 全采样
# 小面积篡改 → 过采样
# 正常区域 → 欠采样
```

### 3. 深度学习方案

如果传统 ML 效果不佳，考虑：

```python
# U-Net 分割
# 优势: 端到端，自动学习特征
# 需要: 更多数据，GPU 训练
```

---

## 效果预期

| 配置 | 预期 F1 | 训练时间 | 内存 |
|------|---------|----------|------|
| default | 0.58 | 30min | 8GB |
| balanced | 0.65-0.70 | 45min | 12GB |
| aggressive | 0.70-0.75 | 60min | 16GB |
| 深度学习 | 0.80+ | 数小时 | GPU |

---

## 快速验证

```bash
# 1. 先用小数据集验证配置
python train_tuned.py --preset balanced --num_workers 8

# 2. 确认效果提升后，全量训练
python train_tuned.py --preset aggressive --num_workers 16
```

**关键指标**:
- 篡改像素占比应达到 15-30%
- 训练集 F1 应 > 0.7
- 验证集 F1 应 > 0.65