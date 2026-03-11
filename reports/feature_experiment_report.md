# 图像篡改检测特征实验报告

## 第一章：背景介绍

### 1.1 项目目标

本项目旨在使用传统OpenCV方法实现图像篡改检测算法，通过多特征融合提高检测准确率。

### 1.2 数据集

- **easy**: 20张简单篡改图像（cover/cpmv/inpaint/splice各类型）
- **difficult**: 16张复杂篡改图像（cover/cpmv/inpaint/splice各类型）
- **good**: 10张未篡改高质量样本（用于误报率测试）

### 1.3 篡改类型说明

| 类型 | 描述 |
|------|------|
| cover | 同图像复制覆盖 |
| cpmv | 同图像复制粘贴 |
| inpaint | 算法填充修复 |
| splice | 跨图像拼接 |

---

## 第二章：特征检测算法

### 2.1 核心特征（10个）

| 序号 | 特征名称 | 原理 | 适用篡改类型 |
|------|----------|------|--------------|
| 1 | ELA | Error Level Analysis - 检测JPEG压缩不一致 | 重压缩、拼接 |
| 2 | CFA | Color Filter Array插值检测 | 传感器伪影不一致 |
| 3 | DCT | DCT系数分析 | JPEG块边界、量化不一致 |
| 4 | NOISE | 噪声一致性分析 | 噪声注入、拼接 |
| 5 | BLK | JPEG块效应检测 | 重压缩、块对齐不一致 |
| 6 | LBP | 局部二值模式纹理分析 | 复制粘贴、纹理不一致 |
| 7 | HOG | 方向梯度直方图分析 | 边缘方向不一致 |
| 8 | SIFT | SIFT特征点密度分析 | 复制移动检测 |
| 9 | EDGE | 边缘模式分析 | 边缘断裂、拼接 |
| 10 | COLOR | 颜色一致性分析 | 颜色异常、光照不一致 |

### 2.2 高级变体（14个）

- ELA_Advanced: 多质量级别ELA
- CFA_Interpolation: CFA插值模式不一致检测
- DCT_Residual: DCT残差分析
- NOISE_Variance: 局部噪声方差分析
- PRNU: 光电响应非均匀性噪声分析
- BLK_Grid: 块网格对齐不一致检测
- LBP_Consistency: LBP纹理一致性分析
- HOG_Variance: HOG局部方差分析
- GRAD_Inconsistency: 梯度方向不一致分析
- CopyMove: SIFT复制移动检测
- EDGE_Consistency: 边缘方向一致性分析
- EDGE_Density: 局部边缘密度分析
- ILLUMINATION: 光照一致性分析
- CHROMATIC: 色差分析

---

## 第三章：实验过程与结果

### 3.1 实验环境

- Python 3.12
- OpenCV 4.13.0
- NumPy, scikit-learn

### 3.2 评估指标

- **IoU**: Intersection over Union，预测与真值的重叠度
- **Precision**: 精确率，预测为正中真为正的比例
- **Recall**: 召回率，真为正中预测为正的比例
- **F1**: 精确率和召回率的调和平均

### 3.3 Easy数据集结果（20张图片）

| 特征 | IoU | Precision | Recall | F1 |
|------|-----|-----------|--------|------|
| DCT | 0.1152 | 0.1471 | 0.5677 | **0.1864** |
| HOG | 0.0800 | 0.1267 | 0.3187 | 0.1413 |
| CFA | 0.0801 | 0.1153 | 0.5259 | 0.1370 |
| NOISE | 0.0698 | 0.1576 | 0.1715 | 0.1262 |
| COLOR | 0.0818 | 0.0944 | 0.2369 | 0.1187 |
| EDGE | 0.0613 | 0.1057 | 0.1980 | 0.1115 |
| ELA | 0.0599 | 0.0891 | 0.4576 | 0.1024 |
| BLK | 0.0287 | 0.1215 | 0.0490 | 0.0554 |
| LBP | 0.0224 | 0.0232 | 0.4624 | 0.0435 |
| SIFT | 0.0177 | 0.0477 | 0.0599 | 0.0344 |

### 3.4 Difficult数据集结果（16张图片）

| 特征 | IoU | Precision | Recall | F1 |
|------|-----|-----------|--------|------|
| COLOR | 0.0238 | 0.0611 | 0.2207 | **0.0367** |
| HOG | 0.0129 | 0.0153 | 0.1817 | 0.0248 |
| EDGE | 0.0118 | 0.0172 | 0.0961 | 0.0226 |
| ELA | 0.0118 | 0.0149 | 0.2465 | 0.0225 |
| DCT | 0.0112 | 0.0118 | 0.2634 | 0.0218 |
| CFA | 0.0113 | 0.0120 | 0.3105 | 0.0218 |
| LBP | 0.0099 | 0.0100 | 0.6548 | 0.0191 |
| NOISE | 0.0092 | 0.0120 | 0.0886 | 0.0180 |
| BLK | 0.0067 | 0.0143 | 0.0236 | 0.0132 |
| SIFT | 0.0046 | 0.0128 | 0.0108 | 0.0090 |

---

## 第四章：结论与分析

### 4.1 特征效果对比

**Easy数据集表现最佳：**
1. DCT (F1=0.1864) - 对简单篡改检测效果最好
2. HOG (F1=0.1413) - 方向梯度特征有效
3. CFA (F1=0.1370) - 传感器伪影检测有效

**Difficult数据集表现最佳：**
1. COLOR (F1=0.0367) - 颜色一致性对复杂篡改稍有效
2. HOG (F1=0.0248) - 梯度特征仍有一定效果
3. EDGE (F1=0.0226) - 边缘模式检测

### 4.2 问题分析

1. **Precision过低**: 所有特征的精确率都低于20%，导致大量误检
2. **阈值问题**: 当前使用Otsu自动阈值，可能不适合篡改检测
3. **特征融合缺失**: 单特征效果有限，需要多特征融合

### 4.3 改进方向

1. **阈值优化**: 实现自适应阈值或手动调参
2. **特征融合**: 结合DCT+HOG+CFA等多个高Recall特征
3. **后处理**: 形态学操作、连通域过滤
4. **高级检测器**: 尝试ELA_Advanced、PRNU等高级变体

---

## 附录

### A. 运行命令

```bash
# 列出所有检测器
python -m src.main --list-detectors

# 测试数据集
python -m src.main --data tamper_data/easy --output results/easy
python -m src.main --data tamper_data/difficult --output results/difficult

# 测试单张图片
python -m src.main --image test.jpg --output results

# 指定检测器
python -m src.main --data tamper_data/easy --detectors ELA DCT NOISE --output results
```

### B. 结果文件

- `results/easy/results.json` - Easy数据集详细结果
- `results/difficult/results.json` - Difficult数据集详细结果

---

*报告生成时间: 2026-03-12*