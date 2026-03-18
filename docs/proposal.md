# 图像篡改区域检测方案

**项目名称**: Forgery Region Detection  
**分支**: greyclaw_0313  
**创建时间**: 2026-03-13  
**作者**: 灰 (OpenClaw Assistant)

---

## 一、问题背景与描述

### 1.1 任务定义

| 任务类型 | 描述 | 输出 |
|---------|------|------|
| **图像篡改分类** (之前) | 判断整张图片是否被篡改 | 二分类标签 (篡改/正常) |
| **图像篡改区域检测** (当前) | 检测图片中哪些区域被篡改 | 篡改区域的二值掩码 (Mask) |

### 1.2 问题分析

**篡改区域检测的核心挑战**：

```
原始图像:     ┌─────────────────┐
              │  正常  │ 篡改  │
              │   ✓    │   ✗   │
              │        │       │
              │  正常  │ 正常  │
              │   ✓    │   ✓   │
              └─────────────────┘

目标输出:     ┌─────────────────┐
              │   0   │   1   │
              │   0   │   0   │
              └─────────────────┘
              (0=正常, 1=篡改)
```

### 1.3 篡改类型

| 篡改类型 | 描述 | 检测难点 |
|---------|------|----------|
| **拼接篡改** (Splicing) | 从其他图片复制内容粘贴 | 来源不同，特征不一致 |
| **复制移动** (Copy-Move) | 图片内部区域复制粘贴 | 来源相同，需检测重复区域 |
| **图像修复** (Inpainting) | 使用修复工具填充区域 | 边界平滑，难以检测 |
| **重采样** (Resampling) | 缩放、旋转等操作 | 插值痕迹 |

### 1.4 约束条件

- ❌ **禁用深度学习方法** (CNN、U-Net等)
- ❌ **禁用机器学习方法** (分类器训练)
- ✅ **仅使用传统图像处理方法**

---

## 二、解决方案

### 2.1 整体思路

```
┌─────────────────────────────────────────────────────────────────┐
│                    篡改区域检测流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   输入图像                                                      │
│      │                                                          │
│      ▼                                                          │
│   ┌─────────────────┐                                          │
│   │   分块处理      │  将图像分成多个重叠块                      │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │   多特征分析    │  对每块计算多个特征                        │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │   异常检测      │  识别异常块 (与周围不一致)                 │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │   掩码生成      │  生成篡改区域掩码                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   输出: 篡改区域掩码                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心方法

#### 方法一：分块异常检测

**原理**: 将图像分块，检测与周围区域特征不一致的异常块

```python
def block_anomaly_detection(image, block_size=32, threshold=2.0):
    """
    分块异常检测
    
    步骤:
    1. 将图像分成 32x32 的块
    2. 计算每块的特征 (如ELA、噪声、颜色一致性)
    3. 计算每块与周围块的差异
    4. 差异超过阈值的块标记为篡改
    """
    blocks = split_into_blocks(image, block_size)
    
    for block in blocks:
        # 计算特征
        features = extract_features(block)
        
        # 与周围块比较
        neighbors = get_neighbors(block)
        neighbor_features = [extract_features(n) for n in neighbors]
        
        # 计算异常分数
        anomaly_score = compute_anomaly(features, neighbor_features)
        
        # 标记
        if anomaly_score > threshold:
            block.is_tampered = True
    
    return generate_mask(blocks)
```

#### 方法二：ELA热力图

**原理**: 错误级别分析生成差异热力图，高差异区域为篡改区域

```python
def ela_heatmap(image, quality=90):
    """
    ELA热力图生成
    
    步骤:
    1. 重新压缩图像
    2. 计算像素级差异
    3. 生成差异热力图
    4. 阈值分割得到篡改区域
    """
    # 重新压缩
    reconstructed = jpeg_compress(image, quality)
    
    # 像素级差异
    diff = abs(image - reconstructed)
    
    # 热力图
    heatmap = gaussian_blur(diff)
    
    # 阈值分割
    mask = heatmap > threshold
    
    return mask
```

#### 方法三：噪声一致性分析

**原理**: 篡改区域的噪声模式与原始区域不同

```python
def noise_consistency_detection(image):
    """
    噪声一致性检测
    
    步骤:
    1. 估计每个区域的噪声水平
    2. 检测噪声不一致的区域
    3. 生成篡改掩码
    """
    # 估计噪声
    noise_map = estimate_noise(image)
    
    # 计算局部噪声一致性
    consistency = compute_consistency(noise_map)
    
    # 低一致性区域为篡改
    mask = consistency < threshold
    
    return mask
```

#### 方法四：DCT块效应分析

**原理**: JPEG压缩产生8x8块效应，篡改区域块效应不一致

```python
def dct_block_artifact_analysis(image):
    """
    DCT块效应分析
    
    步骤:
    1. 检测8x8块边界
    2. 计算块效应强度
    3. 异常块效应区域为篡改
    """
    # DCT变换
    dct_blocks = compute_dct_blocks(image)
    
    # 块效应强度
    artifact_map = compute_block_artifacts(dct_blocks)
    
    # 异常检测
    mask = detect_artifact_anomalies(artifact_map)
    
    return mask
```

#### 方法五：复制移动检测

**原理**: 检测图像内部重复区域

```python
def copy_move_detection(image, min_area=100):
    """
    复制移动检测
    
    步骤:
    1. 提取特征点 (SIFT/SURF)
    2. 特征匹配
    3. 聚类匹配对
    4. 生成篡改掩码
    """
    # 特征提取
    keypoints, descriptors = extract_features(image)
    
    # 特征匹配
    matches = match_features(descriptors)
    
    # 过滤自匹配
    matches = filter_self_matches(matches)
    
    # 聚类
    clusters = cluster_matches(matches)
    
    # 生成掩码
    mask = generate_mask_from_clusters(clusters, image.shape)
    
    return mask
```

### 2.3 融合策略

**多方法融合**：

```python
def fusion_detection(image):
    """
    多方法融合检测
    
    策略: 多个方法投票决定
    """
    masks = []
    
    # 方法1: ELA
    masks.append(ela_heatmap(image))
    
    # 方法2: 噪声一致性
    masks.append(noise_consistency_detection(image))
    
    # 方法3: DCT块效应
    masks.append(dct_block_artifact_analysis(image))
    
    # 方法4: 分块异常
    masks.append(block_anomaly_detection(image))
    
    # 融合 (至少2个方法认为篡改)
    final_mask = sum(masks) >= 2
    
    return final_mask
```

### 2.4 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **IoU** | TP / (TP + FP + FN) | 预测掩码与真实掩码的重叠度 |
| **Precision** | TP / (TP + FP) | 预测篡改区域中真正篡改的比例 |
| **Recall** | TP / (TP + FN) | 真实篡改区域被正确检测的比例 |
| **F1-score** | 2PR/(P+R) | Precision和Recall的调和平均 |

---

## 三、步骤排期

### 阶段一：基础框架搭建 (2天)

| 任务 | 内容 | 交付物 |
|------|------|--------|
| 1.1 | 创建项目结构 | 目录结构 |
| 1.2 | 实现分块处理模块 | `block_utils.py` |
| 1.3 | 实现基础可视化 | `visualization.py` |
| 1.4 | 数据集准备 | 测试图片+真实掩码 |

**目录结构**:

```
forgery_region_detection/
├── src/
│   ├── detection/
│   │   ├── ela_detector.py      # ELA检测
│   │   ├── noise_detector.py    # 噪声检测
│   │   ├── dct_detector.py      # DCT检测
│   │   ├── copy_move_detector.py # 复制移动检测
│   │   └── fusion.py            # 融合策略
│   ├── utils/
│   │   ├── block_utils.py       # 分块工具
│   │   ├── visualization.py     # 可视化
│   │   └── metrics.py           # 评估指标
│   └── pipeline.py              # 主流程
├── tests/
│   └── test_detectors.py
├── data/
│   ├── images/                  # 测试图片
│   └── masks/                   # 真实掩码
├── results/                     # 输出结果
├── requirements.txt
└── README.md
```

### 阶段二：单方法实现 (4天)

| 任务 | 方法 | 工期 |
|------|------|------|
| 2.1 | ELA热力图检测 | 1天 |
| 2.2 | 噪声一致性检测 | 1天 |
| 2.3 | DCT块效应检测 | 1天 |
| 2.4 | 分块异常检测 | 1天 |

**每个方法实现**:

```python
class Detector:
    def detect(self, image_path: str) -> np.ndarray:
        """返回二值掩码 (0=正常, 1=篡改)"""
        pass
    
    def get_heatmap(self, image_path: str) -> np.ndarray:
        """返回连续值热力图"""
        pass
```

### 阶段三：融合与优化 (2天)

| 任务 | 内容 | 工期 |
|------|------|------|
| 3.1 | 多方法融合策略 | 1天 |
| 3.2 | 参数调优 | 0.5天 |
| 3.3 | 后处理优化 (形态学操作) | 0.5天 |

**后处理优化**:

```python
def post_process(mask):
    """后处理优化"""
    # 形态学开运算 (去噪点)
    mask = morphological_open(mask, kernel_size=3)
    
    # 形态学闭运算 (填孔)
    mask = morphological_close(mask, kernel_size=5)
    
    # 连通域过滤 (去除小区域)
    mask = remove_small_regions(mask, min_area=100)
    
    return mask
```

### 阶段四：评估与文档 (2天)

| 任务 | 内容 | 工期 |
|------|------|------|
| 4.1 | 在测试集上评估 | 1天 |
| 4.2 | 生成实验报告 | 0.5天 |
| 4.3 | 完善README和文档 | 0.5天 |

### 总工期: 10天

```
Week 1: 阶段一 + 阶段二 (6天)
Week 2: 阶段三 + 阶段四 (4天)
```

---

## 四、技术细节

### 4.1 特征计算 (区域级)

| 特征 | 计算方式 | 篡改表现 |
|------|----------|----------|
| **ELA差异** | 块内误差标准差 | 篡改区域差异大 |
| **噪声水平** | 块内噪声方差 | 篡改区域噪声不一致 |
| **DCT系数** | 块内DCT分布 | 篡改区域分布异常 |
| **颜色一致性** | 块间颜色差异 | 篡改区域颜色跳变 |
| **纹理一致性** | LBP特征差异 | 篡改区域纹理不同 |

### 4.2 异常分数计算

```python
def compute_anomaly(features, neighbor_features):
    """
    计算异常分数
    
    方法: 马氏距离 (Mahalanobis Distance)
    """
    mean = np.mean(neighbor_features, axis=0)
    cov = np.cov(neighbor_features.T)
    
    # 马氏距离
    diff = features - mean
    inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    score = np.sqrt(diff @ inv_cov @ diff.T)
    
    return score
```

### 4.3 掩码生成流程

```
特征图 → 归一化 → 阈值分割 → 形态学处理 → 最终掩码
   │          │          │            │
   │          │          │            └─ 开闭运算去噪
   │          │          └─ Otsu自适应阈值
   │          └─ Min-Max归一化
   └─ 多特征加权融合
```

---

## 五、预期成果

### 5.1 输出示例

```
输入: tampered_image.jpg
输出:
├── mask.png          # 二值掩码
├── heatmap.png       # 热力图
├── overlay.png       # 叠加显示
└── result.json       # 检测结果详情
```

### 5.2 评估目标

| 指标 | 目标值 |
|------|--------|
| IoU | ≥ 0.5 |
| Precision | ≥ 0.7 |
| Recall | ≥ 0.6 |
| F1-score | ≥ 0.65 |

### 5.3 可视化效果

```
原图          热力图          掩码          叠加效果
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│      │    │ ░░░░ │    │  ██  │    │  ██  │
│  XX  │ →  │ ▓▓▓▓ │ →  │  ██  │ →  │  XX  │
│      │    │ ░░░░ │    │      │    │      │
└──────┘    └──────┘    └──────┘    └──────┘
             (高亮篡改区)
```

---

## 六、风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 传统方法精度有限 | 中 | 多方法融合提升效果 |
| 不同篡改类型检测难度不同 | 中 | 针对性优化各方法 |
| 计算效率 | 低 | 分块并行处理 |
| 阈值敏感性 | 中 | 自适应阈值方法 |

---

**创建时间**: 2026-03-13  
**分支**: greyclaw_0313