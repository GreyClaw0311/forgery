# 图像篡改区域检测方案 v2.0

**项目名称**: Forgery Region Detection  
**分支**: greyclaw_0313  
**更新时间**: 2026-03-13  
**作者**: 灰 (OpenClaw Assistant)

---

## 一、问题背景与描述

### 1.1 任务定义

| 任务类型 | 描述 | 输出 |
|---------|------|------|
| **图像篡改分类** (之前) | 判断整张图片是否被篡改 | 二分类标签 (篡改/正常) |
| **图像篡改区域检测** (当前) | 检测图片中哪些区域被篡改 | 篡改区域的二值掩码 (Mask) |

### 1.2 约束条件

- ❌ **禁用深度学习方法** (CNN、U-Net等)
- ❌ **禁用机器学习方法** (分类器训练)
- ✅ **仅使用传统图像处理方法**

---

## 二、前人工作调研

### 2.1 经典传统方法综述

#### 方法一：Error Level Analysis (ELA)

**来源**: Krawetz, N. (2007). "A Picture's Worth"

**原理**:
- JPEG压缩是有损压缩，不同区域如果经历过不同的压缩历史，在重新压缩时会表现出不同的误差特征
- 篡改区域通常经过二次压缩，误差特征与原始区域不同

**效果**: 对JPEG拼接篡改检测效果较好

**局限**: 
- 对无损压缩图像无效
- 对小范围篡改敏感度较低

**实现要点**:
```python
def ela_detection(image, quality=90):
    # 重新JPEG压缩
    encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    reconstructed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    # 计算差异
    diff = np.abs(image.astype(float) - reconstructed.astype(float))
    
    # 生成热力图
    heatmap = np.max(diff, axis=2)
    
    return heatmap
```

---

#### 方法二：Noise Consistency Analysis

**来源**: Mahdian, B., & Saic, S. (2009). "Using noise inconsistencies for blind image forensics"

**原理**:
- 每个相机传感器都有固定的噪声模式
- 篡改区域的噪声模式会与原始区域不一致
- 通过估计局部噪声水平，检测不一致区域

**效果**: 对各类篡改都有一定检测能力

**实现要点**:
```python
def estimate_noise(image):
    """使用Mihcak方法估计局部噪声"""
    # 使用小波分解估计噪声
    # 或者使用高通滤波
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高通滤波
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    high_freq = cv2.filter2D(gray.astype(float), -1, kernel)
    
    # 分块计算噪声方差
    block_size = 32
    noise_map = np.zeros_like(gray, dtype=float)
    
    for i in range(0, gray.shape[0], block_size):
        for j in range(0, gray.shape[1], block_size):
            block = high_freq[i:i+block_size, j:j+block_size]
            noise_map[i:i+block_size, j:j+block_size] = np.var(block)
    
    return noise_map
```

---

#### 方法三：DCT Block Artifact Analysis

**来源**: Ye, S., et al. (2007). "Detecting digital image forgeries by measuring inconsistencies of blocking artifacts"

**原理**:
- JPEG压缩产生8x8块效应
- 篡改区域的块效应与原始区域不一致
- 通过分析块边界的不连续性检测篡改

**效果**: 对JPEG图像篡改检测效果好

**实现要点**:
```python
def dct_block_analysis(image):
    """DCT块效应分析"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算8x8块边界的跳变
    h, w = gray.shape
    block_artifact_map = np.zeros_like(gray, dtype=float)
    
    # 水平边界
    for i in range(7, h-1, 8):
        diff = np.abs(gray[i, :].astype(float) - gray[i+1, :].astype(float))
        block_artifact_map[i, :] = diff
    
    # 垂直边界
    for j in range(7, w-1, 8):
        diff = np.abs(gray[:, j].astype(float) - gray[:, j+1].astype(float))
        block_artifact_map[:, j] += diff
    
    return block_artifact_map
```

---

#### 方法四：Copy-Move Forgery Detection (CMFD)

**来源**: Amerini, I., et al. (2011). "A SIFT-based forensic method for copy–move attack detection"

**原理**:
- 复制移动篡改会在图像中产生相似区域
- 通过特征点匹配检测重复区域
- 使用SIFT/SURF等特征

**效果**: 专门针对Copy-Move篡改

**实现要点**:
```python
def copy_move_detection(image):
    """基于SIFT的复制移动检测"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # SIFT特征提取
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # 特征匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors, k=2)
    
    # 过滤自匹配和低质量匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            # 排除自匹配
            if abs(keypoints[m.queryIdx].pt[0] - keypoints[m.trainIdx].pt[0]) > 50 or \
               abs(keypoints[m.queryIdx].pt[1] - keypoints[m.trainIdx].pt[1]) > 50:
                good_matches.append((m, keypoints[m.queryIdx], keypoints[m.trainIdx]))
    
    # 生成掩码
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for match, kp1, kp2 in good_matches:
        cv2.circle(mask, (int(kp1.pt[0]), int(kp1.pt[1])), 10, 255, -1)
        cv2.circle(mask, (int(kp2.pt[0]), int(kp2.pt[1])), 10, 255, -1)
    
    return mask
```

---

#### 方法五：CFA Interpolation Detection

**来源**: Popescu, A. C., & Farid, H. (2005). "Exposing digital forgeries in color filter array interpolated images"

**原理**:
- 大多数数码相机使用CFA (Color Filter Array) 进行插值
- 篡改区域可能破坏原始的CFA插值模式
- 通过检测插值痕迹的一致性判断篡改

**效果**: 对从未压缩图像的篡改检测效果好

---

#### 方法六：Splicing Detection via Camera Response Function

**来源**: Ng, T. T., et al. (2007). "Physics-motivated features for distinguishing photographic images and computer graphics"

**原理**:
- 不同相机有不同的响应函数
- 拼接区域的响应函数可能不一致
- 通过估计局部响应函数检测篡改

---

### 2.2 方法对比

| 方法 | 适用篡改类型 | 计算复杂度 | 检测效果 | 主要局限 |
|------|-------------|-----------|---------|---------|
| **ELA** | 拼接、重压缩 | 低 | 中 | 仅JPEG图像 |
| **噪声分析** | 各类篡改 | 中 | 中 | 需要足够的噪声差异 |
| **DCT块效应** | JPEG篡改 | 低 | 中高 | 仅JPEG图像 |
| **复制移动检测** | Copy-Move | 中 | 高 | 仅复制移动篡改 |
| **CFA插值** | 各类篡改 | 高 | 中 | 计算复杂 |
| **相机响应函数** | 拼接 | 高 | 中 | 需要相机信息 |

### 2.3 学术界最佳实践

**基于调研的建议组合**:

1. **JPEG图像篡改检测最佳组合**:
   - ELA + DCT块效应 + 噪声分析
   
2. **通用篡改检测最佳组合**:
   - 噪声分析 + CFA插值 + 分块异常检测
   
3. **Copy-Move专门检测**:
   - SIFT特征匹配 + 块匹配

---

## 三、改进的实现方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        篡改区域检测系统 v2.0                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   输入图像                                                                  │
│      │                                                                      │
│      ▼                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    第一层：图像类型识别                              │  │
│   │   判断是否为JPEG图像 → 选择最优检测方法组合                         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    第二层：并行多特征提取                            │  │
│   │                                                                     │  │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │  │
│   │   │   ELA   │ │ 噪声分析 │ │ DCT块效应│ │复制移动  │ │ 颜色一致 │    │  │
│   │   │ 热力图  │ │  热力图  │ │  热力图  │ │  检测   │ │  性检测  │    │  │
│   │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │  │
│   │        │           │           │           │           │          │  │
│   └────────┼───────────┼───────────┼───────────┼───────────┼──────────┘  │
│            │           │           │           │           │              │
│            └───────────┴───────────┼───────────┴───────────┘              │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    第三层：自适应融合策略                            │  │
│   │                                                                     │  │
│   │   根据图像类型和各方法置信度，自适应加权融合                        │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    第四层：后处理优化                                │  │
│   │                                                                     │  │
│   │   形态学操作 → 连通域分析 → 边界平滑 → 最终掩码                     │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│   输出: 篡改区域掩码 + 热力图 + 置信度图                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心模块设计

#### 模块一：ELA热力图生成器 (改进版)

```python
class ELADetector:
    """ELA热力图检测器 - 改进版"""
    
    def __init__(self, quality_levels=[75, 85, 95]):
        """
        多质量级别ELA，提高鲁棒性
        
        参考: Krawetz (2007) 的多重压缩质量方法
        """
        self.quality_levels = quality_levels
    
    def detect(self, image):
        heatmaps = []
        
        for quality in self.quality_levels:
            # 重新压缩
            _, encoded = cv2.imencode('.jpg', image, 
                                      [cv2.IMWRITE_JPEG_QUALITY, quality])
            reconstructed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            # 计算差异
            diff = np.abs(image.astype(float) - reconstructed.astype(float))
            
            # 取RGB三通道最大值
            heatmap = np.max(diff, axis=2)
            heatmaps.append(heatmap)
        
        # 多质量级别融合
        final_heatmap = np.mean(heatmaps, axis=0)
        
        # 高斯模糊平滑
        final_heatmap = cv2.GaussianBlur(final_heatmap, (5, 5), 0)
        
        return final_heatmap
    
    def get_mask(self, heatmap, threshold=None):
        """自适应阈值分割"""
        if threshold is None:
            # Otsu自适应阈值
            _, mask = cv2.threshold(heatmap.astype(np.uint8), 
                                    0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(heatmap.astype(np.uint8), 
                                    threshold, 255, 
                                    cv2.THRESH_BINARY)
        return mask
```

#### 模块二：噪声一致性检测器

```python
class NoiseConsistencyDetector:
    """噪声一致性检测器"""
    
    def __init__(self, block_size=32):
        self.block_size = block_size
    
    def estimate_local_noise(self, image):
        """
        使用Mihcak方法估计局部噪声方差
        
        参考: Mahdian & Saic (2009)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用小波分解估计噪声
        # 或使用高通滤波 + 分块方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        h, w = gray.shape
        noise_map = np.zeros((h // self.block_size, w // self.block_size))
        
        for i in range(0, h - self.block_size, self.block_size):
            for j in range(0, w - self.block_size, self.block_size):
                block = laplacian[i:i+self.block_size, j:j+self.block_size]
                noise_map[i//self.block_size, j//self.block_size] = np.var(block)
        
        return noise_map
    
    def detect_inconsistency(self, noise_map):
        """检测噪声不一致区域"""
        # 计算局部噪声与周围块的差异
        h, w = noise_map.shape
        inconsistency = np.zeros_like(noise_map)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                # 8邻域
                neighbors = noise_map[i-1:i+2, j-1:j+2].flatten()
                neighbors = np.delete(neighbors, 4)  # 排除中心
                
                # 计算z-score
                mean = np.mean(neighbors)
                std = np.std(neighbors)
                if std > 0:
                    inconsistency[i, j] = abs(noise_map[i, j] - mean) / std
        
        # 上采样到原始尺寸
        inconsistency_full = cv2.resize(inconsistency, (w * self.block_size, h * self.block_size))
        
        return inconsistency_full
```

#### 模块三：DCT块效应检测器

```python
class DCTBlockDetector:
    """DCT块效应检测器"""
    
    def detect(self, image):
        """
        检测JPEG块效应不一致
        
        参考: Ye et al. (2007)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 方法1: 块边界跳变检测
        boundary_map = np.zeros_like(gray, dtype=float)
        
        # 水平边界 (每8像素)
        for i in range(7, h-1, 8):
            diff = np.abs(gray[i, :].astype(float) - gray[i+1, :].astype(float))
            boundary_map[i, :] = diff
        
        # 垂直边界 (每8像素)
        for j in range(7, w-1, 8):
            diff = np.abs(gray[:, j].astype(float) - gray[:, j+1].astype(float))
            boundary_map[:, j] = np.maximum(boundary_map[:, j], diff)
        
        # 计算局部块效应的一致性
        block_size = 32
        artifact_consistency = np.zeros((h // block_size, w // block_size))
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = boundary_map[i:i+block_size, j:j+block_size]
                
                # 统计8x8边界上的平均值
                boundary_pixels = []
                for bi in range(7, block_size, 8):
                    boundary_pixels.extend(block[bi, :].tolist())
                for bj in range(7, block_size, 8):
                    boundary_pixels.extend(block[:, bj].tolist())
                
                artifact_consistency[i//block_size, j//block_size] = np.mean(boundary_pixels)
        
        # 检测不一致
        mean_artifact = np.mean(artifact_consistency)
        std_artifact = np.std(artifact_consistency)
        
        anomaly_map = np.abs(artifact_consistency - mean_artifact) / (std_artifact + 1e-6)
        
        # 上采样
        anomaly_full = cv2.resize(anomaly_map, (w, h))
        
        return anomaly_full
```

#### 模块四：复制移动检测器

```python
class CopyMoveDetector:
    """复制移动检测器 - 基于SIFT"""
    
    def __init__(self, min_distance=50, ratio_thresh=0.75):
        self.min_distance = min_distance
        self.ratio_thresh = ratio_thresh
    
    def detect(self, image):
        """
        基于SIFT的复制移动检测
        
        参考: Amerini et al. (2011)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # SIFT特征提取
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 2:
            return np.zeros(gray.shape, dtype=np.uint8)
        
        # 特征匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors, descriptors, k=2)
        
        # 过滤
        good_pairs = []
        for m, n in matches:
            if m.distance < self.ratio_thresh * n.distance:
                kp1 = keypoints[m.queryIdx]
                kp2 = keypoints[m.trainIdx]
                
                # 排除自匹配
                dist = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + 
                              (kp1.pt[1] - kp2.pt[1])**2)
                if dist > self.min_distance:
                    good_pairs.append((kp1, kp2))
        
        # 生成掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        for kp1, kp2 in good_pairs:
            # 在匹配点周围画圆
            cv2.circle(mask, (int(kp1.pt[0]), int(kp1.pt[1])), 15, 255, -1)
            cv2.circle(mask, (int(kp2.pt[0]), int(kp2.pt[1])), 15, 255, -1)
        
        # 聚类匹配点对 (检测多个复制区域)
        if len(good_pairs) > 0:
            mask = self._cluster_and_refine(mask, good_pairs)
        
        return mask
    
    def _cluster_and_refine(self, mask, pairs):
        """聚类和细化掩码"""
        # 使用形态学操作连接邻近区域
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
```

#### 模块五：自适应融合器

```python
class AdaptiveFusion:
    """自适应多方法融合"""
    
    def __init__(self):
        self.weights = {
            'ela': 0.25,
            'noise': 0.25,
            'dct': 0.25,
            'copy_move': 0.25
        }
    
    def fusion(self, heatmaps, is_jpeg=True):
        """
        自适应融合多个热力图
        
        Args:
            heatmaps: dict of {method_name: heatmap}
            is_jpeg: 是否为JPEG图像
        """
        # 根据图像类型调整权重
        if is_jpeg:
            weights = {
                'ela': 0.35,
                'noise': 0.20,
                'dct': 0.35,
                'copy_move': 0.10
            }
        else:
            weights = {
                'ela': 0.10,
                'noise': 0.40,
                'dct': 0.10,
                'copy_move': 0.40
            }
        
        # 归一化热力图
        normalized = {}
        for name, heatmap in heatmaps.items():
            if heatmap.max() > 0:
                normalized[name] = heatmap / heatmap.max()
            else:
                normalized[name] = heatmap
        
        # 加权融合
        fused = np.zeros_like(list(normalized.values())[0])
        total_weight = 0
        
        for name, heatmap in normalized.items():
            if name in weights:
                fused += weights[name] * heatmap
                total_weight += weights[name]
        
        if total_weight > 0:
            fused /= total_weight
        
        return fused
    
    def threshold(self, heatmap, method='otsu'):
        """阈值分割"""
        if method == 'otsu':
            # Otsu自适应阈值
            _, mask = cv2.threshold((heatmap * 255).astype(np.uint8),
                                    0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            # 自适应阈值
            mask = cv2.adaptiveThreshold((heatmap * 255).astype(np.uint8),
                                         255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY,
                                         11, 2)
        else:
            # 固定阈值
            _, mask = cv2.threshold((heatmap * 255).astype(np.uint8),
                                    127, 255,
                                    cv2.THRESH_BINARY)
        
        return mask
```

#### 模块六：后处理器

```python
class PostProcessor:
    """掩码后处理"""
    
    def __init__(self, min_area=100):
        self.min_area = min_area
    
    def process(self, mask):
        """
        后处理流程:
        1. 形态学开运算 (去噪)
        2. 形态学闭运算 (填孔)
        3. 连通域过滤
        4. 边界平滑
        """
        # 1. 开运算
        kernel_open = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # 2. 闭运算
        kernel_close = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 3. 连通域过滤
        mask = self._remove_small_regions(mask, self.min_area)
        
        # 4. 边界平滑
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _remove_small_regions(self, mask, min_area):
        """移除小区域"""
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        # 创建新掩码
        new_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # 跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                new_mask[labels == i] = 255
        
        return new_mask
```

### 3.3 完整Pipeline

```python
class ForgeryRegionDetector:
    """篡改区域检测主流程"""
    
    def __init__(self):
        self.ela_detector = ELADetector()
        self.noise_detector = NoiseConsistencyDetector()
        self.dct_detector = DCTBlockDetector()
        self.copy_move_detector = CopyMoveDetector()
        self.fusion = AdaptiveFusion()
        self.postprocessor = PostProcessor()
    
    def detect(self, image_path, output_dir=None):
        """
        检测篡改区域
        
        Returns:
            dict: {
                'mask': 二值掩码,
                'heatmap': 热力图,
                'confidence': 置信度图,
                'method_scores': 各方法得分
            }
        """
        # 读取图像
        image = cv2.imread(image_path)
        
        # 判断是否为JPEG
        is_jpeg = self._is_jpeg(image_path)
        
        # 并行计算各方法热力图
        heatmaps = {
            'ela': self.ela_detector.detect(image),
            'noise': self.noise_detector.detect_inconsistency(
                self.noise_detector.estimate_local_noise(image)
            ),
            'dct': self.dct_detector.detect(image),
            'copy_move': self.copy_move_detector.detect(image).astype(float) / 255
        }
        
        # 自适应融合
        fused_heatmap = self.fusion.fusion(heatmaps, is_jpeg)
        
        # 阈值分割
        mask = self.fusion.threshold(fused_heatmap, method='otsu')
        
        # 后处理
        mask = self.postprocessor.process(mask)
        
        return {
            'mask': mask,
            'heatmap': fused_heatmap,
            'confidence': self._compute_confidence(fused_heatmap, mask),
            'method_scores': {k: np.mean(v) for k, v in heatmaps.items()}
        }
    
    def _is_jpeg(self, image_path):
        """判断是否为JPEG图像"""
        ext = os.path.splitext(image_path)[1].lower()
        return ext in ['.jpg', '.jpeg']
    
    def _compute_confidence(self, heatmap, mask):
        """计算置信度图"""
        # 篡改区域的平均热力值作为置信度
        confidence = np.zeros_like(heatmap)
        tampered_pixels = mask > 0
        
        if np.any(tampered_pixels):
            mean_tampered = np.mean(heatmap[tampered_pixels])
            confidence[tampered_pixels] = heatmap[tampered_pixels] / (mean_tampered + 1e-6)
        
        return np.clip(confidence, 0, 1)
```

---

## 四、评估方法

### 4.1 评估指标

```python
def evaluate_iou(pred_mask, gt_mask):
    """计算IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / (union + 1e-6)

def evaluate_pixel_metrics(pred_mask, gt_mask):
    """计算像素级指标"""
    tp = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
    fp = np.logical_and(pred_mask > 0, gt_mask == 0).sum()
    fn = np.logical_and(pred_mask == 0, gt_mask > 0).sum()
    tn = np.logical_and(pred_mask == 0, gt_mask == 0).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': evaluate_iou(pred_mask, gt_mask)
    }
```

---

## 五、步骤排期

### 阶段一：基础框架 (2天)

| 任务 | 工期 |
|------|------|
| 项目结构搭建 | 0.5天 |
| 分块处理工具 | 0.5天 |
| 可视化模块 | 0.5天 |
| 测试数据准备 | 0.5天 |

### 阶段二：单方法实现 (4天)

| 任务 | 工期 |
|------|------|
| ELA热力图检测器 | 1天 |
| 噪声一致性检测器 | 1天 |
| DCT块效应检测器 | 1天 |
| 复制移动检测器 | 1天 |

### 阶段三：融合优化 (2天)

| 任务 | 工期 |
|------|------|
| 自适应融合策略 | 1天 |
| 后处理优化 | 0.5天 |
| 参数调优 | 0.5天 |

### 阶段四：评估文档 (2天)

| 任务 | 工期 |
|------|------|
| 测试集评估 | 1天 |
| 实验报告 | 0.5天 |
| README完善 | 0.5天 |

---

## 六、预期成果

| 指标 | 目标值 |
|------|--------|
| IoU | ≥ 0.45 |
| Precision | ≥ 0.65 |
| Recall | ≥ 0.55 |
| F1-score | ≥ 0.60 |

---

**更新时间**: 2026-03-13  
**分支**: greyclaw_0313