"""
图像篡改检测特征提取模块
推理专用 - 从训练代码中提取
"""

import cv2
import numpy as np
from typing import List

# 尝试导入 skimage (LBP 加速)
try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("警告: 未安装 scikit-image, LBP 将使用慢速实现。建议: pip install scikit-image")


# ============== GB分类器特征提取 ==============

def extract_fft_features(image_path):
    """FFT频域分析"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    h, w = img.shape
    h = 2 ** int(np.log2(h))
    w = 2 ** int(np.log2(w))
    img = img[:h, :w]
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log(magnitude + 1)
    
    center_size = min(h, w) // 8
    center = magnitude_log[h//2-center_size:h//2+center_size,
                           w//2-center_size:w//2+center_size]
    low_freq = np.mean(center)
    
    high_freq_mask = np.ones_like(magnitude_log, dtype=bool)
    edge_size = min(h, w) // 4
    high_freq_mask[h//2-edge_size:h//2+edge_size,
                   w//2-edge_size:w//2+edge_size] = False
    high_freq = np.mean(magnitude_log[high_freq_mask]) if np.any(high_freq_mask) else 0
    
    return float(high_freq / (low_freq + 1e-6))


def extract_resampling_features(image_path):
    """重采样检测"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    gx_fft = np.abs(np.fft.fft2(gx))
    gy_fft = np.abs(np.fft.fft2(gy))
    
    score = np.std(gx_fft) + np.std(gy_fft)
    return float(score / 1000)


def extract_color_features(image_path):
    """颜色一致性"""
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    block_size = 32
    
    block_means = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = hsv[i:i+block_size, j:j+block_size]
            block_means.append(np.mean(block, axis=(0, 1)))
    
    if len(block_means) == 0:
        return 0.0
    
    block_means = np.array(block_means)
    score = np.sum(np.std(block_means, axis=0))
    return float(score)


def extract_jpeg_block_features(image_path):
    """JPEG块效应"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    h, w = img.shape
    block_scores = []
    
    for i in range(7, h-1, 8):
        diff = np.abs(img[i, :].astype(float) - img[i+1, :].astype(float))
        block_scores.append(np.mean(diff))
    
    for j in range(7, w-1, 8):
        diff = np.abs(img[:, j].astype(float) - img[:, j+1].astype(float))
        block_scores.append(np.mean(diff))
    
    return float(np.mean(block_scores)) if block_scores else 0.0


def extract_splicing_features(image_path):
    """拼接检测"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    edges = cv2.Canny(img, 50, 150)
    
    h, w = edges.shape
    block_size = 32
    densities = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = edges[i:i+block_size, j:j+block_size]
            densities.append(np.sum(block > 0) / (block_size * block_size))
    
    if len(densities) == 0:
        return 0.0
    
    score = np.std(densities) * 100
    return float(score)


def extract_cfa_features(image_path):
    """CFA插值检测"""
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = magnitude.shape
    center_region = magnitude[h//2-20:h//2+20, w//2-20:w//2+20]
    
    score = np.std(center_region) / (np.mean(center_region) + 1e-6)
    return float(score)


def extract_contrast_features(image_path):
    """对比度一致性"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    h, w = img.shape
    block_size = 32
    contrasts = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            contrasts.append(np.std(block))
    
    if len(contrasts) == 0:
        return 0.0
    
    score = np.std(contrasts)
    return float(score)


def extract_edge_features(image_path):
    """边缘一致性"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    edges = cv2.Canny(img, 50, 150)
    
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    angles = np.arctan2(gy, gx)
    angles = angles[edges > 0]
    
    if len(angles) == 0:
        return 0.0
    
    score = np.std(angles)
    return float(score)


def extract_jpeg_ghost_features(image_path):
    """JPEG伪影检测"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 90]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    
    diff = np.abs(img.astype(float) - decoded.astype(float))
    score = np.mean(diff)
    return float(score)


def extract_saturation_features(image_path):
    """饱和度一致性"""
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    h, w = saturation.shape
    block_size = 32
    sat_means = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = saturation[i:i+block_size, j:j+block_size]
            sat_means.append(np.mean(block))
    
    if len(sat_means) == 0:
        return 0.0
    
    score = np.std(sat_means)
    return float(score)


# 特征名称列表
FEATURE_NAMES = ['jpeg_block', 'contrast', 'saturation', 'jpeg_ghost', 'fft', 
                 'cfa', 'edge', 'color', 'resampling', 'splicing']

# 特征提取函数映射
FEATURE_EXTRACTORS = {
    'fft': extract_fft_features,
    'resampling': extract_resampling_features,
    'color': extract_color_features,
    'jpeg_block': extract_jpeg_block_features,
    'splicing': extract_splicing_features,
    'cfa': extract_cfa_features,
    'contrast': extract_contrast_features,
    'edge': extract_edge_features,
    'jpeg_ghost': extract_jpeg_ghost_features,
    'saturation': extract_saturation_features,
}


def extract_all_features(image_path: str) -> np.ndarray:
    """提取所有特征 (GB分类器用)"""
    features = []
    for name in FEATURE_NAMES:
        extractor = FEATURE_EXTRACTORS.get(name)
        if extractor:
            try:
                value = extractor(image_path)
                features.append(float(value) if value is not None else 0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
    return np.array(features)


# ============== 像素级特征提取器 ==============

# 特征选择索引 (35维优化版)
# 基于特征重要性筛选的 Top 35 特征
SELECTED_FEATURE_INDICES_35 = [
    # Noise (6) - 最重要
    12, 13, 14, 15, 16, 17,
    # Edge (6) - 重要
    18, 19, 20, 21, 22, 23,
    # DCT (8)
    0, 1, 2, 3, 4, 5, 6, 7,
    # ELA (4)
    8, 9, 10, 11,
    # 纹理 (8)
    24, 25, 26, 27, 28, 29, 30, 31,
    # 频域 (3)
    47, 48, 51
]


class PixelFeatureExtractor:
    """像素级特征提取器 (用于滑动窗口)
    
    支持两种特征维度:
    - 57维: 完整特征集 (默认)
    - 35维: 优化特征集 (与旧模型兼容)
    """
    
    def __init__(self, window_size: int = 32, feature_dim: int = 57):
        """
        初始化特征提取器
        
        Args:
            window_size: 滑动窗口大小
            feature_dim: 特征维度 (57 或 35)
        """
        self.window_size = window_size
        self.half = window_size // 2
        self.feature_dim = feature_dim
        
        if feature_dim == 35:
            self.selected_indices = SELECTED_FEATURE_INDICES_35
        else:
            self.selected_indices = None
    
    def extract(self, patch: np.ndarray) -> np.ndarray:
        """提取增强特征
        
        Args:
            patch: 图像块 (32x32)
            
        Returns:
            特征向量 (57维或35维)
        """
        features = []
        
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        gray = gray.astype(np.float32)
        
        # 1. DCT特征 (8个)
        dct = cv2.dct(gray)
        dct_low = dct[:8, :8]
        dct_high = dct[8:, 8:]
        
        features.append(np.mean(np.abs(dct_low)))
        features.append(np.std(dct_low))
        features.append(np.mean(np.abs(dct_high)))
        features.append(np.std(dct_high))
        features.append(np.percentile(np.abs(dct_low), 95))
        features.append(np.percentile(np.abs(dct_high), 95))
        features.append(np.max(np.abs(dct_low)))
        features.append(np.sum(np.abs(dct_high)) / (np.sum(np.abs(dct)) + 1e-8))
        
        # 2. ELA特征 (4个)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded = cv2.imencode('.jpg', patch, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        ela = np.abs(patch.astype(np.float32) - decoded.astype(np.float32))
        ela_flat = ela.flatten()
        
        ela_p95 = np.percentile(ela_flat, 95)
        features.append(np.mean(ela_flat))
        features.append(np.std(ela_flat))
        features.append(ela_p95)
        features.append(np.max(ela_flat))
        
        # 3. Noise特征 (6个)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        noise_abs = np.abs(noise)
        noise_flat = noise_abs.flatten()
        
        # 合并 percentile 计算
        noise_p95, noise_p99 = np.percentile(noise_flat, [95, 99])
        features.append(np.mean(noise_abs))
        features.append(np.std(noise))
        features.append(noise_p95)
        features.append(np.max(noise_abs))
        features.append(noise_p99)
        features.append(np.median(noise_abs))
        
        # 4. Edge特征 (6个)
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.append(np.mean(edges > 0))
        features.append(np.mean(mag))
        features.append(np.std(mag))
        features.append(np.max(mag))
        features.append(np.percentile(mag, 95))
        features.append(np.sum(mag > np.percentile(mag, 90)) / mag.size)
        
        # 5. 纹理特征 (8个)
        diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
        diff_v = np.abs(gray[1:, :] - gray[:-1, :])
        
        features.append(np.mean(diff_h))
        features.append(np.std(diff_h))
        features.append(np.mean(diff_v))
        features.append(np.std(diff_v))
        features.append(np.percentile(diff_h, 95))
        features.append(np.percentile(diff_v, 95))
        features.append(np.percentile(np.abs(gray[1:, 1:] - gray[:-1, :-1]), 95))
        features.append(np.std(gray))
        
        # 6. Color特征 (5个)
        if len(patch.shape) == 3:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            features.append(np.std(hsv[:,:,0]))
            features.append(np.std(hsv[:,:,1]))
            features.append(np.std(hsv[:,:,2]))
            features.append(np.std(patch[:,:,0]))
            features.append(np.std(patch[:,:,1]))
        else:
            features.extend([0] * 5)
        
        # 7. LBP特征 (8个)
        lbp = self._compute_lbp(gray.astype(np.uint8))
        lbp_hist, _ = np.histogram(lbp, bins=8, range=(0, 256))
        lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-8)
        features.extend(lbp_hist.tolist())
        
        # 8. 频域特征 (6个)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        low_freq = magnitude[center_h-radius:center_h+radius, center_w-radius:center_w+radius]
        features.append(np.mean(low_freq))
        features.append(np.std(low_freq))
        
        high_freq_mask = np.ones_like(magnitude, dtype=bool)
        high_freq_mask[center_h-radius:center_h+radius, center_w-radius:center_w+radius] = False
        high_freq = magnitude[high_freq_mask]
        features.append(np.mean(high_freq))
        features.append(np.std(high_freq))
        
        features.append(np.sum(low_freq) / (np.sum(magnitude) + 1e-8))
        features.append(np.percentile(high_freq, 95))
        
        # 9. 局部对比度特征 (6个)
        local_mean = cv2.blur(gray, (8, 8))
        local_var = cv2.blur((gray - local_mean)**2, (8, 8))
        
        features.append(np.mean(local_var))
        features.append(np.std(local_var))
        features.append(np.percentile(local_var, 95))
        
        local_contrast = np.sqrt(local_var)
        features.append(np.mean(local_contrast))
        features.append(np.std(local_contrast))
        features.append(np.percentile(local_contrast, 95))
        
        features_array = np.array(features, dtype=np.float32)
        
        # 特征选择 (35维)
        if self.selected_indices is not None:
            features_array = features_array[self.selected_indices]
        
        return features_array
    
    def _compute_lbp(self, gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """计算LBP特征 - 与训练脚本保持一致"""
        if HAS_SKIMAGE:
            # 使用 skimage 向量化实现 (与训练一致)
            # method='uniform' 产生 0-58 的编码，但我们在 bins=8 范围内统计
            return local_binary_pattern(gray, P=n_points, R=radius, method='uniform').astype(np.uint8)
        else:
            # 慢速备用实现
            h, w = gray.shape
            lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)
            
            for i in range(radius, h - radius):
                for j in range(radius, w - radius):
                    center = gray[i, j]
                    code = 0
                    for p in range(n_points):
                        angle = 2 * np.pi * p / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if gray[x, y] >= center:
                            code |= (1 << p)
                    lbp[i - radius, j - radius] = code
            
            return lbp


# 兼容性别名
EnhancedFeatureExtractor = PixelFeatureExtractor