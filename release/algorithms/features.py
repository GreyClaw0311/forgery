"""
图像篡改检测特征提取模块
推理专用 - 优化版

优化点:
1. 全局 ELA 预计算 - 避免每个窗口重复 JPEG 编解码
2. LBP 向量化 - skimage 加速
3. percentile 合并计算
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# 尝试导入 skimage (LBP 加速)
try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


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


# ============== 全局特征预计算器 ==============

class GlobalFeatureCache:
    """
    全局特征预计算缓存
    
    对整张图片预先计算特征图，滑动窗口时只需取局部统计
    避免每个窗口重复计算 ELA/DCT 等耗时操作
    """
    
    def __init__(self, image: np.ndarray, quality: int = 90):
        """
        预计算全局特征图
        
        Args:
            image: BGR 图像
            quality: ELA JPEG 质量
        """
        self.image = image
        self.h, self.w = image.shape[:2]
        
        # 转换灰度图
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            self.gray = image.astype(np.float32)
        
        # 预计算全局特征图
        self._compute_ela_map(quality)
        self._compute_noise_map()
        self._compute_edge_map()
        self._compute_dct_map()
        self._compute_lbp_map()
        self._compute_frequency_map()
        self._compute_local_contrast_map()
    
    def _compute_ela_map(self, quality: int):
        """预计算 ELA 特征图 (一次 JPEG 编解码)"""
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, encoded = cv2.imencode('.jpg', self.image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # ELA 差异图 (3通道)
        self.ela_map = np.abs(self.image.astype(np.float32) - decoded.astype(np.float32))
        
        # ELA 灰度图 (单通道)
        self.ela_gray = np.max(self.ela_map, axis=2)
    
    def _compute_noise_map(self):
        """预计算噪声特征图"""
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        self.noise_map = np.abs(self.gray - blurred)
    
    def _compute_edge_map(self):
        """预计算边缘特征图"""
        self.edge_binary = cv2.Canny(self.gray.astype(np.uint8), 50, 150)
        
        sobel_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        self.edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    def _compute_dct_map(self):
        """预计算 DCT 特征图 (块级)"""
        h, w = self.gray.shape
        block_size = 8
        
        # DCT 块系数图
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size
        
        self.dct_dc_map = np.zeros((n_blocks_h, n_blocks_w))
        self.dct_ac_std_map = np.zeros((n_blocks_h, n_blocks_w))
        
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = self.gray[i*block_size:(i+1)*block_size,
                                  j*block_size:(j+1)*block_size]
                dct = cv2.dct(block)
                self.dct_dc_map[i, j] = dct[0, 0]
                self.dct_ac_std_map[i, j] = np.std(dct[1:, 1:])
        
        # 上采样到原图尺寸
        self.dct_dc_full = cv2.resize(self.dct_dc_map, (w, h))
        self.dct_ac_std_full = cv2.resize(self.dct_ac_std_map, (w, h))
    
    def _compute_lbp_map(self):
        """预计算 LBP 特征图"""
        if HAS_SKIMAGE:
            self.lbp_map = local_binary_pattern(
                self.gray.astype(np.uint8), P=8, R=1, method='uniform'
            )
        else:
            # 慢速备用
            self.lbp_map = self._compute_lbp_slow(self.gray.astype(np.uint8))
    
    def _compute_lbp_slow(self, gray: np.ndarray) -> np.ndarray:
        """慢速 LBP 计算"""
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.float32)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                for p in range(8):
                    angle = 2 * np.pi * p / 8
                    x = int(i + np.cos(angle))
                    y = int(j + np.sin(angle))
                    if gray[x, y] >= center:
                        code |= (1 << p)
                lbp[i - 1, j - 1] = code
        
        # 填充边界
        return np.pad(lbp, ((1, 1), (1, 1)), mode='edge')
    
    def _compute_frequency_map(self):
        """预计算频域特征图"""
        f = np.fft.fft2(self.gray)
        fshift = np.fft.fftshift(f)
        self.freq_magnitude = np.abs(fshift)
    
    def _compute_local_contrast_map(self):
        """预计算局部对比度特征图"""
        self.local_mean = cv2.blur(self.gray, (8, 8))
        self.local_var = cv2.blur((self.gray - self.local_mean)**2, (8, 8))
        self.local_contrast = np.sqrt(self.local_var)


# ============== 快速像素级特征提取器 ==============

# 特征选择索引 (35维优化版)
SELECTED_FEATURE_INDICES_35 = [
    12, 13, 14, 15, 16, 17,  # Noise (6)
    18, 19, 20, 21, 22, 23,  # Edge (6)
    0, 1, 2, 3, 4, 5, 6, 7,  # DCT (8)
    8, 9, 10, 11,            # ELA (4)
    24, 25, 26, 27, 28, 29, 30, 31,  # 纹理 (8)
    47, 48, 51               # 频域 (3)
]

# 特征选择索引 (49维无LBP版)
# 57维特征中移除8个LBP特征 (索引37-44)
# 特征顺序: DCT(0-7), ELA(8-11), Noise(12-17), Edge(18-23), 纹理(24-31), Color(32-36), LBP(37-44), 频域(45-50), 局部对比度(51-56)
SELECTED_FEATURE_INDICES_49 = [
    0, 1, 2, 3, 4, 5, 6, 7,  # DCT (8)
    8, 9, 10, 11,            # ELA (4)
    12, 13, 14, 15, 16, 17,  # Noise (6)
    18, 19, 20, 21, 22, 23,  # Edge (6)
    24, 25, 26, 27, 28, 29, 30, 31,  # 纹理 (8)
    32, 33, 34, 35, 36,      # Color (5)
    # 跳过 LBP (37-44)
    45, 46, 47, 48, 49, 50,  # 频域 (6)
    51, 52, 53, 54, 55, 56   # 局部对比度 (6)
]


class FastPixelFeatureExtractor:
    """
    快速像素级特征提取器
    
    使用全局预计算特征图，滑动窗口只需取局部统计
    相比逐窗口计算，加速 5-10 倍
    
    支持特征维度:
    - 57: 完整特征 (含LBP)
    - 49: 优化特征 (无LBP，推荐)
    - 35: 精简特征 (兼容旧模型)
    """
    
    def __init__(self, window_size: int = 32, feature_dim: int = 57, use_lbp: bool = None):
        """
        初始化
        
        Args:
            window_size: 窗口大小
            feature_dim: 特征维度 (57/49/35)
            use_lbp: 是否使用LBP特征 (None表示根据feature_dim自动判断)
        """
        self.window_size = window_size
        self.half = window_size // 2
        self.feature_dim = feature_dim
        
        # 自动判断是否使用LBP
        if use_lbp is None:
            self.use_lbp = (feature_dim == 57)
        else:
            self.use_lbp = use_lbp
        
        # 特征选择
        if feature_dim == 35:
            self.selected_indices = SELECTED_FEATURE_INDICES_35
        elif feature_dim == 49:
            # 49维通过use_lbp=False直接实现，不需要索引选择
            self.selected_indices = None
        else:
            self.selected_indices = None
        
        self.cache = None
    
    def set_cache(self, cache: GlobalFeatureCache):
        """设置全局特征缓存"""
        self.cache = cache
    
    def extract_from_cache(self, y: int, x: int) -> np.ndarray:
        """
        从全局缓存中提取窗口特征 (快速)
        
        Args:
            y: 窗口中心 y 坐标
            x: 窗口中心 x 坐标
            
        Returns:
            特征向量 (57维或35维)
        """
        if self.cache is None:
            raise ValueError("请先调用 set_cache() 设置全局特征缓存")
        
        half = self.half
        features = []
        
        # 获取窗口区域
        y1, y2 = y - half, y + half
        x1, x2 = x - half, x + half
        
        # 从缓存中提取
        gray_patch = self.cache.gray[y1:y2, x1:x2]
        ela_patch = self.cache.ela_gray[y1:y2, x1:x2]
        noise_patch = self.cache.noise_map[y1:y2, x1:x2]
        edge_mag_patch = self.cache.edge_mag[y1:y2, x1:x2]
        dct_dc_patch = self.cache.dct_dc_full[y1:y2, x1:x2]
        dct_ac_patch = self.cache.dct_ac_std_full[y1:y2, x1:x2]
        lbp_patch = self.cache.lbp_map[y1:y2, x1:x2]
        freq_patch = self.cache.freq_magnitude[y1:y2, x1:x2]
        local_var_patch = self.cache.local_var[y1:y2, x1:x2]
        local_contrast_patch = self.cache.local_contrast[y1:y2, x1:x2]
        
        # ===== 1. DCT特征 (8个) - 从预计算DCT图提取 =====
        dct_dc_flat = dct_dc_patch.flatten()
        dct_ac_flat = dct_ac_patch.flatten()
        
        features.append(np.mean(np.abs(dct_dc_flat)))
        features.append(np.std(dct_dc_flat))
        features.append(np.mean(np.abs(dct_ac_flat)))
        features.append(np.std(dct_ac_flat))
        features.append(np.percentile(np.abs(dct_dc_flat), 95))
        features.append(np.percentile(np.abs(dct_ac_flat), 95))
        features.append(np.max(np.abs(dct_dc_flat)))
        
        # DCT 高频比 (从灰度图计算)
        dct = cv2.dct(gray_patch)
        dct_low = dct[:8, :8]
        dct_high = dct[8:, 8:]
        features.append(np.sum(np.abs(dct_high)) / (np.sum(np.abs(dct)) + 1e-8))
        
        # ===== 2. ELA特征 (4个) - 从预计算ELA图提取 =====
        ela_flat = ela_patch.flatten()
        ela_p95 = np.percentile(ela_flat, 95)
        features.append(np.mean(ela_flat))
        features.append(np.std(ela_flat))
        features.append(ela_p95)
        features.append(np.max(ela_flat))
        
        # ===== 3. Noise特征 (6个) - 从预计算噪声图提取 =====
        noise_flat = noise_patch.flatten()
        noise_p95, noise_p99 = np.percentile(noise_flat, [95, 99])
        features.append(np.mean(noise_flat))
        features.append(np.std(noise_patch))
        features.append(noise_p95)
        features.append(np.max(noise_flat))
        features.append(noise_p99)
        features.append(np.median(noise_flat))
        
        # ===== 4. Edge特征 (6个) - 从预计算边缘图提取 =====
        edge_flat = edge_mag_patch.flatten()
        features.append(np.mean(self.cache.edge_binary[y1:y2, x1:x2] > 0))
        features.append(np.mean(edge_flat))
        features.append(np.std(edge_flat))
        features.append(np.max(edge_flat))
        features.append(np.percentile(edge_flat, 95))
        features.append(np.sum(edge_flat > np.percentile(edge_flat, 90)) / edge_flat.size)
        
        # ===== 5. 纹理特征 (8个) =====
        diff_h = np.abs(gray_patch[:, 1:] - gray_patch[:, :-1])
        diff_v = np.abs(gray_patch[1:, :] - gray_patch[:-1, :])
        
        features.append(np.mean(diff_h))
        features.append(np.std(diff_h))
        features.append(np.mean(diff_v))
        features.append(np.std(diff_v))
        features.append(np.percentile(diff_h, 95))
        features.append(np.percentile(diff_v, 95))
        features.append(np.percentile(np.abs(gray_patch[1:, 1:] - gray_patch[:-1, :-1]), 95))
        features.append(np.std(gray_patch))
        
        # ===== 6. Color特征 (5个) =====
        if len(self.cache.image.shape) == 3:
            color_patch = self.cache.image[y1:y2, x1:x2]
            hsv = cv2.cvtColor(color_patch, cv2.COLOR_BGR2HSV)
            features.append(np.std(hsv[:,:,0]))
            features.append(np.std(hsv[:,:,1]))
            features.append(np.std(hsv[:,:,2]))
            features.append(np.std(color_patch[:,:,0]))
            features.append(np.std(color_patch[:,:,1]))
        else:
            features.extend([0] * 5)
        
        # ===== 7. LBP特征 (8个) - 从预计算LBP图提取 (可选) =====
        if self.use_lbp:
            lbp_hist, _ = np.histogram(lbp_patch, bins=8, range=(0, 256))
            lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-8)
            features.extend(lbp_hist.tolist())
        
        # ===== 8. 频域特征 (6个) - 从预计算频谱图提取 =====
        h_p, w_p = freq_patch.shape
        center_h, center_w = h_p // 2, w_p // 2
        radius = min(h_p, w_p) // 4
        
        low_freq = freq_patch[max(0,center_h-radius):center_h+radius,
                              max(0,center_w-radius):center_w+radius]
        features.append(np.mean(low_freq))
        features.append(np.std(low_freq))
        
        high_freq_mask = np.ones_like(freq_patch, dtype=bool)
        high_freq_mask[max(0,center_h-radius):center_h+radius,
                       max(0,center_w-radius):center_w+radius] = False
        high_freq = freq_patch[high_freq_mask]
        features.append(np.mean(high_freq))
        features.append(np.std(high_freq))
        features.append(np.sum(low_freq) / (np.sum(freq_patch) + 1e-8))
        features.append(np.percentile(high_freq, 95))
        
        # ===== 9. 局部对比度特征 (6个) - 从预计算对比度图提取 =====
        local_var_flat = local_var_patch.flatten()
        local_contrast_flat = local_contrast_patch.flatten()
        
        features.append(np.mean(local_var_flat))
        features.append(np.std(local_var_flat))
        features.append(np.percentile(local_var_flat, 95))
        features.append(np.mean(local_contrast_flat))
        features.append(np.std(local_contrast_flat))
        features.append(np.percentile(local_contrast_flat, 95))
        
        # 转换为数组
        features_array = np.array(features, dtype=np.float32)
        
        # 特征选择逻辑:
        # - use_lbp=True, feature_dim=57: 直接返回57维
        # - use_lbp=False, feature_dim=49: 直接返回49维 (已跳过LBP)
        # - feature_dim=35: 从57维中选择35个重要特征
        if self.feature_dim == 35 and self.selected_indices is not None:
            # 35维需要从完整特征中选择
            # 如果use_lbp=False，需要先补齐LBP特征
            if not self.use_lbp:
                # 补零占位，保持索引正确
                lbp_placeholder = [0.0] * 8
                features_array = np.concatenate([features_array[:37], lbp_placeholder, features_array[37:]])
            features_array = features_array[self.selected_indices]
        
        return features_array


# ============== 兼容旧版接口 ==============

class PixelFeatureExtractor:
    """
    像素级特征提取器 (兼容旧版接口)
    
    支持两种模式:
    1. 快速模式 (推荐): 使用 GlobalFeatureCache 预计算
    2. 兼容模式: 逐窗口计算
    
    支持特征维度:
    - 57: 完整特征 (含LBP)
    - 49: 优化特征 (无LBP，推荐)
    - 35: 精简特征 (兼容旧模型)
    """
    
    def __init__(self, window_size: int = 32, feature_dim: int = 57, use_lbp: bool = None):
        self.window_size = window_size
        self.half = window_size // 2
        self.feature_dim = feature_dim
        
        # 自动判断是否使用LBP
        if use_lbp is None:
            self.use_lbp = (feature_dim == 57)
        else:
            self.use_lbp = use_lbp
        
        # 特征选择
        if feature_dim == 35:
            self.selected_indices = SELECTED_FEATURE_INDICES_35
        elif feature_dim == 49:
            # 49维通过use_lbp=False直接实现，不需要索引选择
            self.selected_indices = None
        else:
            self.selected_indices = None
        
        # 快速提取器
        self._fast_extractor = FastPixelFeatureExtractor(window_size, feature_dim, self.use_lbp)
    
    def set_global_cache(self, cache: GlobalFeatureCache):
        """设置全局特征缓存 (启用快速模式)"""
        self._fast_extractor.set_cache(cache)
        self._cache = cache
    
    def extract(self, patch: np.ndarray) -> np.ndarray:
        """
        提取特征
        
        Args:
            patch: 图像块 (32x32)
            
        Returns:
            特征向量
        """
        # 如果有全局缓存，使用快速模式
        if hasattr(self, '_cache') and self._cache is not None:
            # 这种情况应该用 extract_from_cache
            return self._extract_from_patch(patch)
        
        return self._extract_from_patch(patch)
    
    def extract_from_cache(self, y: int, x: int) -> np.ndarray:
        """从缓存提取特征 (快速)"""
        return self._fast_extractor.extract_from_cache(y, x)
    
    def _extract_from_patch(self, patch: np.ndarray) -> np.ndarray:
        """从图像块提取特征 (兼容模式)"""
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
        
        # 7. LBP特征 (8个) - 可选
        if self.use_lbp:
            if HAS_SKIMAGE:
                lbp = local_binary_pattern(gray.astype(np.uint8), P=8, R=1, method='uniform')
            else:
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
        
        # 特征选择逻辑:
        # - use_lbp=True, feature_dim=57: 直接返回57维
        # - use_lbp=False, feature_dim=49: 直接返回49维 (已跳过LBP)
        # - feature_dim=35: 从57维中选择35个重要特征
        if self.feature_dim == 35 and self.selected_indices is not None:
            features_array = features_array[self.selected_indices]
        
        return features_array
    
    def _compute_lbp(self, gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """计算LBP特征"""
        if HAS_SKIMAGE:
            return local_binary_pattern(gray, P=n_points, R=radius, method='uniform').astype(np.uint8)
        
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


# 兼容别名
EnhancedFeatureExtractor = PixelFeatureExtractor