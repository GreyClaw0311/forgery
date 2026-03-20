"""
图像篡改检测特征提取模块
包含 Top 10 关键特征的实现
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.fftpack import dct


def extract_fft_features(image_path):
    """FFT频域分析 - 重要性: 14.23%"""
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


def detect_tampering_fft(image_path, threshold=0.3):
    score = extract_fft_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


def extract_resampling_features(image_path):
    """重采样检测 - 重要性: 12.09%"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    # 计算梯度
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算周期性
    gx_fft = np.abs(np.fft.fft2(gx))
    gy_fft = np.abs(np.fft.fft2(gy))
    
    score = np.std(gx_fft) + np.std(gy_fft)
    return float(score / 1000)


def detect_tampering_resampling(image_path, threshold=0.5):
    score = extract_resampling_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


def extract_color_features(image_path):
    """颜色一致性 - 重要性: 11.45%"""
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


def detect_tampering_color(image_path, threshold=5.0):
    score = extract_color_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


def extract_jpeg_block_features(image_path):
    """JPEG块效应 - 重要性: 10.48%"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    h, w = img.shape
    block_scores = []
    
    # 检测8x8块边界的跳变
    for i in range(7, h-1, 8):
        diff = np.abs(img[i, :].astype(float) - img[i+1, :].astype(float))
        block_scores.append(np.mean(diff))
    
    for j in range(7, w-1, 8):
        diff = np.abs(img[:, j].astype(float) - img[:, j+1].astype(float))
        block_scores.append(np.mean(diff))
    
    return float(np.mean(block_scores)) if block_scores else 0.0


def detect_tampering_jpeg_block(image_path, threshold=10.0):
    score = extract_jpeg_block_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


def extract_splicing_features(image_path):
    """拼接检测 - 重要性: 9.94%"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    # Canny边缘检测
    edges = cv2.Canny(img, 50, 150)
    
    # 计算边缘密度变化
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


def detect_tampering_splicing(image_path, threshold=0.5):
    score = extract_splicing_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


def extract_cfa_features(image_path):
    """CFA插值检测 - 重要性: 9.53%"""
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    
    # 转换到频域检测插值痕迹
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # 检测周期性峰值（插值痕迹）
    h, w = magnitude.shape
    center_region = magnitude[h//2-20:h//2+20, w//2-20:w//2+20]
    
    score = np.std(center_region) / (np.mean(center_region) + 1e-6)
    return float(score)


def detect_tampering_cfa(image_path, threshold=0.1):
    score = extract_cfa_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


def extract_contrast_features(image_path):
    """对比度一致性 - 重要性: 8.98%"""
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


def detect_tampering_contrast(image_path, threshold=5.0):
    score = extract_contrast_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


def extract_edge_features(image_path):
    """边缘一致性 - 重要性: 8.48%"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    edges = cv2.Canny(img, 50, 150)
    
    # 计算边缘方向的一致性
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    angles = np.arctan2(gy, gx)
    angles = angles[edges > 0]
    
    if len(angles) == 0:
        return 0.0
    
    score = np.std(angles)
    return float(score)


def detect_tampering_edge(image_path, threshold=1.0):
    score = extract_edge_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


def extract_jpeg_ghost_features(image_path):
    """JPEG伪影检测 - 重要性: 7.62%"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    
    # 检测不同压缩质量下的差异
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 90]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    
    diff = np.abs(img.astype(float) - decoded.astype(float))
    score = np.mean(diff)
    return float(score)


def detect_tampering_jpeg_ghost(image_path, threshold=1.0):
    score = extract_jpeg_ghost_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


def extract_saturation_features(image_path):
    """饱和度一致性 - 重要性: 7.20%"""
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


def detect_tampering_saturation(image_path, threshold=10.0):
    score = extract_saturation_features(image_path)
    if score is None:
        return False, 0.0
    return score > threshold, score


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

# 检测函数映射
DETECT_FUNCTIONS = {
    'fft': detect_tampering_fft,
    'resampling': detect_tampering_resampling,
    'color': detect_tampering_color,
    'jpeg_block': detect_tampering_jpeg_block,
    'splicing': detect_tampering_splicing,
    'cfa': detect_tampering_cfa,
    'contrast': detect_tampering_contrast,
    'edge': detect_tampering_edge,
    'jpeg_ghost': detect_tampering_jpeg_ghost,
    'saturation': detect_tampering_saturation,
}


def extract_all_features(image_path):
    """提取所有特征"""
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
