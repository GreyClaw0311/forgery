"""
重采样检测
原理：检测图像中的插值痕迹
"""

import cv2
import numpy as np

def extract_resampling_features(image_path):
    """提取重采样特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    # 计算二阶导数
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # FFT分析周期性
    fft = np.fft.fft2(laplacian)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # 计算径向平均
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_radius = min(h, w) // 4
    
    radial_mean = []
    for r in range(1, max_radius):
        mask = np.zeros((h, w), dtype=bool)
        for angle in np.linspace(0, 2*np.pi, 360):
            y = int(cy + r * np.sin(angle))
            x = int(cx + r * np.cos(angle))
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = True
        if np.sum(mask) > 0:
            radial_mean.append(np.mean(magnitude[mask]))
    
    if len(radial_mean) == 0:
        return None
    
    radial_mean = np.array(radial_mean)
    
    # 分析周期性峰值
    diff = np.diff(radial_mean)
    peaks = np.sum(diff[:-1] > 0)
    
    # 分块分析
    block_size = 64
    block_variances = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = laplacian[i:i+block_size, j:j+block_size]
            block_variances.append(np.var(block))
    
    features = [
        peaks / max_radius if max_radius > 0 else 0,
        np.std(block_variances) if len(block_variances) > 0 else 0
    ]
    
    return np.array(features)


def detect_tampering_resampling(image_path, threshold=0.5):
    """使用重采样检测篡改"""
    features = extract_resampling_features(image_path)
    
    if features is None:
        return False, 0
    
    score = np.mean(features)
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)