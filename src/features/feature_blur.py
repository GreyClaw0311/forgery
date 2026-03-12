"""
模糊检测
原理：检测图像局部模糊程度的一致性
"""

import cv2
import numpy as np

def extract_blur_features(image_path):
    """提取模糊特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    # 使用Laplacian变异性估计模糊度
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    h, w = gray.shape
    block_size = 32
    
    block_blur_scores = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = laplacian[i:i+block_size, j:j+block_size]
            
            # Laplacian变异性的倒数表示模糊程度
            blur_score = 1.0 / (np.var(block) + 1e-10)
            block_blur_scores.append(blur_score)
    
    if len(block_blur_scores) == 0:
        return None
    
    block_blur_scores = np.array(block_blur_scores)
    
    # 使用频域分析
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    # 计算高频能量比
    radius = min(h, w) // 8
    low_freq_mask = np.zeros((h, w), dtype=bool)
    for i in range(h):
        for j in range(w):
            if (i - cy)**2 + (j - cx)**2 <= radius**2:
                low_freq_mask[i, j] = True
    
    low_freq_energy = np.sum(magnitude[low_freq_mask]**2)
    high_freq_energy = np.sum(magnitude[~low_freq_mask]**2)
    freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
    
    features = [
        np.mean(block_blur_scores),
        np.std(block_blur_scores),
        freq_ratio
    ]
    
    return np.array(features)


def detect_tampering_blur(image_path, threshold=0.1):
    """使用模糊检测篡改"""
    features = extract_blur_features(image_path)
    
    if features is None:
        return False, 0
    
    # 标准差越大表示模糊程度不一致
    score = features[1]
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)