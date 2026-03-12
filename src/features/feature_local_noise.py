"""
局部噪声检测
原理：分析图像局部噪声分布的一致性
"""

import cv2
import numpy as np

def extract_local_noise_features(image_path):
    """提取局部噪声特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    # 使用高斯滤波估计无噪声图像
    denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # 估计噪声
    noise = gray - denoised
    
    h, w = gray.shape
    block_size = 32
    
    block_noises = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = noise[i:i+block_size, j:j+block_size]
            
            # 计算噪声统计
            noise_mean = np.mean(block)
            noise_std = np.std(block)
            noise_kurtosis = np.mean((block - noise_mean)**4) / (noise_std**4 + 1e-10)
            
            block_noises.append([noise_std, noise_kurtosis])
    
    if len(block_noises) == 0:
        return None
    
    block_noises = np.array(block_noises)
    
    # 计算噪声一致性
    noise_consistency = np.std(block_noises, axis=0)
    
    return noise_consistency


def detect_tampering_local_noise(image_path, threshold=0.5):
    """使用局部噪声检测篡改"""
    features = extract_local_noise_features(image_path)
    
    if features is None:
        return False, 0
    
    score = np.mean(features)
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)