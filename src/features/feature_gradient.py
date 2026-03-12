"""
梯度一致性检测
原理：分析图像梯度的空间分布一致性
"""

import cv2
import numpy as np

def extract_gradient_features(image_path):
    """提取梯度特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    # 计算梯度
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    h, w = gray.shape
    block_size = 32
    
    block_features = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_mag = magnitude[i:i+block_size, j:j+block_size]
            block_dir = direction[i:i+block_size, j:j+block_size]
            
            mean_mag = np.mean(block_mag)
            std_mag = np.std(block_mag)
            mean_dir = np.mean(np.abs(block_dir))
            std_dir = np.std(block_dir)
            
            block_features.append([mean_mag, std_mag, mean_dir, std_dir])
    
    if len(block_features) == 0:
        return None
    
    block_features = np.array(block_features)
    
    # 计算全局一致性
    consistency = np.std(block_features, axis=0)
    
    return consistency


def detect_tampering_gradient(image_path, threshold=2.0):
    """使用梯度一致性检测篡改"""
    features = extract_gradient_features(image_path)
    
    if features is None:
        return False, 0
    
    score = np.mean(features)
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)