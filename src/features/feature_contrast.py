"""
对比度一致性检测
原理：分析图像局部对比度的一致性
"""

import cv2
import numpy as np

def extract_contrast_features(image_path):
    """提取对比度特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    h, w = gray.shape
    block_size = 32
    
    block_contrasts = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            
            # 计算局部对比度
            max_val = np.max(block)
            min_val = np.min(block)
            
            if max_val + min_val > 0:
                contrast = (max_val - min_val) / (max_val + min_val)
            else:
                contrast = 0
            
            block_contrasts.append(contrast)
    
    if len(block_contrasts) == 0:
        return None
    
    block_contrasts = np.array(block_contrasts)
    
    features = [
        np.mean(block_contrasts),
        np.std(block_contrasts),
        np.max(block_contrasts) - np.min(block_contrasts)
    ]
    
    return np.array(features)


def detect_tampering_contrast(image_path, threshold=0.2):
    """使用对比度一致性检测篡改"""
    features = extract_contrast_features(image_path)
    
    if features is None:
        return False, 0
    
    score = features[1] + features[2]
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)