"""
饱和度一致性检测
原理：分析图像颜色饱和度的空间一致性
"""

import cv2
import numpy as np

def extract_saturation_features(image_path):
    """提取饱和度特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 转换到HSV空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(float)
    
    h, w = saturation.shape
    block_size = 32
    
    block_sat_means = []
    block_sat_stds = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = saturation[i:i+block_size, j:j+block_size]
            block_sat_means.append(np.mean(block))
            block_sat_stds.append(np.std(block))
    
    if len(block_sat_means) == 0:
        return None
    
    block_sat_means = np.array(block_sat_means)
    block_sat_stds = np.array(block_sat_stds)
    
    features = [
        np.std(block_sat_means),
        np.std(block_sat_stds),
        np.max(block_sat_means) - np.min(block_sat_means)
    ]
    
    return np.array(features)


def detect_tampering_saturation(image_path, threshold=10.0):
    """使用饱和度一致性检测篡改"""
    features = extract_saturation_features(image_path)
    
    if features is None:
        return False, 0
    
    score = features[0] + features[1]
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)