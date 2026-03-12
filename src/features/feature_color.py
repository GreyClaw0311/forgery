"""
颜色一致性检测
原理：分析图像颜色分布的空间一致性
"""

import cv2
import numpy as np

def extract_color_features(image_path):
    """提取颜色特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 转换到HSV空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h, w = hsv.shape[:2]
    block_size = 32
    
    # 分块计算颜色特征
    block_means = []
    block_stds = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = hsv[i:i+block_size, j:j+block_size]
            block_means.append(np.mean(block, axis=(0, 1)))
            block_stds.append(np.std(block, axis=(0, 1)))
    
    if len(block_means) == 0:
        return None
    
    block_means = np.array(block_means)
    block_stds = np.array(block_stds)
    
    # 计算一致性
    mean_consistency = np.std(block_means, axis=0)  # 均值变化
    std_consistency = np.std(block_stds, axis=0)    # 标准差变化
    
    features = np.concatenate([mean_consistency, std_consistency])
    
    return features


def detect_tampering_color(image_path, threshold=5.0):
    """使用颜色一致性检测篡改"""
    features = extract_color_features(image_path)
    
    if features is None:
        return False, 0
    
    # 使用颜色不一致性计算分数
    score = np.mean(features[:3]) * 10  # H, S, V的均值变化
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)