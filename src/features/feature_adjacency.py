"""
邻域一致性检测
原理：分析像素邻域关系的连续性
"""

import cv2
import numpy as np

def extract_adjacency_features(image_path):
    """提取邻域特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 计算邻域差异
    diff_h = np.abs(gray[1:, :].astype(float) - gray[:-1, :].astype(float))
    diff_w = np.abs(gray[:, 1:].astype(float) - gray[:, :-1].astype(float))
    
    # 分块统计
    block_size = 32
    block_consistency = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_h = diff_h[i:i+block_size, j:j+block_size]
            block_w = diff_w[i:i+block_size, j:j+block_size]
            
            mean_h = np.mean(block_h)
            mean_w = np.mean(block_w)
            std_h = np.std(block_h)
            std_w = np.std(block_w)
            
            block_consistency.append([mean_h, mean_w, std_h, std_w])
    
    if len(block_consistency) == 0:
        return None
    
    block_consistency = np.array(block_consistency)
    
    # 计算全局一致性
    consistency_std = np.std(block_consistency, axis=0)
    
    return consistency_std


def detect_tampering_adjacency(image_path, threshold=2.0):
    """使用邻域一致性检测篡改"""
    features = extract_adjacency_features(image_path)
    
    if features is None:
        return False, 0
    
    score = np.mean(features)
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)