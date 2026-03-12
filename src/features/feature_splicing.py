"""
拼接检测
原理：检测图像中的拼接边缘和区域
"""

import cv2
import numpy as np

def extract_splicing_features(image_path):
    """提取拼接特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    # 使用多个方向的边缘检测
    edges_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_v = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_mag = np.sqrt(edges_h**2 + edges_v**2)
    
    h, w = gray.shape
    block_size = 32
    
    block_edge_scores = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = edges_mag[i:i+block_size, j:j+block_size]
            
            # 边缘强度
            edge_strength = np.mean(block)
            
            # 边缘密度
            edge_density = np.sum(block > 30) / block.size
            
            block_edge_scores.append([edge_strength, edge_density])
    
    if len(block_edge_scores) == 0:
        return None
    
    block_edge_scores = np.array(block_edge_scores)
    
    # 计算边缘一致性
    edge_consistency = np.std(block_edge_scores, axis=0)
    
    # 分析边缘方向直方图
    angles = np.arctan2(edges_h, edges_v)
    angle_hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
    angle_hist = angle_hist / np.sum(angle_hist)
    
    # 边缘方向熵
    hist_nonzero = angle_hist[angle_hist > 0]
    angle_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10))
    
    features = np.concatenate([edge_consistency, [angle_entropy]])
    
    return features


def detect_tampering_splicing(image_path, threshold=2.0):
    """使用拼接检测"""
    features = extract_splicing_features(image_path)
    
    if features is None:
        return False, 0
    
    score = np.mean(features)
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)