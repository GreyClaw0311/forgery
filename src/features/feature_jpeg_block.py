"""
JPEG块效应检测
原理：检测JPEG压缩的8x8块边界伪影
"""

import cv2
import numpy as np

def extract_jpeg_block_features(image_path):
    """提取JPEG块效应特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    h, w = gray.shape
    block_size = 8
    
    # 计算块边界的跳变
    horizontal_artifacts = []
    vertical_artifacts = []
    
    # 水平边界
    for i in range(block_size, h - block_size, block_size):
        diff = np.abs(gray[i, :] - gray[i-1, :])
        horizontal_artifacts.append(np.mean(diff))
    
    # 垂直边界
    for j in range(block_size, w - block_size, block_size):
        diff = np.abs(gray[:, j] - gray[:, j-1])
        vertical_artifacts.append(np.mean(diff))
    
    # 内部块差异（作为对照）
    internal_diffs = []
    for i in range(block_size, h - 2*block_size, block_size):
        for j in range(block_size, w - 2*block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            internal_diffs.append(np.mean(np.abs(np.diff(block.flatten()))))
    
    if len(horizontal_artifacts) == 0 or len(vertical_artifacts) == 0:
        return None
    
    horizontal_artifacts = np.array(horizontal_artifacts)
    vertical_artifacts = np.array(vertical_artifacts)
    internal_diffs = np.array(internal_diffs)
    
    # 块效应强度：边界跳变与内部差异的比值
    block_artifact_strength = (np.mean(horizontal_artifacts) + np.mean(vertical_artifacts)) / (np.mean(internal_diffs) + 1e-10)
    
    # 方差分析
    block_variance = np.std(horizontal_artifacts) + np.std(vertical_artifacts)
    
    features = [
        block_artifact_strength,
        block_variance,
        np.max(horizontal_artifacts) - np.min(horizontal_artifacts),
        np.max(vertical_artifacts) - np.min(vertical_artifacts)
    ]
    
    return np.array(features)


def detect_tampering_jpeg_block(image_path, threshold=0.5):
    """使用JPEG块效应检测篡改"""
    features = extract_jpeg_block_features(image_path)
    
    if features is None:
        return False, 0
    
    # 块效应不一致性
    score = features[1] + features[2] + features[3]
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)