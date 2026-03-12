"""
边缘一致性检测
原理：检测图像边缘的连续性和一致性异常
"""

import cv2
import numpy as np

def extract_edge_features(image_path):
    """
    提取边缘特征

    Args:
        image_path: 图像路径

    Returns:
        feature_vector: 特征向量
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 计算边缘统计特征
    edge_ratio = np.sum(edges > 0) / edges.size

    # 分块分析边缘一致性
    h, w = edges.shape
    block_size = 32
    block_edge_ratios = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = edges[i:i+block_size, j:j+block_size]
            block_ratio = np.sum(block > 0) / block.size
            block_edge_ratios.append(block_ratio)

    # 边缘一致性（标准差越小越一致）
    edge_consistency = np.std(block_edge_ratios)

    # 使用Sobel计算边缘方向
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 边缘方向
    angles = np.arctan2(sobely, sobelx)

    # 方向直方图
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    hist = hist / np.sum(hist)

    # 方向熵（熵越大表示边缘方向越分散）
    hist_nonzero = hist[hist > 0]
    direction_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    features = [
        edge_ratio,
        edge_consistency,
        direction_entropy,
        np.mean(block_edge_ratios),
        np.max(block_edge_ratios) - np.min(block_edge_ratios)
    ]

    return np.array(features)


def detect_tampering_edge(image_path, threshold=0.3):
    """
    使用边缘一致性检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features = extract_edge_features(image_path)

    if features is None:
        return False, 0

    # 使用边缘不一致性和方向熵计算分数
    score = features[1] * 5 + features[2] / 3

    is_tampered = score > threshold

    return is_tampered, score
