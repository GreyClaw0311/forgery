"""
直方图分析
原理：分析颜色直方图分布的一致性
"""

import cv2
import numpy as np

def extract_histogram_features(image_path):
    """
    提取直方图特征

    Args:
        image_path: 图像路径

    Returns:
        feature_vector: 特征向量
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return None

    features = []

    # 对每个通道计算直方图
    for c in range(3):
        channel = img[:, :, c]
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # 归一化

        # 直方图统计特征
        mean = np.sum(np.arange(256) * hist)
        variance = np.sum(((np.arange(256) - mean) ** 2) * hist)

        # 峰值和谷值
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)

        # 偏度和峰度
        if variance > 0:
            skewness = np.sum(((np.arange(256) - mean) ** 3) * hist) / (variance ** 1.5)
            kurtosis = np.sum(((np.arange(256) - mean) ** 4) * hist) / (variance ** 2) - 3
        else:
            skewness = 0
            kurtosis = 0

        features.extend([mean, np.sqrt(variance), skewness, kurtosis, len(peaks)])

    # 分块直方图一致性
    h, w = img.shape[:2]
    block_size = 64
    block_hists = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            hist = cv2.calcHist([block], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-10)
            block_hists.append(hist)

    # 计算块间直方图差异
    if len(block_hists) > 1:
        block_hists = np.array(block_hists)
        hist_consistency = np.mean(np.std(block_hists, axis=0))
    else:
        hist_consistency = 0

    features.append(hist_consistency)

    return np.array(features)


def detect_tampering_histogram(image_path, threshold=0.3):
    """
    使用直方图分析检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features = extract_histogram_features(image_path)

    if features is None:
        return False, 0

    # 使用直方图一致性计算分数
    score = features[-1] * 10

    is_tampered = score > threshold

    return is_tampered, score
