"""
CFA (Color Filter Array) 插值检测
原理：检测Bayer模式插值痕迹的不一致性
"""

import cv2
import numpy as np

def extract_cfa_features(image_path):
    """
    提取CFA插值特征

    Args:
        image_path: 图像路径

    Returns:
        feature_vector: 特征向量
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 转换为浮点型
    img = img.astype(np.float32)

    features = []

    # 对每个通道分析插值痕迹
    for c in range(3):
        channel = img[:, :, c]
        h, w = channel.shape

        # 计算水平和垂直方向的差分
        diff_h = np.abs(channel[:, 1:] - channel[:, :-1])
        diff_v = np.abs(channel[1:, :] - channel[:-1, :])

        # 计算周期性模式（Bayer模式特征）
        # 水平方向周期性
        h_period = np.mean(diff_h[:, ::2]) - np.mean(diff_h[:, 1::2])
        # 垂直方向周期性
        v_period = np.mean(diff_v[::2, :]) - np.mean(diff_v[1::2, :])

        features.extend([h_period, v_period])

    # 计算通道间的相关性
    corr_rg = np.corrcoef(img[:, :, 2].flatten(), img[:, :, 1].flatten())[0, 1]
    corr_rb = np.corrcoef(img[:, :, 2].flatten(), img[:, :, 0].flatten())[0, 1]
    corr_gb = np.corrcoef(img[:, :, 1].flatten(), img[:, :, 0].flatten())[0, 1]

    features.extend([corr_rg, corr_rb, corr_gb])

    return np.array(features)


def detect_tampering_cfa(image_path, threshold=0.3):
    """
    使用CFA检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features = extract_cfa_features(image_path)

    if features is None:
        return False, 0

    # 计算周期性差异的标准差作为篡改分数
    periodicity = features[:6]
    score = np.std(periodicity)

    # 归一化
    score = min(score / 10, 1.0)

    is_tampered = score > threshold

    return is_tampered, score
