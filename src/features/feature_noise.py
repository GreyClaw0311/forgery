"""
噪声一致性分析
原理：检测图像中噪声分布的不一致性
"""

import cv2
import numpy as np

def extract_noise_features(image_path):
    """
    提取噪声特征

    Args:
        image_path: 图像路径

    Returns:
        feature_vector: 特征向量
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = img.astype(np.float32)

    features = []

    # 对每个通道分析噪声
    for c in range(3):
        channel = img[:, :, c]

        # 使用高通滤波提取噪声
        # 均值滤波
        blurred = cv2.blur(channel, (5, 5))
        noise = channel - blurred

        # 计算噪声统计特征
        noise_std = np.std(noise)
        noise_mean = np.mean(np.abs(noise))

        # 分块分析噪声一致性
        h, w = noise.shape
        block_size = 32
        block_stds = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = noise[i:i+block_size, j:j+block_size]
                block_stds.append(np.std(block))

        # 噪声不一致性
        noise_inconsistency = np.std(block_stds)

        features.extend([noise_std, noise_mean, noise_inconsistency])

    return np.array(features)


def detect_tampering_noise(image_path, threshold=0.4):
    """
    使用噪声一致性检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features = extract_noise_features(image_path)

    if features is None:
        return False, 0

    # 使用噪声不一致性计算分数
    inconsistency = features[2::3]  # 每个通道的噪声不一致性
    score = np.mean(inconsistency)

    # 归一化
    score = min(score / 10, 1.0)

    is_tampered = score > threshold

    return is_tampered, score
