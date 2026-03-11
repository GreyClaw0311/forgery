"""
DCT域分析 - 离散余弦变换系数分析
原理：篡改区域的DCT系数分布会呈现异常
"""

import cv2
import numpy as np
from scipy.fftpack import dct, idct

def extract_dct_features(image_path, block_size=8):
    """
    提取DCT特征

    Args:
        image_path: 图像路径
        block_size: DCT块大小

    Returns:
        feature_vector: 特征向量
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 调整图像大小为block_size的倍数
    h, w = img.shape
    h = (h // block_size) * block_size
    w = (w // block_size) * block_size
    img = img[:h, :w]

    # 分块DCT
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size].astype(np.float32)
            block = block - 128  # 中心化
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            blocks.append(dct_block)

    blocks = np.array(blocks)

    # 提取DC和AC系数
    dc_coeffs = blocks[:, 0, 0]
    ac_coeffs = blocks[:, 1:, 1:].flatten()

    # 计算特征
    features = [
        np.mean(dc_coeffs),
        np.std(dc_coeffs),
        np.mean(np.abs(ac_coeffs)),
        np.std(ac_coeffs),
        # AC系数的零值比例
        np.sum(ac_coeffs == 0) / len(ac_coeffs),
        # 高频AC系数的分布
        np.std(blocks[:, -3:, -3:].flatten()),
    ]

    return np.array(features)


def detect_tampering_dct(image_path, threshold=0.5):
    """
    使用DCT检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features = extract_dct_features(image_path)

    if features is None:
        return False, 0

    # 基于AC系数标准差和零值比例计算分数
    score = features[3] + features[4] * 10

    # 归一化
    score = min(score / 20, 1.0)

    is_tampered = score > threshold

    return is_tampered, score
