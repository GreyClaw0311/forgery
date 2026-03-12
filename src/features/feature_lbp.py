"""
LBP (Local Binary Pattern) 纹理特征
原理：分析局部纹理模式的一致性
"""

import cv2
import numpy as np

def extract_lbp_features(image_path, radius=1, n_points=8):
    """
    提取LBP纹理特征

    Args:
        image_path: 图像路径
        radius: LBP半径
        n_points: LBP采样点数

    Returns:
        feature_vector: 特征向量
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape

    # 实现简单的LBP
    def compute_lbp(img, radius, n_points):
        lbp = np.zeros_like(img)
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = img[i, j]
                binary = 0
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if img[x, y] >= center:
                        binary |= (1 << p)
                lbp[i, j] = binary
        return lbp

    # 计算LBP
    lbp = compute_lbp(img.astype(np.float32), radius, n_points)

    # 计算LBP直方图
    hist, _ = np.histogram(lbp[radius:-radius, radius:-radius],
                           bins=2**n_points, range=(0, 2**n_points))
    hist = hist.astype(np.float32)
    hist /= hist.sum()  # 归一化

    # 计算纹理特征
    # 均匀模式
    uniform_patterns = 0
    for i in range(2**n_points):
        binary = bin(i)[2:].zfill(n_points)
        transitions = sum(1 for j in range(len(binary)-1)
                         if binary[j] != binary[j+1])
        if transitions <= 2:
            uniform_patterns += hist[i]

    # 熵
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

    # 分块分析纹理一致性
    block_size = 32
    block_entropies = []

    for i in range(radius, h - block_size - radius, block_size):
        for j in range(radius, w - block_size - radius, block_size):
            block = lbp[i:i+block_size, j:j+block_size]
            block_hist, _ = np.histogram(block, bins=2**n_points, range=(0, 2**n_points))
            block_hist = block_hist.astype(np.float32)
            block_hist /= block_hist.sum() + 1e-10
            block_hist_nz = block_hist[block_hist > 0]
            if len(block_hist_nz) > 0:
                block_entropies.append(-np.sum(block_hist_nz * np.log2(block_hist_nz)))

    texture_consistency = np.std(block_entropies) if block_entropies else 0

    features = [
        uniform_patterns,
        entropy,
        texture_consistency
    ]

    return np.array(features)


def detect_tampering_lbp(image_path, threshold=0.5):
    """
    使用LBP纹理特征检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features = extract_lbp_features(image_path)

    if features is None:
        return False, 0

    # 使用纹理一致性计算分数
    score = features[2] / 5

    is_tampered = score > threshold

    return is_tampered, score
