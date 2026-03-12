"""
FFT频域分析
原理：分析图像频谱分布的异常
"""

import cv2
import numpy as np

def extract_fft_features(image_path):
    """
    提取FFT频域特征

    Args:
        image_path: 图像路径

    Returns:
        feature_vector: 特征向量
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 调整图像大小为2的幂次
    h, w = img.shape
    h = 2 ** int(np.log2(h))
    w = 2 ** int(np.log2(w))
    img = img[:h, :w]

    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # 对数变换增强显示
    magnitude_log = np.log(magnitude + 1)

    # 计算频谱特征
    # 中心区域（低频）
    center_size = min(h, w) // 8
    center = magnitude_log[h//2-center_size:h//2+center_size,
                           w//2-center_size:w//2+center_size]
    low_freq_energy = np.mean(center)

    # 边缘区域（高频）
    edge_size = min(h, w) // 4
    high_freq_mask = np.ones_like(magnitude_log, dtype=bool)
    high_freq_mask[h//2-edge_size:h//2+edge_size,
                   w//2-edge_size:w//2+edge_size] = False
    high_freq_energy = np.mean(magnitude_log[high_freq_mask])

    # 频谱能量比
    energy_ratio = low_freq_energy / (high_freq_energy + 1e-10)

    # 分块频谱一致性
    block_size = 32
    block_energies = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = img[i:i+block_size, j:j+block_size]
            f_block = np.fft.fft2(block)
            magnitude_block = np.abs(f_block)
            block_energies.append(np.mean(np.log(magnitude_block + 1)))

    freq_consistency = np.std(block_energies)

    # 频谱方向性
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    radial_energies = []

    for angle in angles:
        # 沿着特定角度提取频谱能量
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 3
        x = center_x + int(radius * np.cos(angle))
        y = center_y + int(radius * np.sin(angle))
        if 0 <= x < w and 0 <= y < h:
            radial_energies.append(magnitude_log[y, x])

    directionality = np.std(radial_energies) if radial_energies else 0

    features = [
        low_freq_energy,
        high_freq_energy,
        energy_ratio,
        freq_consistency,
        directionality
    ]

    return np.array(features)


def detect_tampering_fft(image_path, threshold=0.4):
    """
    使用FFT频域分析检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features = extract_fft_features(image_path)

    if features is None:
        return False, 0

    # 使用频谱一致性计算分数
    score = features[3] / 5

    is_tampered = score > threshold

    return is_tampered, score
