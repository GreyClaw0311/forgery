"""
ELA (Error Level Analysis) 错误级别分析
原理：通过重新压缩图像并比较误差来检测篡改区域
"""

import cv2
import numpy as np

def extract_ela_features(image_path, quality=90):
    """
    提取ELA特征

    Args:
        image_path: 图像路径
        quality: JPEG压缩质量 (1-100)

    Returns:
        feature_vector: 特征向量
        ela_image: ELA差异图像
    """
    # 读取原始图像
    original = cv2.imread(image_path)
    if original is None:
        return None, None

    # 保存为JPEG格式（模拟二次压缩）
    temp_path = '/tmp/ela_temp.jpg'
    cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # 读取压缩后的图像
    compressed = cv2.imread(temp_path)

    # 计算差异
    diff = cv2.absdiff(original, compressed)

    # 放大差异
    ela_image = diff * 20

    # 转换为灰度
    gray_ela = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)

    # 提取特征
    mean_val = np.mean(gray_ela)
    std_val = np.std(gray_ela)
    max_val = np.max(gray_ela)

    # 计算高频区域的占比
    threshold = mean_val + 2 * std_val
    high_freq_ratio = np.sum(gray_ela > threshold) / gray_ela.size

    feature_vector = [mean_val, std_val, max_val, high_freq_ratio]

    return np.array(feature_vector), ela_image


def detect_tampering_ela(image_path, threshold=15):
    """
    使用ELA检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features, _ = extract_ela_features(image_path)

    if features is None:
        return False, 0

    # 使用均值和标准差计算篡改分数
    score = features[0] + features[1]  # mean + std

    is_tampered = score > threshold

    return is_tampered, score
