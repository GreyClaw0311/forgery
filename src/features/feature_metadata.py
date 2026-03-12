"""
元数据分析
原理：检查图像EXIF信息和元数据的一致性
"""

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

def extract_metadata_features(image_path):
    """
    提取元数据特征

    Args:
        image_path: 图像路径

    Returns:
        feature_vector: 特征向量
    """
    features = []

    try:
        # 使用PIL读取EXIF信息
        img = Image.open(image_path)
        exif_data = img._getexif()

        if exif_data is None:
            # 无EXIF信息
            features = [0, 0, 0, 0, 0]
        else:
            # 解析EXIF标签
            exif = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                exif[tag] = value

            # 提取关键元数据
            has_make = 1 if 'Make' in exif else 0
            has_model = 1 if 'Model' in exif else 0
            has_software = 1 if 'Software' in exif else 0
            has_datetime = 1 if 'DateTime' in exif else 0

            # 检查元数据完整性
            important_tags = ['Make', 'Model', 'DateTime', 'ExifImageWidth',
                            'ExifImageHeight', 'Software']
            completeness = sum(1 for tag in important_tags if tag in exif) / len(important_tags)

            features = [has_make, has_model, has_software, has_datetime, completeness]

    except Exception as e:
        # 读取失败
        features = [0, 0, 0, 0, 0]

    # 检查文件创建时间和修改时间
    import os
    try:
        stat = os.stat(image_path)
        # 文件大小
        file_size = stat.st_size / (1024 * 1024)  # MB
        features.append(file_size)
    except:
        features.append(0)

    return np.array(features)


def detect_tampering_metadata(image_path, threshold=0.5):
    """
    使用元数据分析检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features = extract_metadata_features(image_path)

    # 元数据缺失或异常可能是篡改的迹象
    # 完整性分数越低，越可能是篡改
    completeness = features[4]

    # 计算篡改分数（元数据越少，分数越高）
    score = 1 - completeness

    is_tampered = score > threshold

    return is_tampered, score
