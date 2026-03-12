"""
SIFT特征匹配
原理：检测关键点匹配异常，识别复制粘贴篡改
"""

import cv2
import numpy as np

def extract_sift_features(image_path):
    """
    提取SIFT特征

    Args:
        image_path: 图像路径

    Returns:
        feature_vector: 特征向量
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和描述符
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 2:
        return np.zeros(6)

    # 关键点统计
    num_keypoints = len(keypoints)

    # 关键点响应强度
    responses = [kp.response for kp in keypoints]
    mean_response = np.mean(responses)
    std_response = np.std(responses)

    # 关键点大小分布
    sizes = [kp.size for kp in keypoints]
    mean_size = np.mean(sizes)
    std_size = np.std(sizes)

    # 检测可能的复制粘贴区域（自匹配）
    if num_keypoints >= 4 and descriptors.shape[0] >= 4:
        # 使用FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors, descriptors, k=2)

        # 统计自匹配数量（排除自身匹配）
        self_match_count = 0
        for i, (m, n) in enumerate(matches):
            # 排除自身匹配和距离太近的关键点
            if m.distance < 0.7 * n.distance:
                kp1 = keypoints[m.queryIdx]
                kp2 = keypoints[m.trainIdx]
                dist = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + (kp1.pt[1] - kp2.pt[1])**2)
                if dist > 30:  # 排除距离太近的点
                    self_match_count += 1

        self_match_ratio = self_match_count / num_keypoints
    else:
        self_match_ratio = 0

    features = [
        num_keypoints,
        mean_response,
        std_response,
        mean_size,
        std_size,
        self_match_ratio
    ]

    return np.array(features)


def detect_tampering_sift(image_path, threshold=0.1):
    """
    使用SIFT特征检测篡改

    Args:
        image_path: 图像路径
        threshold: 篡改判定阈值

    Returns:
        is_tampered: 是否篡改
        score: 篡改分数
    """
    features = extract_sift_features(image_path)

    if features is None or np.all(features == 0):
        return False, 0

    # 使用自匹配比例作为篡改分数
    score = features[5]

    is_tampered = score > threshold

    return is_tampered, score
