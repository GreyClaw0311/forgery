"""
JPEG Ghost检测
原理：检测JPEG压缩伪影的异常区域
"""

import cv2
import numpy as np

def extract_jpeg_ghost_features(image_path):
    """提取JPEG Ghost特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    # 不同质量重新压缩
    ghost_maps = []
    
    for quality in [50, 60, 70, 80, 90]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE).astype(float)
        
        # 计算差异
        diff = np.abs(gray - decoded)
        ghost_maps.append(diff)
    
    # 分块分析
    h, w = gray.shape
    block_size = 32
    
    block_ghost_scores = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_scores = []
            for ghost in ghost_maps:
                block = ghost[i:i+block_size, j:j+block_size]
                block_scores.append(np.mean(block))
            block_ghost_scores.append(np.std(block_scores))
    
    if len(block_ghost_scores) == 0:
        return None
    
    block_ghost_scores = np.array(block_ghost_scores)
    
    features = [
        np.mean(block_ghost_scores),
        np.std(block_ghost_scores),
        np.max(block_ghost_scores)
    ]
    
    return np.array(features)


def detect_tampering_jpeg_ghost(image_path, threshold=2.0):
    """使用JPEG Ghost检测篡改"""
    features = extract_jpeg_ghost_features(image_path)
    
    if features is None:
        return False, 0
    
    score = features[0] + features[1]
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)