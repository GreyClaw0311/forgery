"""
分块DCT检测
原理：分块分析DCT系数的分布一致性
"""

import cv2
import numpy as np

def extract_block_dct_features(image_path):
    """提取分块DCT特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    h, w = gray.shape
    block_size = 8
    
    block_dc_values = []
    block_ac_energies = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            
            # DCT变换
            dct_block = cv2.dct(block)
            
            # DC系数
            dc = dct_block[0, 0]
            block_dc_values.append(dc)
            
            # AC能量
            ac_energy = np.sum(dct_block[1:, :]**2) + np.sum(dct_block[:, 1:]**2) - dct_block[0, 0]**2
            block_ac_energies.append(ac_energy)
    
    if len(block_dc_values) == 0:
        return None
    
    block_dc_values = np.array(block_dc_values)
    block_ac_energies = np.array(block_ac_energies)
    
    # 计算统计特征
    features = [
        np.std(block_dc_values),
        np.std(block_ac_energies),
        np.mean(block_ac_energies),
        np.max(block_dc_values) - np.min(block_dc_values)
    ]
    
    return np.array(features)


def detect_tampering_block_dct(image_path, threshold=10.0):
    """使用分块DCT检测篡改"""
    features = extract_block_dct_features(image_path)
    
    if features is None:
        return False, 0
    
    score = features[0] + features[1] / 1000
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)