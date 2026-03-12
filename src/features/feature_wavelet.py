"""
小波分析检测
原理：使用小波变换分析图像的多尺度特征
"""

import cv2
import numpy as np
import pywt

def extract_wavelet_features(image_path):
    """提取小波特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    
    # 小波分解
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    
    # 计算各分量统计
    features = [
        np.mean(np.abs(cA)),
        np.std(cA),
        np.mean(np.abs(cH)),
        np.std(cH),
        np.mean(np.abs(cV)),
        np.std(cV),
        np.mean(np.abs(cD)),
        np.std(cD)
    ]
    
    # 分块分析
    block_size = 32
    h, w = cH.shape
    block_energies = []
    
    for i in range(0, h - block_size, block_size // 2):
        for j in range(0, w - block_size, block_size // 2):
            block_h = cH[i:i+block_size//2, j:j+block_size//2]
            block_v = cV[i:i+block_size//2, j:j+block_size//2]
            block_d = cD[i:i+block_size//2, j:j+block_size//2]
            
            energy = np.mean(block_h**2) + np.mean(block_v**2) + np.mean(block_d**2)
            block_energies.append(energy)
    
    if len(block_energies) > 0:
        features.append(np.std(block_energies))
    
    return np.array(features)


def detect_tampering_wavelet(image_path, threshold=0.3):
    """使用小波分析检测篡改"""
    features = extract_wavelet_features(image_path)
    
    if features is None:
        return False, 0
    
    # 使用高频分量的不一致性
    if len(features) > 8:
        score = features[8]
    else:
        score = np.mean(features[2:8])
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)