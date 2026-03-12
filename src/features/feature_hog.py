"""
HOG特征 - 方向梯度直方图
原理：分析图像局部梯度方向分布的一致性
"""

import cv2
import numpy as np

def extract_hog_features(image_path):
    """提取HOG特征"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算梯度
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180
    
    # 分块计算HOG
    h, w = gray.shape
    cell_size = 16
    block_size = 2
    nbins = 9
    
    cells_y = h // cell_size
    cells_x = w // cell_size
    
    hog_cells = []
    
    for i in range(cells_y):
        for j in range(cells_x):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ang = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            hist = np.zeros(nbins)
            for y in range(cell_size):
                for x in range(cell_size):
                    bin_idx = int(cell_ang[y, x] / 20) % nbins
                    hist[bin_idx] += cell_mag[y, x]
            
            hog_cells.append(hist)
    
    if len(hog_cells) == 0:
        return None
    
    hog_cells = np.array(hog_cells)
    
    # 计算特征统计
    mean_hist = np.mean(hog_cells, axis=0)
    std_hist = np.std(hog_cells, axis=0)
    
    # 特征向量
    features = np.concatenate([mean_hist, std_hist])
    
    return features


def detect_tampering_hog(image_path, threshold=0.15):
    """使用HOG特征检测篡改"""
    features = extract_hog_features(image_path)
    
    if features is None:
        return False, 0
    
    # 使用标准差计算不一致性
    std_features = features[9:]  # 后9个是标准差
    score = np.mean(std_features)
    
    is_tampered = score > threshold
    
    return is_tampered, float(score)