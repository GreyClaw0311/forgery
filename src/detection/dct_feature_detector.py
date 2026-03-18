"""
DCT特征检测器 - 像素级DCT特征提取

基于DCT系数分析检测篡改区域
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from scipy import ndimage


class DCTFeatureDetector:
    """
    DCT特征检测器
    
    原理：
    - 篡改区域的DCT系数分布与原始区域不同
    - 通过滑动窗口提取DCT特征
    - 使用统计特征进行像素级检测
    """
    
    def __init__(self, 
                 window_size: int = 32,
                 stride: int = 16,
                 threshold: float = 0.5):
        """
        初始化DCT特征检测器
        
        Args:
            window_size: 滑动窗口大小
            stride: 滑动步长
            threshold: 检测阈值
        """
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold
        self.half = window_size // 2
    
    def extract_dct_features(self, patch: np.ndarray) -> np.ndarray:
        """
        从图像块提取DCT特征
        
        Args:
            patch: 图像块 (window_size x window_size)
            
        Returns:
            features: DCT特征向量
        """
        features = []
        
        # 转灰度
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        
        gray = gray.astype(np.float32)
        
        # DCT变换
        dct = cv2.dct(gray)
        
        # 低频特征 (左上8x8)
        dct_low = dct[:8, :8].flatten()
        features.append(np.mean(np.abs(dct_low)))
        features.append(np.std(dct_low))
        features.append(np.max(np.abs(dct_low)))
        features.append(np.percentile(np.abs(dct_low), 95))
        
        # 高频特征 (右下区域)
        dct_high = dct[8:, 8:].flatten()
        features.append(np.mean(np.abs(dct_high)))
        features.append(np.std(dct_high))
        features.append(np.percentile(np.abs(dct_high), 95))
        
        # 中频特征
        dct_mid = dct[8:16, 8:16].flatten()
        features.append(np.mean(np.abs(dct_mid)))
        features.append(np.std(dct_mid))
        
        # 频域能量比
        total_energy = np.sum(dct ** 2)
        low_energy = np.sum(dct[:8, :8] ** 2)
        features.append(low_energy / (total_energy + 1e-8))
        
        # AC系数特征 (排除DC)
        ac_coeffs = dct[1:, 1:].flatten()
        features.append(np.mean(np.abs(ac_coeffs)))
        features.append(np.std(ac_coeffs))
        features.append(np.max(np.abs(ac_coeffs)))
        
        return np.array(features)
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测篡改区域
        
        Args:
            image: 输入图像 (BGR)
            
        Returns:
            heatmap: 热力图 (0-1)
            mask: 二值掩码
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        
        # 滑动窗口检测
        for y in range(0, h - self.window_size, self.stride):
            for x in range(0, w - self.window_size, self.stride):
                patch = image[y:y+self.window_size, x:x+self.window_size]
                
                # 提取DCT特征
                features = self.extract_dct_features(patch)
                
                # 计算异常分数 (基于高频能量)
                score = features[4] + features[5]  # 高频均值+标准差
                
                # 累加到热力图
                heatmap[y:y+self.window_size, x:x+self.window_size] += score
                count[y:y+self.window_size, x:x+self.window_size] += 1
        
        # 归一化
        count[count == 0] = 1
        heatmap = heatmap / count
        
        # 归一化到0-1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # 高斯平滑
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        # 生成掩码
        mask = (heatmap > self.threshold).astype(np.uint8) * 255
        
        return heatmap, mask
    
    def get_mask(self, heatmap: np.ndarray, 
                 threshold: Optional[float] = None) -> np.ndarray:
        """
        从热力图生成二值掩码
        
        Args:
            heatmap: 热力图
            threshold: 阈值
            
        Returns:
            mask: 二值掩码
        """
        if threshold is None:
            threshold = self.threshold
        
        return (heatmap > threshold).astype(np.uint8) * 255


class DCTBlockFeatureDetector:
    """
    DCT块特征检测器
    
    分析8x8块的DCT特征不一致性
    """
    
    def __init__(self, block_size: int = 8):
        """
        初始化
        
        Args:
            block_size: 块大小
        """
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测篡改区域
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        gray = gray.astype(np.float32)
        
        # 块DCT特征
        block_h = h // self.block_size
        block_w = w // self.block_size
        
        feature_map = np.zeros((block_h, block_w))
        
        for i in range(block_h):
            for j in range(block_w):
                y = i * self.block_size
                x = j * self.block_size
                block = gray[y:y+self.block_size, x:x+self.block_size]
                
                # DCT
                dct = cv2.dct(block)
                
                # AC能量
                ac_energy = np.sum(dct[1:, 1:] ** 2)
                feature_map[i, j] = ac_energy
        
        # 归一化
        if feature_map.max() > 0:
            feature_map = feature_map / feature_map.max()
        
        # 上采样到原图大小
        heatmap = cv2.resize(feature_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 使用局部异常检测
        local_mean = cv2.blur(heatmap, (32, 32))
        anomaly = np.abs(heatmap - local_mean)
        
        if anomaly.max() > 0:
            anomaly = anomaly / anomaly.max()
        
        mask = (anomaly > 0.3).astype(np.uint8) * 255
        
        return anomaly, mask


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        detector = DCTFeatureDetector()
        heatmap, mask = detector.detect(image)
        
        cv2.imwrite("dct_heatmap.png", (heatmap * 255).astype(np.uint8))
        cv2.imwrite("dct_mask.png", mask)
        
        print(f"检测结果已保存")