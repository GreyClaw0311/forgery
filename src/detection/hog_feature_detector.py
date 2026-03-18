"""
HOG特征检测器 - 像素级HOG特征提取

基于HOG (Histogram of Oriented Gradients) 特征检测篡改区域
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from scipy import ndimage


class HOGFeatureDetector:
    """
    HOG特征检测器
    
    原理：
    - 篡改区域的边缘方向分布与原始区域不同
    - 通过HOG特征分析边缘不一致性
    - 使用滑动窗口进行像素级检测
    """
    
    def __init__(self,
                 window_size: int = 32,
                 stride: int = 16,
                 cell_size: int = 8,
                 bins: int = 9,
                 threshold: float = 0.3):
        """
        初始化HOG特征检测器
        
        Args:
            window_size: 滑动窗口大小
            stride: 滑动步长
            cell_size: HOG cell大小
            bins: 方向直方图bin数
            threshold: 检测阈值
        """
        self.window_size = window_size
        self.stride = stride
        self.cell_size = cell_size
        self.bins = bins
        self.threshold = threshold
        self.half = window_size // 2
    
    def compute_gradients(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算图像梯度
        
        Args:
            gray: 灰度图像
            
        Returns:
            magnitude: 梯度幅值
            angle: 梯度方向
        """
        # Sobel梯度
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 180
        
        return magnitude, angle
    
    def extract_hog_features(self, patch: np.ndarray) -> np.ndarray:
        """
        从图像块提取HOG特征
        
        Args:
            patch: 图像块
            
        Returns:
            features: HOG特征向量
        """
        features = []
        
        # 转灰度
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        
        gray = gray.astype(np.float32)
        
        # 计算梯度
        magnitude, angle = self.compute_gradients(gray)
        
        # 全局梯度特征
        features.append(np.mean(magnitude))
        features.append(np.std(magnitude))
        features.append(np.max(magnitude))
        features.append(np.percentile(magnitude, 95))
        
        # 梯度方向分布
        hist, _ = np.histogram(angle, bins=self.bins, range=(0, 180))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        
        # 方向直方图统计
        features.append(np.max(hist))  # 主导方向强度
        features.append(np.std(hist))   # 方向分布均匀性
        features.append(-np.sum(hist * np.log(hist + 1e-8)))  # 熵
        
        # Cell-based HOG
        cells_x = self.window_size // self.cell_size
        cells_y = self.window_size // self.cell_size
        
        cell_hists = []
        for cy in range(cells_y):
            for cx in range(cells_x):
                y1 = cy * self.cell_size
                y2 = y1 + self.cell_size
                x1 = cx * self.cell_size
                x2 = x1 + self.cell_size
                
                cell_mag = magnitude[y1:y2, x1:x2]
                cell_angle = angle[y1:y2, x1:x2]
                
                cell_hist, _ = np.histogram(cell_angle, bins=self.bins, 
                                            range=(0, 180), weights=cell_mag)
                cell_hist = cell_hist.astype(np.float32)
                cell_hist = cell_hist / (cell_hist.sum() + 1e-8)
                cell_hists.append(cell_hist)
        
        # Cell一致性特征
        cell_hists = np.array(cell_hists)
        cell_std = np.std(cell_hists, axis=0)
        features.append(np.mean(cell_std))  # Cell间不一致性
        
        # 边缘密度
        edge_density = np.mean(magnitude > (np.mean(magnitude) + np.std(magnitude)))
        features.append(edge_density)
        
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
                
                # 提取HOG特征
                features = self.extract_hog_features(patch)
                
                # 计算异常分数 (基于方向分布不均匀性)
                score = features[5] + features[7]  # 方向熵 + Cell不一致性
                
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
        """
        if threshold is None:
            threshold = self.threshold
        
        return (heatmap > threshold).astype(np.uint8) * 255


class HOGAnomalyDetector:
    """
    基于HOG的异常检测器
    
    使用HOG特征的局部不一致性检测篡改
    """
    
    def __init__(self, block_size: int = 16):
        """
        初始化
        
        Args:
            block_size: 分析块大小
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
        
        # 计算全局梯度
        magnitude, angle = self._compute_gradients(gray)
        
        # 块级HOG特征
        block_h = h // self.block_size
        block_w = w // self.block_size
        
        hog_map = np.zeros((block_h, block_w, 9))  # 9个方向bin
        
        for i in range(block_h):
            for j in range(block_w):
                y = i * self.block_size
                x = j * self.block_size
                block_mag = magnitude[y:y+self.block_size, x:x+self.block_size]
                block_angle = angle[y:y+self.block_size, x:x+self.block_size]
                
                # 块HOG直方图
                hist, _ = np.histogram(block_angle, bins=9, range=(0, 180),
                                       weights=block_mag)
                hog_map[i, j] = hist
        
        # 归一化
        norms = np.linalg.norm(hog_map, axis=2, keepdims=True) + 1e-8
        hog_map = hog_map / norms
        
        # 计算局部不一致性
        anomaly_map = np.zeros((block_h, block_w))
        
        for i in range(1, block_h - 1):
            for j in range(1, block_w - 1):
                # 与邻域比较
                center = hog_map[i, j]
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        neighbors.append(hog_map[i+di, j+dj])
                
                neighbors = np.array(neighbors)
                mean_neighbor = np.mean(neighbors, axis=0)
                
                # 卡方距离
                diff = np.sum((center - mean_neighbor) ** 2 / (mean_neighbor + 1e-8))
                anomaly_map[i, j] = diff
        
        # 归一化
        if anomaly_map.max() > 0:
            anomaly_map = anomaly_map / anomaly_map.max()
        
        # 上采样
        heatmap = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 阈值化
        mask = (heatmap > 0.4).astype(np.uint8) * 255
        
        return heatmap, mask
    
    def _compute_gradients(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算梯度"""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 180
        
        return magnitude, angle


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        detector = HOGFeatureDetector()
        heatmap, mask = detector.detect(image)
        
        cv2.imwrite("hog_heatmap.png", (heatmap * 255).astype(np.uint8))
        cv2.imwrite("hog_mask.png", mask)
        
        print(f"检测结果已保存")