"""
噪声一致性检测器

基于 Mahdian & Saic (2009) 的方法
通过检测局部噪声水平的不一致性来识别篡改区域
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from scipy import ndimage


class NoiseConsistencyDetector:
    """
    噪声一致性检测器
    
    原理：
    - 每个相机传感器都有固定的噪声模式
    - 篡改区域的噪声模式会与原始区域不一致
    - 通过估计局部噪声水平，检测不一致区域
    """
    
    def __init__(self, block_size: int = 32, neighbor_radius: int = 1):
        """
        初始化噪声一致性检测器
        
        Args:
            block_size: 分块大小
            neighbor_radius: 邻域半径
        """
        self.block_size = block_size
        self.neighbor_radius = neighbor_radius
    
    def estimate_noise_level(self, image: np.ndarray) -> np.ndarray:
        """
        估计图像的局部噪声水平
        
        使用Mihcak方法：通过高通滤波估计噪声方差
        
        Args:
            image: 输入图像
            
        Returns:
            noise_map: 噪声水平图
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 方法1：使用Laplacian算子估计噪声
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # 分块计算噪声方差
        h, w = gray.shape
        n_blocks_h = h // self.block_size
        n_blocks_w = w // self.block_size
        
        noise_map = np.zeros((n_blocks_h, n_blocks_w))
        
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = laplacian[i*self.block_size:(i+1)*self.block_size,
                                 j*self.block_size:(j+1)*self.block_size]
                # 噪声方差估计
                noise_map[i, j] = np.var(block)
        
        return noise_map
    
    def estimate_noise_wavelet(self, image: np.ndarray) -> np.ndarray:
        """
        使用小波分解估计噪声（更精确的方法）
        
        Args:
            image: 输入图像
            
        Returns:
            noise_map: 噪声水平图
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            gray = image.astype(float)
        
        # 使用高通滤波模拟小波高频分量
        # Sobel算子
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 梯度幅值
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 分块计算
        h, w = gray.shape
        n_blocks_h = h // self.block_size
        n_blocks_w = w // self.block_size
        
        noise_map = np.zeros((n_blocks_h, n_blocks_w))
        
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = gradient[i*self.block_size:(i+1)*self.block_size,
                                j*self.block_size:(j+1)*self.block_size]
                noise_map[i, j] = np.std(block)
        
        return noise_map
    
    def detect_inconsistency(self, noise_map: np.ndarray) -> np.ndarray:
        """
        检测噪声不一致区域
        
        计算每个块与周围块的噪声差异
        
        Args:
            noise_map: 噪声水平图 (block_size缩放后)
            
        Returns:
            inconsistency: 不一致性分数图
        """
        h, w = noise_map.shape
        inconsistency = np.zeros_like(noise_map)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # 获取邻域
                neighbors = []
                for di in range(-self.neighbor_radius, self.neighbor_radius + 1):
                    for dj in range(-self.neighbor_radius, self.neighbor_radius + 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            neighbors.append(noise_map[ni, nj])
                
                if len(neighbors) > 0:
                    mean = np.mean(neighbors)
                    std = np.std(neighbors)
                    
                    if std > 0:
                        # z-score
                        inconsistency[i, j] = abs(noise_map[i, j] - mean) / std
        
        return inconsistency
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        主检测函数
        
        Args:
            image: 输入图像
            
        Returns:
            heatmap: 归一化热力图 (0-1)
        """
        # 估计噪声
        noise_map = self.estimate_noise_level(image)
        
        # 检测不一致性
        inconsistency = self.detect_inconsistency(noise_map)
        
        # 上采样到原始尺寸
        h, w = image.shape[:2]
        heatmap = cv2.resize(inconsistency, (w, h))
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # 平滑
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        return heatmap
    
    def get_mask(self, heatmap: np.ndarray,
                 threshold: Optional[float] = None) -> np.ndarray:
        """
        生成二值掩码
        
        Args:
            heatmap: 热力图
            threshold: 阈值
            
        Returns:
            mask: 二值掩码
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        if threshold is None:
            # Otsu自适应阈值
            _, mask = cv2.threshold(heatmap_uint8, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(heatmap_uint8, int(threshold * 255), 255,
                                    cv2.THRESH_BINARY)
        
        return mask
    
    def detect_full(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        完整检测流程
        
        Args:
            image: 输入图像
            
        Returns:
            (heatmap, mask, noise_map)
        """
        # 估计噪声
        noise_map = self.estimate_noise_level(image)
        
        # 检测不一致性
        inconsistency = self.detect_inconsistency(noise_map)
        
        # 上采样
        h, w = image.shape[:2]
        heatmap = cv2.resize(inconsistency, (w, h))
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        # 生成掩码
        mask = self.get_mask(heatmap)
        
        # 噪声图上采样
        noise_map_full = cv2.resize(noise_map, (w, h))
        
        return heatmap, mask, noise_map_full


def detect_noise_inconsistency(image_path: str, 
                                output_path: Optional[str] = None) -> np.ndarray:
    """
    便捷函数：从文件路径检测
    
    Args:
        image_path: 图像路径
        output_path: 输出路径
        
    Returns:
        heatmap: 热力图
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    detector = NoiseConsistencyDetector()
    heatmap = detector.detect(image)
    
    if output_path:
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cv2.imwrite(output_path, heatmap_colored)
    
    return heatmap


if __name__ == "__main__":
    # 测试
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 添加一些噪声
    noise = np.random.normal(0, 25, test_image.shape).astype(np.int16)
    noisy_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    detector = NoiseConsistencyDetector(block_size=32)
    heatmap, mask, noise_map = detector.detect_full(noisy_image)
    
    print(f"热力图范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"掩码篡改比例: {np.sum(mask > 0) / mask.size * 100:.2f}%")