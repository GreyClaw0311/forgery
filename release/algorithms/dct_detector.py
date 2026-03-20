"""
DCT块效应检测器

基于 Ye et al. (2007) 的方法
通过分析JPEG 8x8块效应的不一致性检测篡改区域
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class DCTBlockDetector:
    """
    DCT块效应检测器
    
    原理：
    - JPEG压缩产生8x8块效应
    - 篡改区域的块效应与原始区域不一致
    - 通过分析块边界的不连续性检测篡改
    """
    
    def __init__(self, block_size: int = 8, analysis_block: int = 32):
        """
        初始化DCT块效应检测器
        
        Args:
            block_size: JPEG块大小（通常为8）
            analysis_block: 分析块大小
        """
        self.block_size = block_size
        self.analysis_block = analysis_block
    
    def compute_block_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        计算块效应强度图
        
        Args:
            image: 输入图像
            
        Returns:
            artifact_map: 块效应强度图
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        artifact_map = np.zeros_like(gray, dtype=float)
        
        # 检测水平边界（每8像素）
        for i in range(self.block_size - 1, h - 1, self.block_size):
            diff = np.abs(gray[i, :].astype(float) - gray[i + 1, :].astype(float))
            artifact_map[i, :] = diff
        
        # 检测垂直边界（每8像素）
        for j in range(self.block_size - 1, w - 1, self.block_size):
            diff = np.abs(gray[:, j].astype(float) - gray[:, j + 1].astype(float))
            artifact_map[:, j] = np.maximum(artifact_map[:, j], diff)
        
        return artifact_map
    
    def analyze_block_consistency(self, image: np.ndarray) -> np.ndarray:
        """
        分析块效应一致性
        
        计算每个分析块内的块效应统计特征
        
        Args:
            image: 输入图像
            
        Returns:
            consistency_map: 一致性分数图
        """
        artifact_map = self.compute_block_artifacts(image)
        
        h, w = artifact_map.shape
        n_blocks_h = h // self.analysis_block
        n_blocks_w = w // self.analysis_block
        
        consistency = np.zeros((n_blocks_h, n_blocks_w))
        
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = artifact_map[i*self.analysis_block:(i+1)*self.analysis_block,
                                    j*self.analysis_block:(j+1)*self.analysis_block]
                
                # 计算块边界像素的统计特征
                boundary_values = []
                
                # 水平边界
                for bi in range(self.block_size - 1, self.analysis_block, self.block_size):
                    if bi < block.shape[0]:
                        boundary_values.extend(block[bi, :].tolist())
                
                # 垂直边界
                for bj in range(self.block_size - 1, self.analysis_block, self.block_size):
                    if bj < block.shape[1]:
                        boundary_values.extend(block[:, bj].tolist())
                
                if len(boundary_values) > 0:
                    consistency[i, j] = np.mean(boundary_values)
        
        return consistency
    
    def detect_artifact_anomalies(self, consistency: np.ndarray) -> np.ndarray:
        """
        检测块效应异常区域
        
        Args:
            consistency: 块效应一致性图
            
        Returns:
            anomaly_map: 异常分数图
        """
        h, w = consistency.shape
        anomaly = np.zeros_like(consistency)
        
        # 计算全局统计
        mean_val = np.mean(consistency)
        std_val = np.std(consistency)
        
        if std_val > 0:
            # 计算z-score
            anomaly = np.abs(consistency - mean_val) / std_val
        
        return anomaly
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        主检测函数
        
        Args:
            image: 输入图像
            
        Returns:
            heatmap: 归一化热力图 (0-1)
        """
        # 分析块效应一致性
        consistency = self.analyze_block_consistency(image)
        
        # 检测异常
        anomaly = self.detect_artifact_anomalies(consistency)
        
        # 上采样到原始尺寸
        h, w = image.shape[:2]
        heatmap = cv2.resize(anomaly, (w, h))
        
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
            _, mask = cv2.threshold(heatmap_uint8, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(heatmap_uint8, int(threshold * 255), 255,
                                    cv2.THRESH_BINARY)
        
        return mask
    
    def compute_dct_coefficients(self, image: np.ndarray) -> np.ndarray:
        """
        计算DCT系数（用于更精细的分析）
        
        Args:
            image: 输入图像
            
        Returns:
            dct_map: DCT系数统计图
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            gray = image.astype(float)
        
        h, w = gray.shape
        n_blocks_h = h // self.block_size
        n_blocks_w = w // self.block_size
        
        # 存储DC系数
        dc_map = np.zeros((n_blocks_h, n_blocks_w))
        
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = gray[i*self.block_size:(i+1)*self.block_size,
                            j*self.block_size:(j+1)*self.block_size]
                
                # DCT变换
                dct = cv2.dct(block)
                
                # DC系数
                dc_map[i, j] = dct[0, 0]
        
        return dc_map
    
    def detect_full(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        完整检测流程
        
        Args:
            image: 输入图像
            
        Returns:
            (heatmap, mask, artifact_map)
        """
        # 计算块效应
        artifact_map = self.compute_block_artifacts(image)
        
        # 分析一致性
        consistency = self.analyze_block_consistency(image)
        
        # 检测异常
        anomaly = self.detect_artifact_anomalies(consistency)
        
        # 上采样
        h, w = image.shape[:2]
        heatmap = cv2.resize(anomaly, (w, h))
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        # 生成掩码
        mask = self.get_mask(heatmap)
        
        return heatmap, mask, artifact_map


def detect_dct_artifacts(image_path: str,
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
    
    detector = DCTBlockDetector()
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
    # 创建一个简单的测试图像
    test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    # 添加一些块效应模拟JPEG压缩
    for i in range(0, 256, 8):
        test_image[i, :] = test_image[i, :] + 5
    
    detector = DCTBlockDetector()
    heatmap, mask, artifact_map = detector.detect_full(test_image)
    
    print(f"热力图范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    print(f"块效应图范围: [{artifact_map.min():.2f}, {artifact_map.max():.2f}]")