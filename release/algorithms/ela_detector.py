"""
ELA (Error Level Analysis) 热力图检测器

基于 Krawetz (2007) 的方法，通过JPEG压缩误差差异检测篡改区域
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class ELADetector:
    """
    ELA热力图检测器
    
    原理：
    - JPEG压缩是有损压缩，不同区域如果经历过不同的压缩历史，
      在重新压缩时会表现出不同的误差特征
    - 篡改区域通常经过二次压缩，误差特征与原始区域不同
    """
    
    def __init__(self, quality_levels: List[int] = None):
        """
        初始化ELA检测器
        
        Args:
            quality_levels: JPEG压缩质量级别列表，默认使用多个级别提高鲁棒性
        """
        self.quality_levels = quality_levels or [75, 85, 95]
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        生成ELA热力图
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            heatmap: 归一化热力图 (0-1)，值越大表示篡改可能性越高
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        heatmaps = []
        
        for quality in self.quality_levels:
            # 重新JPEG压缩
            encoded, buffer = cv2.imencode('.jpg', image, 
                                          [cv2.IMWRITE_JPEG_QUALITY, quality])
            reconstructed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
            # 计算像素级差异
            diff = np.abs(image.astype(float) - reconstructed.astype(float))
            
            # 取RGB三通道最大值
            heatmap = np.max(diff, axis=2)
            heatmaps.append(heatmap)
        
        # 多质量级别融合
        final_heatmap = np.mean(heatmaps, axis=0)
        
        # 高斯模糊平滑
        final_heatmap = cv2.GaussianBlur(final_heatmap, (5, 5), 0)
        
        # 归一化到0-1
        if final_heatmap.max() > 0:
            final_heatmap = final_heatmap / final_heatmap.max()
        
        return final_heatmap
    
    def get_mask(self, heatmap: np.ndarray, 
                 threshold: Optional[float] = None,
                 method: str = 'otsu') -> np.ndarray:
        """
        从热力图生成二值掩码
        
        Args:
            heatmap: 热力图 (0-1)
            threshold: 阈值，如果为None则使用自适应方法
            method: 阈值方法 ('otsu', 'adaptive', 'fixed')
            
        Returns:
            mask: 二值掩码 (0=正常, 255=篡改)
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        if method == 'otsu':
            # Otsu自适应阈值
            _, mask = cv2.threshold(heatmap_uint8, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            # 自适应阈值
            mask = cv2.adaptiveThreshold(heatmap_uint8, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        else:
            # 固定阈值
            if threshold is None:
                threshold = 0.5
            _, mask = cv2.threshold(heatmap_uint8, int(threshold * 255), 255,
                                    cv2.THRESH_BINARY)
        
        return mask
    
    def detect_with_mask(self, image: np.ndarray,
                         threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测并返回热力图和掩码
        
        Args:
            image: 输入图像
            threshold: 阈值
            
        Returns:
            (heatmap, mask)
        """
        heatmap = self.detect(image)
        mask = self.get_mask(heatmap, threshold)
        return heatmap, mask
    
    def analyze_block_consistency(self, image: np.ndarray,
                                  block_size: int = 32) -> np.ndarray:
        """
        分析块级ELA一致性
        
        通过比较每个块的ELA值与周围块的差异，检测异常块
        
        Args:
            image: 输入图像
            block_size: 块大小
            
        Returns:
            inconsistency_map: 不一致性图
        """
        heatmap = self.detect(image)
        h, w = heatmap.shape
        
        # 计算每个块的平均ELA值
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size
        
        block_ela = np.zeros((n_blocks_h, n_blocks_w))
        
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = heatmap[i*block_size:(i+1)*block_size,
                               j*block_size:(j+1)*block_size]
                block_ela[i, j] = np.mean(block)
        
        # 计算每个块与周围块的差异
        inconsistency = np.zeros_like(block_ela)
        
        for i in range(1, n_blocks_h - 1):
            for j in range(1, n_blocks_w - 1):
                # 8邻域
                neighbors = block_ela[i-1:i+2, j-1:j+2].flatten()
                neighbors = np.delete(neighbors, 4)  # 排除中心
                
                mean = np.mean(neighbors)
                std = np.std(neighbors)
                
                if std > 0:
                    inconsistency[i, j] = abs(block_ela[i, j] - mean) / std
        
        # 上采样到原始尺寸
        inconsistency_full = cv2.resize(inconsistency, (w, h))
        
        return inconsistency_full


def detect_ela(image_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    便捷函数：从文件路径检测
    
    Args:
        image_path: 图像路径
        output_path: 输出路径（可选）
        
    Returns:
        heatmap: 热力图
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    detector = ELADetector()
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
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        heatmap = detect_ela(image_path, "ela_output.png")
        print(f"ELA热力图已保存到 ela_output.png")
        print(f"热力图范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    else:
        # 使用随机图像测试
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        detector = ELADetector()
        heatmap = detector.detect(test_image)
        mask = detector.get_mask(heatmap)
        
        print(f"热力图形状: {heatmap.shape}")
        print(f"掩码形状: {mask.shape}")
        print(f"篡改像素比例: {np.sum(mask > 0) / mask.size * 100:.2f}%")