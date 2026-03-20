"""
后处理模块

对检测结果进行后处理优化：
- 形态学操作
- 连通域分析
- 边界平滑
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class PostProcessor:
    """
    掩码后处理器
    """
    
    def __init__(self, 
                 min_area: int = 100,
                 kernel_size: int = 5):
        """
        初始化后处理器
        
        Args:
            min_area: 最小连通域面积
            kernel_size: 形态学核大小
        """
        self.min_area = min_area
        self.kernel_size = kernel_size
    
    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        完整后处理流程
        
        Args:
            mask: 原始掩码
            
        Returns:
            processed_mask: 处理后的掩码
        """
        # 1. 形态学闭运算（填充小孔）
        mask = self.morphological_close(mask)
        
        # 2. 形态学开运算（去除噪点）
        mask = self.morphological_open(mask)
        
        # 3. 移除小连通域
        mask = self.remove_small_regions(mask, self.min_area)
        
        # 4. 边界平滑
        mask = self.smooth_boundary(mask)
        
        return mask
    
    def morphological_open(self, mask: np.ndarray,
                           kernel_size: Optional[int] = None) -> np.ndarray:
        """
        形态学开运算
        
        去除小的噪点和孤立点
        
        Args:
            mask: 输入掩码
            kernel_size: 核大小
            
        Returns:
            处理后的掩码
        """
        if kernel_size is None:
            kernel_size = self.kernel_size
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (kernel_size, kernel_size))
        result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return result
    
    def morphological_close(self, mask: np.ndarray,
                            kernel_size: Optional[int] = None) -> np.ndarray:
        """
        形态学闭运算
        
        填充小孔和连接断开的区域
        
        Args:
            mask: 输入掩码
            kernel_size: 核大小
            
        Returns:
            处理后的掩码
        """
        if kernel_size is None:
            kernel_size = self.kernel_size + 2  # 闭运算用稍大的核
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return result
    
    def remove_small_regions(self, mask: np.ndarray,
                              min_area: Optional[int] = None) -> np.ndarray:
        """
        移除小连通域
        
        Args:
            mask: 输入掩码
            min_area: 最小面积阈值
            
        Returns:
            处理后的掩码
        """
        if min_area is None:
            min_area = self.min_area
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        # 创建新掩码
        result = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # 跳过背景（0）
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                result[labels == i] = 255
        
        return result
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        填充孔洞
        
        Args:
            mask: 输入掩码
            
        Returns:
            填充后的掩码
        """
        # 使用floodFill填充孔洞
        h, w = mask.shape[:2]
        
        # 创建一个比原图稍大的图像
        flood_fill = np.zeros((h + 2, w + 2), dtype=np.uint8)
        flood_fill[1:-1, 1:-1] = mask.copy()
        
        # 从(0,0)开始填充
        cv2.floodFill(flood_fill, None, (0, 0), 255)
        
        # 反转填充结果
        flood_fill_inv = cv2.bitwise_not(flood_fill)
        
        # 裁剪回原始尺寸
        result = flood_fill_inv[1:-1, 1:-1]
        
        # 与原始掩码合并
        result = cv2.bitwise_or(mask, result)
        
        return result
    
    def smooth_boundary(self, mask: np.ndarray,
                        blur_size: int = 5) -> np.ndarray:
        """
        边界平滑
        
        Args:
            mask: 输入掩码
            blur_size: 模糊核大小
            
        Returns:
            平滑后的掩码
        """
        # 高斯模糊
        blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        # 重新二值化
        _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        return result
    
    def dilate(self, mask: np.ndarray,
               iterations: int = 1,
               kernel_size: Optional[int] = None) -> np.ndarray:
        """
        膨胀操作
        
        扩大篡改区域
        
        Args:
            mask: 输入掩码
            iterations: 迭代次数
            kernel_size: 核大小
            
        Returns:
            膨胀后的掩码
        """
        if kernel_size is None:
            kernel_size = self.kernel_size
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        result = cv2.dilate(mask, kernel, iterations=iterations)
        return result
    
    def erode(self, mask: np.ndarray,
              iterations: int = 1,
              kernel_size: Optional[int] = None) -> np.ndarray:
        """
        腐蚀操作
        
        缩小篡改区域
        
        Args:
            mask: 输入掩码
            iterations: 迭代次数
            kernel_size: 核大小
            
        Returns:
            腐蚀后的掩码
        """
        if kernel_size is None:
            kernel_size = self.kernel_size
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        result = cv2.erode(mask, kernel, iterations=iterations)
        return result
    
    def refine_with_contours(self, mask: np.ndarray,
                             simplify_epsilon: float = 2.0) -> np.ndarray:
        """
        基于轮廓的精细化
        
        Args:
            mask: 输入掩码
            simplify_epsilon: 轮廓简化参数
            
        Returns:
            精细化后的掩码
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建新掩码
        result = np.zeros_like(mask)
        
        for contour in contours:
            # 简化轮廓
            epsilon = simplify_epsilon
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 过滤太小的轮廓
            if cv2.contourArea(approx) >= self.min_area:
                cv2.drawContours(result, [approx], -1, 255, -1)
        
        return result
    
    def keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        只保留最大连通域
        
        Args:
            mask: 输入掩码
            
        Returns:
            只包含最大连通域的掩码
        """
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        if num_labels <= 1:
            return mask
        
        # 找最大连通域（跳过背景）
        largest_label = 1
        largest_area = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_label = i
        
        # 创建只包含最大连通域的掩码
        result = np.zeros_like(mask)
        result[labels == largest_label] = 255
        
        return result


class AdaptivePostProcessor(PostProcessor):
    """
    自适应后处理器
    
    根据图像特征自动调整参数
    """
    
    def process_adaptive(self, mask: np.ndarray,
                          image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        自适应后处理
        
        根据掩码特征自动调整参数
        
        Args:
            mask: 输入掩码
            image: 原图（可选，用于参考）
            
        Returns:
            处理后的掩码
        """
        # 计算掩码覆盖比例
        coverage = np.sum(mask > 0) / mask.size
        
        # 根据覆盖比例调整参数
        if coverage > 0.5:
            # 覆盖太多，需要收缩
            min_area = 500
            kernel_size = 7
            result = self.morphological_erode(mask, iterations=1)
        elif coverage > 0.3:
            # 中等覆盖，标准处理
            min_area = 200
            kernel_size = 5
            result = mask.copy()
        else:
            # 覆盖较少，需要扩张
            min_area = 50
            kernel_size = 3
            result = self.morphological_dilate(mask, iterations=1)
        
        # 标准后处理
        result = self.morphological_close(result, kernel_size)
        result = self.morphological_open(result, kernel_size - 2)
        result = self.remove_small_regions(result, min_area)
        result = self.smooth_boundary(result)
        
        return result
    
    def morphological_dilate(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """膨胀"""
        return self.dilate(mask, iterations)
    
    def morphological_erode(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """腐蚀"""
        return self.erode(mask, iterations)


def post_process(mask: np.ndarray,
                 min_area: int = 100,
                 smooth: bool = True) -> np.ndarray:
    """
    便捷函数：后处理
    
    Args:
        mask: 输入掩码
        min_area: 最小连通域面积
        smooth: 是否平滑边界
        
    Returns:
        处理后的掩码
    """
    processor = PostProcessor(min_area=min_area)
    result = processor.process(mask)
    
    if smooth:
        result = processor.smooth_boundary(result)
    
    return result


if __name__ == "__main__":
    # 测试
    # 创建一个带噪点的测试掩码
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    
    # 添加主区域
    cv2.rectangle(test_mask, (50, 50), (150, 150), 255, -1)
    
    # 添加噪点
    np.random.seed(42)
    noise = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    test_mask[noise > 250] = 255
    
    # 添加小孔
    test_mask[80:90, 80:90] = 0
    
    # 后处理
    processor = PostProcessor(min_area=100)
    result = processor.process(test_mask)
    
    print(f"原始掩码篡改比例: {np.sum(test_mask > 0) / test_mask.size * 100:.2f}%")
    print(f"处理后篡改比例: {np.sum(result > 0) / result.size * 100:.2f}%")