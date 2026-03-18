"""
可视化模块

用于可视化检测结果：热力图、掩码、叠加图
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List


def visualize_heatmap(heatmap: np.ndarray, 
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    将热力图转换为彩色可视化
    
    Args:
        heatmap: 归一化热力图 (0-1)
        colormap: OpenCV colormap
        
    Returns:
        彩色热力图 (BGR)
    """
    # 归一化到0-255
    heatmap_normalized = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    
    # 应用colormap
    colored = cv2.applyColorMap(heatmap_normalized, colormap)
    
    return colored


def visualize_mask(mask: np.ndarray, 
                   color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    将二值掩码转换为彩色可视化
    
    Args:
        mask: 二值掩码
        color: RGB颜色
        
    Returns:
        彩色掩码
    """
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored[mask > 0] = color
    return colored


def overlay_mask(image: np.ndarray, 
                  mask: np.ndarray,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  alpha: float = 0.5) -> np.ndarray:
    """
    将掩码叠加到原图上
    
    Args:
        image: 原图 (BGR)
        mask: 二值掩码
        color: RGB颜色
        alpha: 透明度
        
    Returns:
        叠加图像
    """
    # 确保图像是BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    overlay = image.copy()
    
    # 创建掩码颜色层
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = color
    
    # 混合
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool], 1 - alpha,
        mask_colored[mask_bool], alpha, 0
    )
    
    return overlay


def overlay_heatmap(image: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    将热力图叠加到原图上
    
    Args:
        image: 原图 (BGR)
        heatmap: 热力图
        alpha: 热力图透明度
        
    Returns:
        叠加图像
    """
    # 确保图像是BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 生成彩色热力图
    heatmap_colored = visualize_heatmap(heatmap)
    
    # 混合
    result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return result


def draw_contours(image: np.ndarray,
                   mask: np.ndarray,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
    """
    在原图上绘制掩码轮廓
    
    Args:
        image: 原图
        mask: 二值掩码
        color: 轮廓颜色
        thickness: 线宽
        
    Returns:
        带轮廓的图像
    """
    result = image.copy()
    
    # 查找轮廓
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 绘制轮廓
    cv2.drawContours(result, contours, -1, color, thickness)
    
    return result


def create_comparison_view(image: np.ndarray,
                            mask: np.ndarray,
                            heatmap: np.ndarray,
                            gt_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    创建对比视图
    
    包含: 原图 | 热力图 | 掩码叠加 | (真实掩码)
    """
    views = []
    
    # 原图
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    views.append(image)
    
    # 热力图
    heatmap_view = overlay_heatmap(image, heatmap, alpha=0.7)
    views.append(heatmap_view)
    
    # 掩码叠加
    mask_view = overlay_mask(image, mask, alpha=0.5)
    mask_view = draw_contours(mask_view, mask)
    views.append(mask_view)
    
    # 真实掩码
    if gt_mask is not None:
        gt_view = overlay_mask(image, gt_mask, color=(255, 0, 0), alpha=0.5)
        gt_view = draw_contours(gt_view, gt_mask, color=(255, 0, 0))
        views.append(gt_view)
    
    # 水平拼接
    result = np.hstack(views)
    
    return result


def save_results(output_dir: str,
                  image_name: str,
                  image: np.ndarray,
                  mask: np.ndarray,
                  heatmap: np.ndarray,
                  gt_mask: Optional[np.ndarray] = None):
    """
    保存所有结果图像
    
    Args:
        output_dir: 输出目录
        image_name: 图像名称
        image: 原图
        mask: 检测掩码
        heatmap: 热力图
        gt_mask: 真实掩码
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(image_name)[0]
    
    # 保存掩码
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), mask)
    
    # 保存热力图
    heatmap_colored = visualize_heatmap(heatmap)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_heatmap.png"), heatmap_colored)
    
    # 保存叠加图
    overlay = overlay_mask(image, mask, alpha=0.5)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), overlay)
    
    # 保存对比图
    comparison = create_comparison_view(image, mask, heatmap, gt_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_comparison.png"), comparison)


if __name__ == "__main__":
    # 测试
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_heatmap = np.random.rand(256, 256)
    test_mask = (np.random.rand(256, 256) > 0.8).astype(np.uint8) * 255
    
    # 测试各种可视化
    heatmap_colored = visualize_heatmap(test_heatmap)
    print(f"Heatmap shape: {heatmap_colored.shape}")
    
    overlay = overlay_mask(test_image, test_mask)
    print(f"Overlay shape: {overlay.shape}")
    
    comparison = create_comparison_view(test_image, test_mask, test_heatmap)
    print(f"Comparison shape: {comparison.shape}")