"""
自适应多方法融合器

将多个检测器的结果融合，提升整体检测效果
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


class AdaptiveFusion:
    """
    自适应多方法融合器
    
    支持多种融合策略：
    - 加权平均
    - 投票法
    - 最大值融合
    - 自适应权重
    """
    
    # 默认权重（根据调优实验结果）
    DEFAULT_WEIGHTS = {
        'ela': 0.5,      # ELA效果最好，权重最高
        'dct': 0.3,      # DCT次优
        'noise': 0.2,    # Noise效果最差
        'copy_move': 0.0  # CopyMove对当前数据无效
    }
    
    # JPEG图像权重
    JPEG_WEIGHTS = {
        'ela': 0.6,
        'dct': 0.3,
        'noise': 0.1,
        'copy_move': 0.0
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        初始化融合器
        
        Args:
            weights: 自定义权重，None则使用默认权重
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
    
    def normalize_weights(self):
        """归一化权重，使总和为1"""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total
    
    def fusion_weighted_average(self, heatmaps: Dict[str, np.ndarray],
                                 weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        加权平均融合
        
        Args:
            heatmaps: {method_name: heatmap}
            weights: 权重字典
            
        Returns:
            fused_heatmap: 融合后的热力图
        """
        if weights is None:
            weights = self.weights
        
        # 获取图像尺寸
        h, w = list(heatmaps.values())[0].shape
        
        # 初始化融合结果
        fused = np.zeros((h, w), dtype=np.float64)
        total_weight = 0.0
        
        for name, heatmap in heatmaps.items():
            if name in weights:
                w_val = weights[name]
                fused += w_val * heatmap
                total_weight += w_val
        
        # 归一化
        if total_weight > 0:
            fused /= total_weight
        
        return fused
    
    def fusion_voting(self, heatmaps: Dict[str, np.ndarray],
                       vote_threshold: int = 2) -> np.ndarray:
        """
        投票法融合
        
        Args:
            heatmaps: {method_name: heatmap}
            vote_threshold: 至少多少个检测器认为篡改才标记
            
        Returns:
            fused_heatmap: 融合后的热力图
        """
        h, w = list(heatmaps.values())[0].shape
        
        # 将每个热力图二值化
        votes = np.zeros((h, w), dtype=np.int32)
        
        for heatmap in heatmaps.values():
            binary = (heatmap > 0.5).astype(np.int32)
            votes += binary
        
        # 投票结果
        fused = (votes >= vote_threshold).astype(np.float64)
        
        return fused
    
    def fusion_max(self, heatmaps: Dict[str, np.ndarray]) -> np.ndarray:
        """
        最大值融合
        
        取所有检测器中最大的值
        """
        heatmap_list = list(heatmaps.values())
        fused = np.maximum.reduce(heatmap_list)
        return fused
    
    def fusion_adaptive(self, heatmaps: Dict[str, np.ndarray],
                        is_jpeg: bool = True) -> np.ndarray:
        """
        自适应融合
        
        根据图像类型自动调整权重
        
        Args:
            heatmaps: 热力图字典
            is_jpeg: 是否为JPEG图像
            
        Returns:
            fused_heatmap
        """
        if is_jpeg:
            weights = self.JPEG_WEIGHTS
        else:
            weights = self.DEFAULT_WEIGHTS
        
        return self.fusion_weighted_average(heatmaps, weights)
    
    def fusion_confidence_weighted(self, heatmaps: Dict[str, np.ndarray],
                                    confidences: Dict[str, float]) -> np.ndarray:
        """
        置信度加权融合
        
        根据每个检测器的置信度动态调整权重
        
        Args:
            heatmaps: 热力图字典
            confidences: 置信度字典 {method: confidence}
            
        Returns:
            fused_heatmap
        """
        h, w = list(heatmaps.values())[0].shape
        fused = np.zeros((h, w), dtype=np.float64)
        total_weight = 0.0
        
        for name, heatmap in heatmaps.items():
            if name in confidences:
                conf = confidences[name]
                fused += conf * heatmap
                total_weight += conf
        
        if total_weight > 0:
            fused /= total_weight
        
        return fused
    
    def threshold(self, heatmap: np.ndarray, 
                  method: str = 'otsu',
                  threshold: Optional[float] = None) -> np.ndarray:
        """
        阈值分割
        
        Args:
            heatmap: 热力图 (0-1)
            method: 'otsu', 'adaptive', 'fixed'
            threshold: 固定阈值
            
        Returns:
            mask: 二值掩码
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        if method == 'otsu':
            _, mask = cv2.threshold(heatmap_uint8, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            mask = cv2.adaptiveThreshold(heatmap_uint8, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        else:
            if threshold is None:
                threshold = 0.2  # 默认使用调优后的阈值
            _, mask = cv2.threshold(heatmap_uint8, int(threshold * 255), 255,
                                    cv2.THRESH_BINARY)
        
        return mask


class MultiDetectorFusion:
    """
    多检测器融合封装
    
    整合所有检测器的结果
    """
    
    def __init__(self):
        from src.detection.ela_detector import ELADetector
        from src.detection.noise_detector import NoiseConsistencyDetector
        from src.detection.dct_detector import DCTBlockDetector
        from src.detection.copy_move_detector import CopyMoveDetector
        
        self.detectors = {
            'ela': ELADetector(),
            'noise': NoiseConsistencyDetector(block_size=32),
            'dct': DCTBlockDetector(),
            'copy_move': CopyMoveDetector()
        }
        
        # 调优后的最优阈值
        self.optimal_thresholds = {
            'ela': 0.2,
            'dct': 0.3,
            'noise': 0.3,
            'copy_move': 0.5
        }
        
        self.fusion = AdaptiveFusion()
    
    def detect(self, image: np.ndarray, 
               methods: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        检测并融合
        
        Args:
            image: 输入图像
            methods: 使用的方法列表，None则使用所有方法
            
        Returns:
            (fused_heatmap, mask, details)
        """
        if methods is None:
            methods = ['ela', 'dct', 'noise']  # 不使用copy_move
        
        heatmaps = {}
        masks = {}
        details = {}
        
        for method in methods:
            if method not in self.detectors:
                continue
            
            detector = self.detectors[method]
            
            try:
                if method == 'noise':
                    heatmap, mask, _ = detector.detect_full(image)
                elif method == 'copy_move':
                    mask = detector.detect(image)
                    heatmap = detector.get_heatmap(image)
                else:
                    heatmap = detector.detect(image)
                    threshold = self.optimal_thresholds.get(method, 0.5)
                    mask = detector.get_mask(heatmap, threshold=threshold)
                
                heatmaps[method] = heatmap
                masks[method] = mask
                details[method] = {
                    'heatmap_mean': float(np.mean(heatmap)),
                    'mask_ratio': float(np.sum(mask > 0) / mask.size)
                }
            except Exception as e:
                print(f"Warning: {method} detection failed: {e}")
        
        # 融合
        if len(heatmaps) > 0:
            fused_heatmap = self.fusion.fusion_adaptive(heatmaps, is_jpeg=True)
            fused_mask = self.fusion.threshold(fused_heatmap, method='fixed', threshold=0.2)
        else:
            h, w = image.shape[:2]
            fused_heatmap = np.zeros((h, w), dtype=np.float64)
            fused_mask = np.zeros((h, w), dtype=np.uint8)
        
        return fused_heatmap, fused_mask, {
            'heatmaps': heatmaps,
            'masks': masks,
            'details': details
        }
    
    def detect_from_file(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """从文件检测"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.detect(image)


if __name__ == "__main__":
    # 测试
    import sys
    
    fusion = AdaptiveFusion()
    
    # 创建测试热力图
    h, w = 256, 256
    heatmaps = {
        'ela': np.random.rand(h, w) * 0.3 + 0.2,
        'dct': np.random.rand(h, w) * 0.2 + 0.1,
        'noise': np.random.rand(h, w) * 0.1
    }
    
    # 在某个区域添加高值
    heatmaps['ela'][100:150, 100:150] = 0.8
    heatmaps['dct'][100:150, 100:150] = 0.7
    
    # 测试融合
    fused_avg = fusion.fusion_weighted_average(heatmaps)
    fused_max = fusion.fusion_max(heatmaps)
    fused_vote = fusion.fusion_voting(heatmaps, vote_threshold=2)
    
    print(f"加权平均融合: mean={fused_avg.mean():.4f}")
    print(f"最大值融合: mean={fused_max.mean():.4f}")
    print(f"投票融合: mean={fused_vote.mean():.4f}")