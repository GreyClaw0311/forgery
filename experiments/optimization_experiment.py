#!/usr/bin/env python3
"""
图像篡改检测优化实验

实现多种优化方案并对比效果：
1. 特征融合策略（平均、加权、最大值、乘法）
2. 阈值优化（Otsu、自适应、手动、百分位）
3. 后处理（形态学、连通域过滤）
4. 高级检测器测试
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import json
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import (
    ALL_DETECTORS,
    DEFAULT_DETECTORS,
    get_detector_by_name,
)
from src.utils import (
    load_image,
    save_image,
    load_mask,
    compute_iou,
    compute_precision_recall,
    compute_f1,
    visualize_result,
    get_dataset_files,
)


# ============================================================================
# 优化方案实现
# ============================================================================

class ThresholdOptimizer:
    """阈值优化策略"""
    
    @staticmethod
    def otsu(feature_map: np.ndarray) -> np.ndarray:
        """Otsu自动阈值"""
        if len(feature_map.shape) == 3:
            gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
        else:
            gray = feature_map.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    @staticmethod
    def adaptive_mean(feature_map: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
        """自适应均值阈值"""
        if len(feature_map.shape) == 3:
            gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
        else:
            gray = feature_map.copy()
        
        gray = gray.astype(np.uint8)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, block_size, C)
        return binary
    
    @staticmethod
    def adaptive_gaussian(feature_map: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
        """自适应高斯阈值"""
        if len(feature_map.shape) == 3:
            gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
        else:
            gray = feature_map.copy()
        
        gray = gray.astype(np.uint8)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, C)
        return binary
    
    @staticmethod
    def manual(feature_map: np.ndarray, threshold: int = 127) -> np.ndarray:
        """手动固定阈值"""
        if len(feature_map.shape) == 3:
            gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
        else:
            gray = feature_map.copy()
        
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def percentile(feature_map: np.ndarray, percentile: float = 90) -> np.ndarray:
        """百分位阈值"""
        if len(feature_map.shape) == 3:
            gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
        else:
            gray = feature_map.copy()
        
        threshold = np.percentile(gray, percentile)
        _, binary = cv2.threshold(gray.astype(np.uint8), int(threshold), 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def double_threshold(feature_map: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
        """双阈值"""
        if len(feature_map.shape) == 3:
            gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
        else:
            gray = feature_map.copy()
        
        gray = gray.astype(np.uint8)
        high_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = int(0.5 * high_thresh)
        
        _, strong = cv2.threshold(gray, high_thresh, 255, cv2.THRESH_BINARY)
        _, weak = cv2.threshold(gray, low_thresh, 255, cv2.THRESH_BINARY)
        
        # 强点直接保留，弱点在强点附近才保留
        kernel = np.ones((3, 3), np.uint8)
        strong_dilated = cv2.dilate(strong, kernel, iterations=1)
        result = np.where((weak == 255) & (strong_dilated == 255), 255, 0).astype(np.uint8)
        
        return result


class PostProcessor:
    """后处理策略"""
    
    @staticmethod
    def morphological_open(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """形态学开运算（去除小噪点）"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def morphological_close(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """形态学闭运算（填充空洞）"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    @staticmethod
    def morphological_combo(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """组合形态学操作（开+闭）"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed
    
    @staticmethod
    def connected_component_filter(binary: np.ndarray, min_area: int = 100) -> np.ndarray:
        """连通域过滤（去除小区域）"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        result = np.zeros_like(binary)
        for i in range(1, num_labels):  # 跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                result[labels == i] = 255
        
        return result
    
    @staticmethod
    def median_blur(binary: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """中值滤波"""
        return cv2.medianBlur(binary, kernel_size)
    
    @staticmethod
    def full_postprocess(binary: np.ndarray, min_area: int = 100, kernel_size: int = 3) -> np.ndarray:
        """完整后处理流程"""
        # 1. 中值滤波去噪
        denoised = cv2.medianBlur(binary, 3)
        # 2. 形态学操作
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        # 3. 连通域过滤
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        result = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                result[labels == i] = 255
        return result


class FeatureFusion:
    """特征融合策略"""
    
    @staticmethod
    def average(feature_maps: List[np.ndarray]) -> np.ndarray:
        """平均融合"""
        # 确保所有特征图尺寸一致
        resized = []
        target_shape = feature_maps[0].shape[:2]
        for fm in feature_maps:
            if fm.shape[:2] != target_shape:
                fm = cv2.resize(fm, (target_shape[1], target_shape[0]))
            resized.append(fm.astype(np.float32))
        
        return np.mean(resized, axis=0).astype(np.uint8)
    
    @staticmethod
    def weighted_average(feature_maps: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """加权平均融合"""
        assert len(feature_maps) == len(weights)
        
        target_shape = feature_maps[0].shape[:2]
        weighted_sum = np.zeros(target_shape, dtype=np.float32)
        total_weight = 0
        
        for fm, w in zip(feature_maps, weights):
            if fm.shape[:2] != target_shape:
                fm = cv2.resize(fm, (target_shape[1], target_shape[0]))
            weighted_sum += fm.astype(np.float32) * w
            total_weight += w
        
        return (weighted_sum / total_weight).astype(np.uint8)
    
    @staticmethod
    def maximum(feature_maps: List[np.ndarray]) -> np.ndarray:
        """最大值融合"""
        target_shape = feature_maps[0].shape[:2]
        resized = []
        for fm in feature_maps:
            if fm.shape[:2] != target_shape:
                fm = cv2.resize(fm, (target_shape[1], target_shape[0]))
            resized.append(fm.astype(np.float32))
        
        return np.maximum.reduce(resized).astype(np.uint8)
    
    @staticmethod
    def minimum(feature_maps: List[np.ndarray]) -> np.ndarray:
        """最小值融合"""
        target_shape = feature_maps[0].shape[:2]
        resized = []
        for fm in feature_maps:
            if fm.shape[:2] != target_shape:
                fm = cv2.resize(fm, (target_shape[1], target_shape[0]))
            resized.append(fm.astype(np.float32))
        
        return np.minimum.reduce(resized).astype(np.uint8)
    
    @staticmethod
    def multiplication(feature_maps: List[np.ndarray]) -> np.ndarray:
        """乘法融合（需要所有特征都高才高）"""
        target_shape = feature_maps[0].shape[:2]
        product = np.ones(target_shape, dtype=np.float32)
        
        for fm in feature_maps:
            if fm.shape[:2] != target_shape:
                fm = cv2.resize(fm, (target_shape[1], target_shape[0]))
            # 归一化到0-1
            normalized = fm.astype(np.float32) / 255.0
            product *= (normalized + 0.1)  # 加小偏移避免完全为0
        
        # 归一化回0-255
        result = ((product - 0.1 ** len(feature_maps)) / (1 - 0.1 ** len(feature_maps))) * 255
        return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================================
# 实验框架
# ============================================================================

class OptimizationExperiment:
    """优化实验框架"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集
        self.files = get_dataset_files(data_dir)
        print(f"加载数据集: {len(self.files)} 张图片")
        
        # 预计算所有特征
        self.feature_cache = {}
        self.gt_masks = {}
        self._precompute_features()
    
    def _precompute_features(self):
        """预计算所有特征（加速实验）"""
        print("\n预计算所有特征...")
        
        all_detectors = ALL_DETECTORS
        
        for i, file_info in enumerate(self.files):
            name = file_info['name']
            print(f"  [{i+1}/{len(self.files)}] {name}")
            
            # 加载图片
            image = load_image(file_info['image'])
            
            # 加载真值
            if file_info.get('mask') and os.path.exists(file_info['mask']):
                self.gt_masks[name] = load_mask(file_info['mask'])
            
            # 计算所有特征
            self.feature_cache[name] = {}
            for DetectorClass in all_detectors:
                detector = DetectorClass()
                try:
                    feature_map = detector.detect(image)
                    self.feature_cache[name][detector.name] = feature_map
                except Exception as e:
                    print(f"    警告: {detector.name} 失败 - {e}")
    
    def evaluate_binary(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """评估二值预测"""
        # 确保尺寸一致
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # 二值化
        pred_binary = (pred_mask > 127).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        iou = compute_iou(pred_binary, gt_binary)
        precision, recall = compute_precision_recall(pred_binary, gt_binary)
        f1 = compute_f1(precision, recall)
        
        return {'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def run_threshold_experiment(self) -> Dict:
        """实验1: 阈值优化对比"""
        print("\n" + "="*60)
        print("实验1: 阈值优化对比")
        print("="*60)
        
        threshold_methods = {
            'otsu': ThresholdOptimizer.otsu,
            'adaptive_mean': lambda x: ThresholdOptimizer.adaptive_mean(x, 11, 2),
            'adaptive_gaussian': lambda x: ThresholdOptimizer.adaptive_gaussian(x, 11, 2),
            'manual_127': lambda x: ThresholdOptimizer.manual(x, 127),
            'percentile_90': lambda x: ThresholdOptimizer.percentile(x, 90),
            'percentile_95': lambda x: ThresholdOptimizer.percentile(x, 95),
            'double_threshold': ThresholdOptimizer.double_threshold,
        }
        
        # 使用效果最好的DCT特征
        detector_name = 'DCT'
        results = {method: {'iou': [], 'precision': [], 'recall': [], 'f1': []} for method in threshold_methods}
        
        for name in self.feature_cache:
            if detector_name not in self.feature_cache[name]:
                continue
            if name not in self.gt_masks:
                continue
            
            feature_map = self.feature_cache[name][detector_name]
            gt_mask = self.gt_masks[name]
            
            for method_name, method_func in threshold_methods.items():
                try:
                    binary = method_func(feature_map)
                    metrics = self.evaluate_binary(binary, gt_mask)
                    for k, v in metrics.items():
                        results[method_name][k].append(v)
                except Exception as e:
                    pass
        
        # 计算平均值
        summary = {}
        for method, metrics in results.items():
            if metrics['f1']:
                summary[method] = {
                    'iou': np.mean(metrics['iou']),
                    'precision': np.mean(metrics['precision']),
                    'recall': np.mean(metrics['recall']),
                    'f1': np.mean(metrics['f1']),
                    'count': len(metrics['f1'])
                }
        
        # 打印结果
        print(f"\n阈值方法对比 (使用{detector_name}特征):")
        print(f"{'方法':<20} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
        print("-" * 62)
        for method, m in sorted(summary.items(), key=lambda x: x[1]['f1'], reverse=True):
            print(f"{method:<20} {m['iou']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
        
        return {'threshold_experiment': summary}
    
    def run_postprocess_experiment(self) -> Dict:
        """实验2: 后处理对比"""
        print("\n" + "="*60)
        print("实验2: 后处理对比")
        print("="*60)
        
        postprocess_methods = {
            'none': lambda x: x,
            'morph_open': lambda x: PostProcessor.morphological_open(x, 3),
            'morph_close': lambda x: PostProcessor.morphological_close(x, 3),
            'morph_combo': lambda x: PostProcessor.morphological_combo(x, 3),
            'connected_filter_100': lambda x: PostProcessor.connected_component_filter(x, 100),
            'connected_filter_500': lambda x: PostProcessor.connected_component_filter(x, 500),
            'median_blur': lambda x: PostProcessor.median_blur(x, 5),
            'full_postprocess': lambda x: PostProcessor.full_postprocess(x, 100, 3),
        }
        
        detector_name = 'DCT'
        results = {method: {'iou': [], 'precision': [], 'recall': [], 'f1': []} for method in postprocess_methods}
        
        for name in self.feature_cache:
            if detector_name not in self.feature_cache[name]:
                continue
            if name not in self.gt_masks:
                continue
            
            feature_map = self.feature_cache[name][detector_name]
            gt_mask = self.gt_masks[name]
            
            # 先用Otsu阈值
            binary = ThresholdOptimizer.otsu(feature_map)
            
            for method_name, method_func in postprocess_methods.items():
                try:
                    processed = method_func(binary)
                    metrics = self.evaluate_binary(processed, gt_mask)
                    for k, v in metrics.items():
                        results[method_name][k].append(v)
                except Exception as e:
                    pass
        
        summary = {}
        for method, metrics in results.items():
            if metrics['f1']:
                summary[method] = {
                    'iou': np.mean(metrics['iou']),
                    'precision': np.mean(metrics['precision']),
                    'recall': np.mean(metrics['recall']),
                    'f1': np.mean(metrics['f1']),
                    'count': len(metrics['f1'])
                }
        
        print(f"\n后处理方法对比 (使用{detector_name}+Otsu):")
        print(f"{'方法':<25} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
        print("-" * 67)
        for method, m in sorted(summary.items(), key=lambda x: x[1]['f1'], reverse=True):
            print(f"{method:<25} {m['iou']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
        
        return {'postprocess_experiment': summary}
    
    def run_fusion_experiment(self) -> Dict:
        """实验3: 特征融合对比"""
        print("\n" + "="*60)
        print("实验3: 特征融合对比")
        print("="*60)
        
        # 选择效果好的特征组合
        top_detectors = ['DCT', 'HOG', 'CFA', 'NOISE', 'ELA']
        
        # 不同融合策略
        fusion_strategies = {
            'single_DCT': ['DCT'],
            'pair_DCT_HOG': ['DCT', 'HOG'],
            'pair_DCT_CFA': ['DCT', 'CFA'],
            'pair_HOG_CFA': ['HOG', 'CFA'],
            'triple_DCT_HOG_CFA': ['DCT', 'HOG', 'CFA'],
            'triple_DCT_NOISE_ELA': ['DCT', 'NOISE', 'ELA'],
            'quad_DCT_HOG_CFA_NOISE': ['DCT', 'HOG', 'CFA', 'NOISE'],
            'all_5': ['DCT', 'HOG', 'CFA', 'NOISE', 'ELA'],
        }
        
        fusion_methods = {
            'average': FeatureFusion.average,
            'maximum': FeatureFusion.maximum,
            'weighted': None,  # 特殊处理
        }
        
        # 权重（根据之前实验结果，DCT效果最好）
        weights = {
            'DCT': 3.0,
            'HOG': 2.0,
            'CFA': 2.0,
            'NOISE': 1.5,
            'ELA': 1.5,
        }
        
        results = {}
        
        for strategy_name, detectors in fusion_strategies.items():
            for fusion_name, fusion_func in fusion_methods.items():
                exp_name = f"{strategy_name}_{fusion_name}"
                results[exp_name] = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
                
                for name in self.feature_cache:
                    if name not in self.gt_masks:
                        continue
                    
                    # 检查是否所有需要的特征都存在
                    missing = [d for d in detectors if d not in self.feature_cache[name]]
                    if missing:
                        continue
                    
                    feature_maps = [self.feature_cache[name][d] for d in detectors]
                    gt_mask = self.gt_masks[name]
                    
                    try:
                        if fusion_name == 'weighted':
                            w = [weights.get(d, 1.0) for d in detectors]
                            fused = FeatureFusion.weighted_average(feature_maps, w)
                        else:
                            fused = fusion_func(feature_maps)
                        
                        binary = ThresholdOptimizer.otsu(fused)
                        metrics = self.evaluate_binary(binary, gt_mask)
                        for k, v in metrics.items():
                            results[exp_name][k].append(v)
                    except Exception as e:
                        pass
        
        summary = {}
        for exp_name, metrics in results.items():
            if metrics['f1']:
                summary[exp_name] = {
                    'iou': np.mean(metrics['iou']),
                    'precision': np.mean(metrics['precision']),
                    'recall': np.mean(metrics['recall']),
                    'f1': np.mean(metrics['f1']),
                    'count': len(metrics['f1'])
                }
        
        print(f"\n特征融合对比:")
        print(f"{'策略':<35} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
        print("-" * 77)
        for exp_name, m in sorted(summary.items(), key=lambda x: x[1]['f1'], reverse=True)[:15]:
            print(f"{exp_name:<35} {m['iou']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
        
        return {'fusion_experiment': summary}
    
    def run_advanced_detector_experiment(self) -> Dict:
        """实验4: 高级检测器对比"""
        print("\n" + "="*60)
        print("实验4: 高级检测器对比")
        print("="*60)
        
        # 核心检测器 vs 高级检测器
        core_detectors = ['ELA', 'CFA', 'DCT', 'NOISE', 'BLK', 'LBP', 'HOG', 'SIFT', 'EDGE', 'COLOR']
        advanced_detectors = [
            'ELA_Advanced', 'CFA_Interpolation', 'DCT_Residual',
            'NOISE_Variance', 'PRNU', 'BLK_Grid', 'LBP_Consistency',
            'HOG_Variance', 'GRAD_Inconsistency', 'CopyMove',
            'EDGE_Consistency', 'EDGE_Density', 'ILLUMINATION', 'CHROMATIC'
        ]
        
        all_detectors = core_detectors + advanced_detectors
        results = {d: {'iou': [], 'precision': [], 'recall': [], 'f1': []} for d in all_detectors}
        
        for name in self.feature_cache:
            if name not in self.gt_masks:
                continue
            
            gt_mask = self.gt_masks[name]
            
            for det_name in all_detectors:
                if det_name not in self.feature_cache[name]:
                    continue
                
                feature_map = self.feature_cache[name][det_name]
                binary = ThresholdOptimizer.otsu(feature_map)
                metrics = self.evaluate_binary(binary, gt_mask)
                for k, v in metrics.items():
                    results[det_name][k].append(v)
        
        summary = {}
        for det_name, metrics in results.items():
            if metrics['f1']:
                summary[det_name] = {
                    'iou': np.mean(metrics['iou']),
                    'precision': np.mean(metrics['precision']),
                    'recall': np.mean(metrics['recall']),
                    'f1': np.mean(metrics['f1']),
                    'count': len(metrics['f1']),
                    'is_advanced': det_name in advanced_detectors
                }
        
        print(f"\n所有检测器对比:")
        print(f"{'检测器':<25} {'类型':<10} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
        print("-" * 77)
        for det_name, m in sorted(summary.items(), key=lambda x: x[1]['f1'], reverse=True):
            det_type = "高级" if m['is_advanced'] else "核心"
            print(f"{det_name:<25} {det_type:<10} {m['iou']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
        
        return {'advanced_detector_experiment': summary}
    
    def run_combined_optimization(self) -> Dict:
        """实验5: 组合优化（最佳方案组合）"""
        print("\n" + "="*60)
        print("实验5: 组合优化方案对比")
        print("="*60)
        
        # 根据前面实验结果，选择最佳组合
        optimization_strategies = {
            # 基线：单特征+Otsu
            'baseline_DCT_otsu': {
                'detectors': ['DCT'],
                'threshold': 'otsu',
                'postprocess': None,
            },
            # 阈值优化
            'DCT_percentile95': {
                'detectors': ['DCT'],
                'threshold': 'percentile_95',
                'postprocess': None,
            },
            # 后处理优化
            'DCT_otsu_postprocess': {
                'detectors': ['DCT'],
                'threshold': 'otsu',
                'postprocess': 'full_postprocess',
            },
            # 融合优化
            'fusion_DCT_HOG_CFA_avg': {
                'detectors': ['DCT', 'HOG', 'CFA'],
                'threshold': 'otsu',
                'postprocess': None,
                'fusion': 'average',
            },
            # 融合+后处理
            'fusion_DCT_HOG_CFA_avg_post': {
                'detectors': ['DCT', 'HOG', 'CFA'],
                'threshold': 'otsu',
                'postprocess': 'full_postprocess',
                'fusion': 'average',
            },
            # 加权融合
            'fusion_weighted_post': {
                'detectors': ['DCT', 'HOG', 'CFA'],
                'threshold': 'otsu',
                'postprocess': 'full_postprocess',
                'fusion': 'weighted',
                'weights': {'DCT': 3.0, 'HOG': 2.0, 'CFA': 2.0},
            },
            # 最大值融合
            'fusion_max_post': {
                'detectors': ['DCT', 'HOG', 'CFA'],
                'threshold': 'otsu',
                'postprocess': 'full_postprocess',
                'fusion': 'maximum',
            },
            # 百分位阈值+后处理
            'fusion_avg_percentile95_post': {
                'detectors': ['DCT', 'HOG', 'CFA'],
                'threshold': 'percentile_95',
                'postprocess': 'full_postprocess',
                'fusion': 'average',
            },
        }
        
        results = {}
        
        for strategy_name, config in optimization_strategies.items():
            results[strategy_name] = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
            
            for name in self.feature_cache:
                if name not in self.gt_masks:
                    continue
                
                detectors = config['detectors']
                missing = [d for d in detectors if d not in self.feature_cache[name]]
                if missing:
                    continue
                
                gt_mask = self.gt_masks[name]
                
                try:
                    # 获取特征图
                    feature_maps = [self.feature_cache[name][d] for d in detectors]
                    
                    # 融合
                    fusion_method = config.get('fusion', None)
                    if len(feature_maps) == 1:
                        feature_map = feature_maps[0]
                    elif fusion_method == 'average':
                        feature_map = FeatureFusion.average(feature_maps)
                    elif fusion_method == 'maximum':
                        feature_map = FeatureFusion.maximum(feature_maps)
                    elif fusion_method == 'weighted':
                        w = [config['weights'].get(d, 1.0) for d in detectors]
                        feature_map = FeatureFusion.weighted_average(feature_maps, w)
                    else:
                        feature_map = FeatureFusion.average(feature_maps)
                    
                    # 阈值
                    threshold_method = config['threshold']
                    if threshold_method == 'otsu':
                        binary = ThresholdOptimizer.otsu(feature_map)
                    elif threshold_method == 'percentile_95':
                        binary = ThresholdOptimizer.percentile(feature_map, 95)
                    elif threshold_method == 'percentile_90':
                        binary = ThresholdOptimizer.percentile(feature_map, 90)
                    else:
                        binary = ThresholdOptimizer.otsu(feature_map)
                    
                    # 后处理
                    postprocess = config.get('postprocess')
                    if postprocess == 'full_postprocess':
                        binary = PostProcessor.full_postprocess(binary, 100, 3)
                    
                    metrics = self.evaluate_binary(binary, gt_mask)
                    for k, v in metrics.items():
                        results[strategy_name][k].append(v)
                except Exception as e:
                    pass
        
        summary = {}
        for strategy_name, metrics in results.items():
            if metrics['f1']:
                summary[strategy_name] = {
                    'iou': np.mean(metrics['iou']),
                    'precision': np.mean(metrics['precision']),
                    'recall': np.mean(metrics['recall']),
                    'f1': np.mean(metrics['f1']),
                    'count': len(metrics['f1'])
                }
        
        print(f"\n组合优化方案对比:")
        print(f"{'策略':<35} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10} {'提升':>10}")
        print("-" * 89)
        
        baseline_f1 = summary.get('baseline_DCT_otsu', {}).get('f1', 0)
        for strategy_name, m in sorted(summary.items(), key=lambda x: x[1]['f1'], reverse=True):
            improvement = ((m['f1'] - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            print(f"{strategy_name:<35} {m['iou']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {improvement:>9.1f}%")
        
        return {'combined_optimization': summary}
    
    def run_all_experiments(self) -> Dict:
        """运行所有实验"""
        print("\n" + "="*60)
        print("图像篡改检测优化实验")
        print("="*60)
        print(f"数据集: {self.data_dir}")
        print(f"图片数: {len(self.files)}")
        print(f"输出目录: {self.output_dir}")
        
        all_results = {}
        
        # 运行所有实验
        all_results.update(self.run_threshold_experiment())
        all_results.update(self.run_postprocess_experiment())
        all_results.update(self.run_fusion_experiment())
        all_results.update(self.run_advanced_detector_experiment())
        all_results.update(self.run_combined_optimization())
        
        # 保存结果
        results_path = os.path.join(self.output_dir, 'optimization_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {results_path}")
        
        return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图像篡改检测优化实验')
    parser.add_argument('--data', '-d', type=str, required=True, help='数据集目录')
    parser.add_argument('--output', '-o', type=str, default='optimization_results', help='输出目录')
    
    args = parser.parse_args()
    
    experiment = OptimizationExperiment(args.data, args.output)
    experiment.run_all_experiments()


if __name__ == '__main__':
    main()