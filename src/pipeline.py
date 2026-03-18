"""
篡改区域检测主Pipeline

整合所有检测器、融合策略和后处理
"""

import os
import sys
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.ela_detector import ELADetector
from src.detection.noise_detector import NoiseConsistencyDetector
from src.detection.dct_detector import DCTBlockDetector
from src.detection.copy_move_detector import CopyMoveDetector
from src.detection.fusion import AdaptiveFusion
from src.utils.postprocess import PostProcessor
from src.utils.visualization import overlay_mask, visualize_heatmap, create_comparison_view


class ForgeryRegionDetector:
    """
    篡改区域检测主Pipeline
    
    流程：
    1. 多检测器并行检测
    2. 自适应融合
    3. 后处理优化
    """
    
    def __init__(self, 
                 use_methods: Optional[List[str]] = None,
                 fusion_threshold: float = 0.2,
                 min_area: int = 100):
        """
        初始化检测器
        
        Args:
            use_methods: 使用的方法列表
            fusion_threshold: 融合后的阈值
            min_area: 最小连通域面积
        """
        self.use_methods = use_methods or ['ela', 'dct', 'noise']
        self.fusion_threshold = fusion_threshold
        self.min_area = min_area
        
        # 初始化检测器
        self.detectors = {
            'ela': ELADetector(),
            'dct': DCTBlockDetector(),
            'noise': NoiseConsistencyDetector(block_size=32),
            'copy_move': CopyMoveDetector()
        }
        
        # 最优阈值（来自调优实验）
        self.thresholds = {
            'ela': 0.2,
            'dct': 0.3,
            'noise': 0.3,
            'copy_move': 0.5
        }
        
        # 融合器和后处理器
        self.fusion = AdaptiveFusion()
        self.postprocessor = PostProcessor(min_area=min_area)
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        检测篡改区域
        
        Args:
            image: 输入图像 (BGR)
            
        Returns:
            {
                'mask': 最终掩码,
                'heatmap': 融合热力图,
                'method_masks': 各方法掩码,
                'method_heatmaps': 各方法热力图
            }
        """
        h, w = image.shape[:2]
        
        # 1. 各检测器检测
        method_heatmaps = {}
        method_masks = {}
        
        for method in self.use_methods:
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
                    # 使用调优后的阈值
                    threshold = self.thresholds.get(method, 0.5)
                    mask = detector.get_mask(heatmap, threshold=threshold)
                
                method_heatmaps[method] = heatmap
                method_masks[method] = mask
                
            except Exception as e:
                print(f"Warning: {method} detection failed: {e}")
        
        # 2. 融合
        if len(method_heatmaps) > 0:
            fused_heatmap = self.fusion.fusion_adaptive(method_heatmaps, is_jpeg=True)
        else:
            fused_heatmap = np.zeros((h, w), dtype=np.float64)
        
        # 3. 阈值分割
        fused_mask = self.fusion.threshold(fused_heatmap, method='fixed', 
                                            threshold=self.fusion_threshold)
        
        # 4. 后处理
        final_mask = self.postprocessor.process(fused_mask)
        
        return {
            'mask': final_mask,
            'heatmap': fused_heatmap,
            'method_masks': method_masks,
            'method_heatmaps': method_heatmaps
        }
    
    def detect_from_file(self, image_path: str) -> Dict:
        """从文件检测"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.detect(image)
    
    def save_results(self, result: Dict, image: np.ndarray, 
                     output_dir: str, name: str):
        """
        保存结果
        
        Args:
            result: 检测结果
            image: 原图
            output_dir: 输出目录
            name: 文件名前缀
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存掩码
        cv2.imwrite(os.path.join(output_dir, f"{name}_mask.png"), result['mask'])
        
        # 保存热力图
        heatmap_colored = visualize_heatmap(result['heatmap'])
        cv2.imwrite(os.path.join(output_dir, f"{name}_heatmap.png"), heatmap_colored)
        
        # 保存叠加图
        overlay = overlay_mask(image, result['mask'], alpha=0.5)
        cv2.imwrite(os.path.join(output_dir, f"{name}_overlay.png"), overlay)


def evaluate_pipeline(images: List[Tuple[str, str]], 
                      detector: ForgeryRegionDetector) -> Dict:
    """
    评估Pipeline效果
    
    Args:
        images: [(image_path, mask_path), ...]
        detector: 检测器
        
    Returns:
        评估结果
    """
    all_metrics = []
    
    for img_path, mask_path in images:
        image = cv2.imread(img_path)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or gt_mask is None:
            continue
        
        if gt_mask.shape[:2] != image.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (image.shape[1], image.shape[0]))
        
        # 检测
        result = detector.detect(image)
        pred_mask = result['mask']
        
        # 计算指标
        pred = (pred_mask > 127).astype(np.uint8)
        gt = (gt_mask > 127).astype(np.uint8)
        
        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))
        
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        all_metrics.append({
            'image': os.path.basename(img_path),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        })
    
    # 计算平均
    avg_metrics = {
        'num_images': len(all_metrics),
        'avg_iou': np.mean([m['iou'] for m in all_metrics]),
        'avg_precision': np.mean([m['precision'] for m in all_metrics]),
        'avg_recall': np.mean([m['recall'] for m in all_metrics]),
        'avg_f1': np.mean([m['f1'] for m in all_metrics])
    }
    
    return avg_metrics


def main():
    """测试Pipeline"""
    print("=" * 60)
    print("篡改区域检测Pipeline测试")
    print("=" * 60)
    
    # 加载测试数据
    data_dir = '/root/forgery_region_detection/data'
    images = []
    
    # Easy
    easy_images = os.path.join(data_dir, 'easy/images')
    easy_masks = os.path.join(data_dir, 'easy/masks')
    for f in sorted(os.listdir(easy_images)):
        if f.endswith(('.jpg', '.png')):
            img_path = os.path.join(easy_images, f)
            base = os.path.splitext(f)[0]
            mask_path = os.path.join(easy_masks, base + '.png')
            if os.path.exists(mask_path):
                images.append((img_path, mask_path))
    
    # Difficult
    diff_images = os.path.join(data_dir, 'difficult/images')
    diff_masks = os.path.join(data_dir, 'difficult/masks')
    for f in sorted(os.listdir(diff_images)):
        if f.endswith(('.jpg', '.png')):
            img_path = os.path.join(diff_images, f)
            base = os.path.splitext(f)[0]
            mask_path = os.path.join(diff_masks, base + '.png')
            if os.path.exists(mask_path):
                images.append((img_path, mask_path))
    
    print(f"\n测试图片: {len(images)} 张")
    
    # 初始化检测器
    detector = ForgeryRegionDetector(
        use_methods=['ela', 'dct', 'noise'],
        fusion_threshold=0.2,
        min_area=100
    )
    
    # 评估
    results = evaluate_pipeline(images, detector)
    
    print("\n" + "=" * 60)
    print("Pipeline评估结果")
    print("=" * 60)
    
    print(f"\n处理图片数: {results['num_images']}")
    print(f"平均 IoU: {results['avg_iou']:.4f}")
    print(f"平均 Precision: {results['avg_precision']:.4f}")
    print(f"平均 Recall: {results['avg_recall']:.4f}")
    print(f"平均 F1: {results['avg_f1']:.4f}")
    
    # 与单检测器对比
    print("\n" + "=" * 60)
    print("对比: 融合Pipeline vs 单检测器(ELA)")
    print("=" * 60)
    
    print(f"\n{'方法':<15} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 55)
    print(f"{'ELA(单检测器)':<15} {'0.0735':>8} {'0.0770':>10} {'0.6518':>8} {'0.1119':>8}")
    print(f"{'融合Pipeline':<15} {results['avg_iou']:>8.4f} {results['avg_precision']:>10.4f} {results['avg_recall']:>8.4f} {results['avg_f1']:>8.4f}")
    
    # 计算提升
    iou_lift = (results['avg_iou'] / 0.0735 - 1) * 100
    p_lift = (results['avg_precision'] / 0.0770 - 1) * 100
    f1_lift = (results['avg_f1'] / 0.1119 - 1) * 100
    
    print(f"\n提升:")
    print(f"  IoU: {'↑' if iou_lift > 0 else '↓'} {abs(iou_lift):.1f}%")
    print(f"  Precision: {'↑' if p_lift > 0 else '↓'} {abs(p_lift):.1f}%")
    print(f"  F1: {'↑' if f1_lift > 0 else '↓'} {abs(f1_lift):.1f}%")
    
    # 保存结果
    output_file = '/root/forgery_region_detection/results/pipeline_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()