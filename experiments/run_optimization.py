#!/usr/bin/env python3
"""简化版优化实验 - 直接运行"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import ALL_DETECTORS, DEFAULT_DETECTORS
from src.utils import load_image, load_mask, compute_iou, compute_precision_recall, compute_f1, get_dataset_files


def threshold_otsu(feature_map):
    """Otsu阈值"""
    if len(feature_map.shape) == 3:
        gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
    else:
        gray = feature_map.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def threshold_percentile(feature_map, percentile=95):
    """百分位阈值"""
    if len(feature_map.shape) == 3:
        gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
    else:
        gray = feature_map.copy()
    threshold = np.percentile(gray, percentile)
    _, binary = cv2.threshold(gray.astype(np.uint8), int(threshold), 255, cv2.THRESH_BINARY)
    return binary


def full_postprocess(binary, min_area=100, kernel_size=3):
    """完整后处理"""
    # 1. 中值滤波
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


def fusion_average(feature_maps):
    """平均融合"""
    target_shape = feature_maps[0].shape[:2]
    resized = []
    for fm in feature_maps:
        if fm.shape[:2] != target_shape:
            fm = cv2.resize(fm, (target_shape[1], target_shape[0]))
        resized.append(fm.astype(np.float32))
    return np.mean(resized, axis=0).astype(np.uint8)


def fusion_weighted(feature_maps, weights):
    """加权融合"""
    target_shape = feature_maps[0].shape[:2]
    weighted_sum = np.zeros(target_shape, dtype=np.float32)
    total_weight = 0
    for fm, w in zip(feature_maps, weights):
        if fm.shape[:2] != target_shape:
            fm = cv2.resize(fm, (target_shape[1], target_shape[0]))
        weighted_sum += fm.astype(np.float32) * w
        total_weight += w
    return (weighted_sum / total_weight).astype(np.uint8)


def fusion_maximum(feature_maps):
    """最大值融合"""
    target_shape = feature_maps[0].shape[:2]
    resized = []
    for fm in feature_maps:
        if fm.shape[:2] != target_shape:
            fm = cv2.resize(fm, (target_shape[1], target_shape[0]))
        resized.append(fm.astype(np.float32))
    return np.maximum.reduce(resized).astype(np.uint8)


def evaluate_binary(pred_mask, gt_mask):
    """评估二值预测"""
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
    
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    iou = compute_iou(pred_binary, gt_binary)
    precision, recall = compute_precision_recall(pred_binary, gt_binary)
    f1 = compute_f1(precision, recall)
    
    return {'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1}


def main():
    print("="*70)
    print("图像篡改检测优化实验")
    print("="*70)
    
    # 加载数据集
    data_dir = "tamper_data/easy"
    files = get_dataset_files(data_dir)
    print(f"\n加载数据集: {len(files)} 张图片")
    
    # 预计算所有特征
    print("\n预计算所有特征...")
    feature_cache = {}
    gt_masks = {}
    
    for i, file_info in enumerate(files):
        name = file_info['name']
        print(f"  [{i+1}/{len(files)}] {name}", end='', flush=True)
        
        image = load_image(file_info['image'])
        
        if file_info.get('mask') and os.path.exists(file_info['mask']):
            gt_masks[name] = load_mask(file_info['mask'])
        
        feature_cache[name] = {}
        for DetectorClass in ALL_DETECTORS:
            detector = DetectorClass()
            try:
                feature_map = detector.detect(image)
                feature_cache[name][detector.name] = feature_map
            except Exception as e:
                pass
        print(" ✓")
    
    # ============================================================
    # 实验1: 阈值优化
    # ============================================================
    print("\n" + "="*70)
    print("实验1: 阈值优化对比")
    print("="*70)
    
    threshold_methods = {
        'otsu': threshold_otsu,
        'percentile_90': lambda x: threshold_percentile(x, 90),
        'percentile_95': lambda x: threshold_percentile(x, 95),
        'percentile_97': lambda x: threshold_percentile(x, 97),
    }
    
    detector_name = 'DCT'
    results_threshold = {}
    
    for method_name, method_func in threshold_methods.items():
        metrics_list = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
        
        for name in feature_cache:
            if detector_name not in feature_cache[name] or name not in gt_masks:
                continue
            
            feature_map = feature_cache[name][detector_name]
            gt_mask = gt_masks[name]
            
            try:
                binary = method_func(feature_map)
                metrics = evaluate_binary(binary, gt_mask)
                for k, v in metrics.items():
                    metrics_list[k].append(v)
            except:
                pass
        
        if metrics_list['f1']:
            results_threshold[method_name] = {
                'iou': np.mean(metrics_list['iou']),
                'precision': np.mean(metrics_list['precision']),
                'recall': np.mean(metrics_list['recall']),
                'f1': np.mean(metrics_list['f1']),
            }
    
    print(f"\n阈值方法对比 (使用{detector_name}特征):")
    print(f"{'方法':<20} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print("-" * 62)
    for method, m in sorted(results_threshold.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{method:<20} {m['iou']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    
    # ============================================================
    # 实验2: 后处理优化
    # ============================================================
    print("\n" + "="*70)
    print("实验2: 后处理对比")
    print("="*70)
    
    postprocess_methods = {
        'none': lambda x: x,
        'morph_open': lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)),
        'morph_close': lambda x: cv2.morphologyEx(x, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)),
        'morph_combo': lambda x: cv2.morphologyEx(
            cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)),
            cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)
        ),
        'connected_100': lambda x: full_postprocess(x, 100, 3),
        'connected_500': lambda x: full_postprocess(x, 500, 3),
        'full_postprocess': lambda x: full_postprocess(x, 100, 3),
    }
    
    results_postprocess = {}
    
    for method_name, method_func in postprocess_methods.items():
        metrics_list = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
        
        for name in feature_cache:
            if detector_name not in feature_cache[name] or name not in gt_masks:
                continue
            
            feature_map = feature_cache[name][detector_name]
            gt_mask = gt_masks[name]
            
            try:
                binary = threshold_otsu(feature_map)
                processed = method_func(binary)
                metrics = evaluate_binary(processed, gt_mask)
                for k, v in metrics.items():
                    metrics_list[k].append(v)
            except:
                pass
        
        if metrics_list['f1']:
            results_postprocess[method_name] = {
                'iou': np.mean(metrics_list['iou']),
                'precision': np.mean(metrics_list['precision']),
                'recall': np.mean(metrics_list['recall']),
                'f1': np.mean(metrics_list['f1']),
            }
    
    print(f"\n后处理方法对比 (使用{detector_name}+Otsu):")
    print(f"{'方法':<25} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print("-" * 67)
    for method, m in sorted(results_postprocess.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{method:<25} {m['iou']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    
    # ============================================================
    # 实验3: 特征融合
    # ============================================================
    print("\n" + "="*70)
    print("实验3: 特征融合对比")
    print("="*70)
    
    fusion_strategies = {
        'DCT_only': (['DCT'], 'none'),
        'DCT+HOG_avg': (['DCT', 'HOG'], 'average'),
        'DCT+CFA_avg': (['DCT', 'CFA'], 'average'),
        'DCT+HOG+CFA_avg': (['DCT', 'HOG', 'CFA'], 'average'),
        'DCT+HOG+CFA_max': (['DCT', 'HOG', 'CFA'], 'maximum'),
        'DCT+HOG+CFA_weighted': (['DCT', 'HOG', 'CFA'], 'weighted'),
        'DCT+NOISE+ELA_avg': (['DCT', 'NOISE', 'ELA'], 'average'),
        'All5_avg': (['DCT', 'HOG', 'CFA', 'NOISE', 'ELA'], 'average'),
    }
    
    weights_dict = {'DCT': 3.0, 'HOG': 2.0, 'CFA': 2.0, 'NOISE': 1.5, 'ELA': 1.5}
    
    results_fusion = {}
    
    for strategy_name, (detectors, fusion_type) in fusion_strategies.items():
        metrics_list = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
        
        for name in feature_cache:
            if name not in gt_masks:
                continue
            
            missing = [d for d in detectors if d not in feature_cache[name]]
            if missing:
                continue
            
            gt_mask = gt_masks[name]
            feature_maps = [feature_cache[name][d] for d in detectors]
            
            try:
                if len(feature_maps) == 1:
                    feature_map = feature_maps[0]
                elif fusion_type == 'average':
                    feature_map = fusion_average(feature_maps)
                elif fusion_type == 'maximum':
                    feature_map = fusion_maximum(feature_maps)
                elif fusion_type == 'weighted':
                    w = [weights_dict.get(d, 1.0) for d in detectors]
                    feature_map = fusion_weighted(feature_maps, w)
                else:
                    feature_map = fusion_average(feature_maps)
                
                binary = threshold_otsu(feature_map)
                metrics = evaluate_binary(binary, gt_mask)
                for k, v in metrics.items():
                    metrics_list[k].append(v)
            except:
                pass
        
        if metrics_list['f1']:
            results_fusion[strategy_name] = {
                'iou': np.mean(metrics_list['iou']),
                'precision': np.mean(metrics_list['precision']),
                'recall': np.mean(metrics_list['recall']),
                'f1': np.mean(metrics_list['f1']),
            }
    
    print(f"\n特征融合对比:")
    print(f"{'策略':<25} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print("-" * 67)
    for strategy, m in sorted(results_fusion.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{strategy:<25} {m['iou']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    
    # ============================================================
    # 实验4: 组合优化
    # ============================================================
    print("\n" + "="*70)
    print("实验4: 组合优化方案对比")
    print("="*70)
    
    combined_strategies = {
        '基线(DCT+Otsu)': {
            'detectors': ['DCT'],
            'threshold': 'otsu',
            'postprocess': False,
        },
        'DCT+百分位95': {
            'detectors': ['DCT'],
            'threshold': 'percentile_95',
            'postprocess': False,
        },
        'DCT+Otsu+后处理': {
            'detectors': ['DCT'],
            'threshold': 'otsu',
            'postprocess': True,
        },
        'DCT+百分位95+后处理': {
            'detectors': ['DCT'],
            'threshold': 'percentile_95',
            'postprocess': True,
        },
        '融合(DCT+HOG+CFA)+Otsu': {
            'detectors': ['DCT', 'HOG', 'CFA'],
            'threshold': 'otsu',
            'postprocess': False,
            'fusion': 'average',
        },
        '融合+Otsu+后处理': {
            'detectors': ['DCT', 'HOG', 'CFA'],
            'threshold': 'otsu',
            'postprocess': True,
            'fusion': 'average',
        },
        '融合+百分位95+后处理': {
            'detectors': ['DCT', 'HOG', 'CFA'],
            'threshold': 'percentile_95',
            'postprocess': True,
            'fusion': 'average',
        },
        '加权融合+Otsu+后处理': {
            'detectors': ['DCT', 'HOG', 'CFA'],
            'threshold': 'otsu',
            'postprocess': True,
            'fusion': 'weighted',
        },
        '最大值融合+Otsu+后处理': {
            'detectors': ['DCT', 'HOG', 'CFA'],
            'threshold': 'otsu',
            'postprocess': True,
            'fusion': 'maximum',
        },
    }
    
    results_combined = {}
    
    for strategy_name, config in combined_strategies.items():
        metrics_list = {'iou': [], 'precision': [], 'recall': [], 'f1': []}
        detectors = config['detectors']
        
        for name in feature_cache:
            if name not in gt_masks:
                continue
            
            missing = [d for d in detectors if d not in feature_cache[name]]
            if missing:
                continue
            
            gt_mask = gt_masks[name]
            feature_maps = [feature_cache[name][d] for d in detectors]
            
            try:
                # 融合
                fusion_type = config.get('fusion', 'average')
                if len(feature_maps) == 1:
                    feature_map = feature_maps[0]
                elif fusion_type == 'average':
                    feature_map = fusion_average(feature_maps)
                elif fusion_type == 'maximum':
                    feature_map = fusion_maximum(feature_maps)
                elif fusion_type == 'weighted':
                    w = [weights_dict.get(d, 1.0) for d in detectors]
                    feature_map = fusion_weighted(feature_maps, w)
                else:
                    feature_map = fusion_average(feature_maps)
                
                # 阈值
                threshold_type = config['threshold']
                if threshold_type == 'otsu':
                    binary = threshold_otsu(feature_map)
                elif threshold_type == 'percentile_95':
                    binary = threshold_percentile(feature_map, 95)
                else:
                    binary = threshold_otsu(feature_map)
                
                # 后处理
                if config.get('postprocess'):
                    binary = full_postprocess(binary, 100, 3)
                
                metrics = evaluate_binary(binary, gt_mask)
                for k, v in metrics.items():
                    metrics_list[k].append(v)
            except:
                pass
        
        if metrics_list['f1']:
            results_combined[strategy_name] = {
                'iou': np.mean(metrics_list['iou']),
                'precision': np.mean(metrics_list['precision']),
                'recall': np.mean(metrics_list['recall']),
                'f1': np.mean(metrics_list['f1']),
            }
    
    print(f"\n组合优化方案对比:")
    baseline_f1 = results_combined.get('基线(DCT+Otsu)', {}).get('f1', 0)
    print(f"{'策略':<30} {'IoU':>10} {'Precision':>12} {'Recall':>10} {'F1':>10} {'提升':>10}")
    print("-" * 82)
    for strategy, m in sorted(results_combined.items(), key=lambda x: x[1]['f1'], reverse=True):
        improvement = ((m['f1'] - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
        print(f"{strategy:<30} {m['iou']:>10.4f} {m['precision']:>12.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {improvement:>9.1f}%")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)
    
    # 找出最佳方案
    best_threshold = max(results_threshold.items(), key=lambda x: x[1]['f1'])
    best_postprocess = max(results_postprocess.items(), key=lambda x: x[1]['f1'])
    best_fusion = max(results_fusion.items(), key=lambda x: x[1]['f1'])
    best_combined = max(results_combined.items(), key=lambda x: x[1]['f1'])
    
    print(f"\n最佳阈值方法: {best_threshold[0]} (F1={best_threshold[1]['f1']:.4f})")
    print(f"最佳后处理方法: {best_postprocess[0]} (F1={best_postprocess[1]['f1']:.4f})")
    print(f"最佳融合策略: {best_fusion[0]} (F1={best_fusion[1]['f1']:.4f})")
    print(f"\n🏆 最佳组合方案: {best_combined[0]} (F1={best_combined[1]['f1']:.4f})")
    
    # 计算提升
    baseline = results_combined.get('基线(DCT+Otsu)', {}).get('f1', 0)
    if baseline > 0:
        improvement = (best_combined[1]['f1'] - baseline) / baseline * 100
        print(f"\n相比基线提升: {improvement:.1f}%")
    
    # 保存结果
    all_results = {
        'threshold': results_threshold,
        'postprocess': results_postprocess,
        'fusion': results_fusion,
        'combined': results_combined,
        'best': {
            'threshold': best_threshold[0],
            'postprocess': best_postprocess[0],
            'fusion': best_fusion[0],
            'combined': best_combined[0],
        }
    }
    
    output_dir = "results/optimization_easy"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_dir}/results.json")


if __name__ == '__main__':
    main()