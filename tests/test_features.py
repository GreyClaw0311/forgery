"""
特征测试脚本
测试各个特征在三类数据上的效果
"""

import os
import sys
import json
import time
from datetime import datetime

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import DATA_DIR, RESULTS_DIR, REPORTS_DIR, CATEGORIES, FEATURES

# 导入特征模块
import features.feature_ela as feature_ela
import features.feature_dct as feature_dct
import features.feature_cfa as feature_cfa
import features.feature_noise as feature_noise
import features.feature_edge as feature_edge
import features.feature_lbp as feature_lbp
import features.feature_histogram as feature_histogram
import features.feature_sift as feature_sift
import features.feature_fft as feature_fft
import features.feature_metadata as feature_metadata

def get_feature_module(feature_name):
    """获取特征模块"""
    modules = {
        'ela': feature_ela,
        'dct': feature_dct,
        'cfa': feature_cfa,
        'noise': feature_noise,
        'edge': feature_edge,
        'lbp': feature_lbp,
        'histogram': feature_histogram,
        'sift': feature_sift,
        'fft': feature_fft,
        'metadata': feature_metadata
    }
    return modules.get(feature_name)

def test_single_feature(feature_name, category):
    """
    测试单个特征在特定类别数据上的效果

    Args:
        feature_name: 特征名称
        category: 数据类别 (easy, difficult, good)

    Returns:
        results: 测试结果字典
    """
    module = get_feature_module(feature_name)
    if module is None:
        return None

    detect_func = getattr(module, f'detect_tampering_{feature_name}', None)
    if detect_func is None:
        return None

    # 获取图像路径
    if category == 'good':
        img_dir = os.path.join(DATA_DIR, category)
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    else:
        img_dir = os.path.join(DATA_DIR, category, 'images')
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    if not os.path.exists(img_dir):
        return None

    results = {
        'feature': feature_name,
        'category': category,
        'total': 0,
        'detected': 0,
        'scores': [],
        'errors': 0
    }

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        try:
            is_tampered, score = detect_func(img_path)
            results['total'] += 1
            results['scores'].append(score)
            if is_tampered:
                results['detected'] += 1
        except Exception as e:
            results['errors'] += 1

    # 计算统计指标
    if results['total'] > 0:
        results['detection_rate'] = results['detected'] / results['total']
        results['mean_score'] = sum(results['scores']) / len(results['scores'])
        results['std_score'] = (sum((s - results['mean_score'])**2 for s in results['scores']) / len(results['scores'])) ** 0.5
    else:
        results['detection_rate'] = 0
        results['mean_score'] = 0
        results['std_score'] = 0

    return results

def run_all_tests():
    """运行所有特征测试"""
    print("=" * 60)
    print("图像篡改检测 - 特征实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}

    for feature_name in FEATURES:
        print(f"\n测试特征: {feature_name.upper()}")
        all_results[feature_name] = {}

        for category in CATEGORIES:
            print(f"  类别: {category}...", end=" ")
            result = test_single_feature(feature_name, category)
            if result:
                all_results[feature_name][category] = result
                print(f"检测率: {result['detection_rate']:.2%}, 平均分: {result['mean_score']:.4f}")
            else:
                print("跳过")

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, 'feature_test_results.json')
    
    # 转换numpy类型为Python原生类型
    def convert_to_native(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    all_results_native = convert_to_native(all_results)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results_native, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {result_file}")
    return all_results

if __name__ == '__main__':
    run_all_tests()
