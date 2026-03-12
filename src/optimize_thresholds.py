"""
Pipeline阈值优化
通过网格搜索找到最优阈值组合
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import TamperDetectionPipeline
from src.config import DATA_DIR
import itertools

def evaluate_with_thresholds(thresholds, weights):
    """使用给定阈值评估性能"""
    pipeline = TamperDetectionPipeline(
        features=['dct', 'noise', 'ela'],
        thresholds=thresholds,
        weights=weights
    )
    
    results = {'easy': {'total': 0, 'detected': 0},
               'difficult': {'total': 0, 'detected': 0},
               'good': {'total': 0, 'detected': 0}}
    
    categories = ['easy', 'difficult', 'good']
    
    for category in categories:
        if category == 'good':
            img_dir = os.path.join(DATA_DIR, category)
        else:
            img_dir = os.path.join(DATA_DIR, category, 'images')
        
        if not os.path.exists(img_dir):
            continue
        
        for img_file in os.listdir(img_dir):
            if not img_file.endswith(('.jpg', '.png')):
                continue
            
            img_path = os.path.join(img_dir, img_file)
            
            try:
                is_tampered, confidence, features = pipeline.predict(img_path, 'weighted')
                results[category]['total'] += 1
                if is_tampered:
                    results[category]['detected'] += 1
            except:
                pass
    
    # 计算分数
    easy_rate = results['easy']['detected'] / results['easy']['total'] if results['easy']['total'] > 0 else 0
    difficult_rate = results['difficult']['detected'] / results['difficult']['total'] if results['difficult']['total'] > 0 else 0
    good_rate = results['good']['detected'] / results['good']['total'] if results['good']['total'] > 0 else 0
    
    # 综合分数：检测率 * (1 - 误报率)
    detection_rate = (easy_rate + difficult_rate) / 2
    false_positive_rate = good_rate
    score = detection_rate * (1 - false_positive_rate)
    
    return score, easy_rate, difficult_rate, good_rate

def optimize():
    """优化阈值"""
    print("=" * 60)
    print("Pipeline阈值优化")
    print("=" * 60)
    
    # 搜索空间
    dct_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    noise_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ela_range = [5, 10, 15, 20, 25, 30, 35, 40]
    
    best_score = 0
    best_thresholds = {}
    best_rates = {}
    
    total = len(dct_range) * len(noise_range) * len(ela_range)
    count = 0
    
    print(f"搜索空间: {total} 种组合")
    print()
    
    for dct_t in dct_range:
        for noise_t in noise_range:
            for ela_t in ela_range:
                count += 1
                
                thresholds = {'dct': dct_t, 'noise': noise_t, 'ela': ela_t}
                weights = {'dct': 1.0, 'noise': 1.0, 'ela': 0.5}
                
                score, easy_r, diff_r, good_r = evaluate_with_thresholds(thresholds, weights)
                
                if score > best_score:
                    best_score = score
                    best_thresholds = thresholds.copy()
                    best_rates = {'easy': easy_r, 'difficult': diff_r, 'good': good_r}
                    print(f"[{count}/{total}] 新最优: score={score:.4f}")
                    print(f"  阈值: dct={dct_t}, noise={noise_t}, ela={ela_t}")
                    print(f"  检测率: easy={easy_r:.2%}, difficult={diff_r:.2%}, good误报={good_r:.2%}")
    
    print()
    print("=" * 60)
    print("优化完成!")
    print("=" * 60)
    print(f"最优阈值: {best_thresholds}")
    print(f"最优分数: {best_score:.4f}")
    print(f"检测率:")
    print(f"  easy: {best_rates['easy']:.2%}")
    print(f"  difficult: {best_rates['difficult']:.2%}")
    print(f"  good误报: {best_rates['good']:.2%}")
    
    return best_thresholds, best_score

if __name__ == '__main__':
    optimize()