"""
Pipeline测试脚本
评估Pipeline在所有数据上的表现
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import TamperDetectionPipeline
from src.config import DATA_DIR, RESULTS_DIR

def evaluate_pipeline(pipeline, method='weighted'):
    """
    评估Pipeline性能
    
    Args:
        pipeline: Pipeline实例
        method: 预测方法
        
    Returns:
        results: 评估结果
    """
    results = {
        'easy': {'total': 0, 'correct': 0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'difficult': {'total': 0, 'correct': 0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'good': {'total': 0, 'correct': 0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    }
    
    categories = ['easy', 'difficult', 'good']
    
    for category in categories:
        if category == 'good':
            img_dir = os.path.join(DATA_DIR, category)
            expected_tampered = False  # good类不应该被检测为篡改
        else:
            img_dir = os.path.join(DATA_DIR, category, 'images')
            expected_tampered = True  # easy和difficult类应该被检测为篡改
        
        if not os.path.exists(img_dir):
            continue
        
        for img_file in os.listdir(img_dir):
            if not img_file.endswith(('.jpg', '.png')):
                continue
            
            img_path = os.path.join(img_dir, img_file)
            
            try:
                is_tampered, confidence, features = pipeline.predict(img_path, method)
                
                results[category]['total'] += 1
                
                if is_tampered == expected_tampered:
                    results[category]['correct'] += 1
                
                # 混淆矩阵
                if expected_tampered and is_tampered:
                    results[category]['tp'] += 1
                elif expected_tampered and not is_tampered:
                    results[category]['fn'] += 1
                elif not expected_tampered and is_tampered:
                    results[category]['fp'] += 1
                else:
                    results[category]['tn'] += 1
                    
            except Exception as e:
                print(f"处理 {img_file} 时出错: {e}")
    
    return results

def calculate_metrics(results):
    """计算评估指标"""
    metrics = {}
    
    for category, data in results.items():
        total = data['total']
        if total == 0:
            continue
        
        accuracy = data['correct'] / total
        
        # 计算precision和recall
        tp = data['tp']
        fp = data['fp']
        tn = data['tn']
        fn = data['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[category] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    
    return metrics

def main():
    print("=" * 60)
    print("Pipeline性能评估")
    print("=" * 60)
    
    # 创建Pipeline
    pipeline = TamperDetectionPipeline(
        features=['dct', 'noise', 'ela'],
        thresholds={'dct': 0.5, 'noise': 0.4, 'ela': 15},
        weights={'dct': 1.0, 'noise': 1.0, 'ela': 0.5}
    )
    
    # 评估
    print("\n使用加权投票法评估...")
    results = evaluate_pipeline(pipeline, method='weighted')
    metrics = calculate_metrics(results)
    
    # 输出结果
    print("\n各类别性能:")
    print("-" * 60)
    print(f"{'类别':<12} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    print("-" * 60)
    
    for category, m in metrics.items():
        print(f"{category:<12} {m['accuracy']:.2%}     {m['precision']:.2%}     {m['recall']:.2%}     {m['f1']:.2%}")
    
    # 计算总体性能
    total_tp = sum(m['tp'] for m in metrics.values())
    total_fp = sum(m['fp'] for m in metrics.values())
    total_tn = sum(m['tn'] for m in metrics.values())
    total_fn = sum(m['fn'] for m in metrics.values())
    
    total_correct = total_tp + total_tn
    total = total_tp + total_fp + total_tn + total_fn
    
    overall_accuracy = total_correct / total if total > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("-" * 60)
    print(f"{'总体':<12} {overall_accuracy:.2%}     {overall_precision:.2%}     {overall_recall:.2%}     {overall_f1:.2%}")
    print("=" * 60)
    
    return metrics

if __name__ == '__main__':
    main()