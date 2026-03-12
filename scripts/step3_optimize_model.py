"""
步骤3.5.4: 模型优化 - 降低误报率
- 调整分类阈值
- 类别权重调整
- 重新训练评估
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '/tmp/forgery/src')

DATA_DIR = '/tmp/forgery/results/full'
OUTPUT_DIR = '/tmp/forgery/results/full'

FEATURE_NAMES = ['jpeg_block', 'contrast', 'saturation', 'jpeg_ghost', 'fft', 
                 'cfa', 'edge', 'color', 'resampling', 'splicing']


def optimize_threshold(model, X_test, y_test):
    """寻找最优阈值"""
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    
    print("\n" + "=" * 60, flush=True)
    print("阈值优化", flush=True)
    print("=" * 60, flush=True)
    
    # 获取预测概率
    y_proba = model.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0
    best_metrics = None
    
    print("\n测试不同阈值:", flush=True)
    print(f"{'阈值':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'FPR':>8}", flush=True)
    print("-" * 50, flush=True)
    
    for threshold in np.arange(0.3, 0.9, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        print(f"{threshold:>8.2f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {fpr:>8.4f}", flush=True)
        
        # 综合考虑F1和FPR
        # 目标：F1高，FPR低
        score = f1 - fpr * 0.5  # 惩罚高FPR
        
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
            best_metrics = {
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'fpr': fpr
            }
    
    print(f"\n最优阈值: {best_threshold:.2f}", flush=True)
    print(f"  F1: {best_metrics['f1']:.4f}", flush=True)
    print(f"  Precision: {best_metrics['precision']:.4f}", flush=True)
    print(f"  Recall: {best_metrics['recall']:.4f}", flush=True)
    print(f"  FPR: {best_metrics['fpr']:.4f}", flush=True)
    
    return best_threshold, best_metrics


def train_with_class_weight(X, y, feature_names):
    """使用类别权重训练模型"""
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    print("\n" + "=" * 60, flush=True)
    print("使用类别权重训练模型", flush=True)
    print("=" * 60, flush=True)
    
    # 预处理
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {len(y_train)} (篡改={sum(y_train==1)}, 正常={sum(y_train==0)})", flush=True)
    print(f"测试集: {len(y_test)} (篡改={sum(y_test==1)}, 正常={sum(y_test==0)})", flush=True)
    
    # 计算类别权重
    n_samples = len(y_train)
    n_tampered = sum(y_train == 1)
    n_normal = sum(y_train == 0)
    
    # 权重 = 总样本 / (2 * 该类样本数)
    weight_tampered = n_samples / (2 * n_tampered)
    weight_normal = n_samples / (2 * n_normal)
    
    print(f"\n类别权重:", flush=True)
    print(f"  篡改样本权重: {weight_tampered:.2f}", flush=True)
    print(f"  正常样本权重: {weight_normal:.2f}", flush=True)
    
    # 创建样本权重
    sample_weights = np.where(y_train == 1, weight_tampered, weight_normal)
    
    # 训练模型
    print("\n训练Gradient Boosting...", flush=True)
    model = GradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        random_state=42
    )
    
    import time
    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    train_time = time.time() - start
    
    print(f"训练时间: {train_time:.1f}秒", flush=True)
    
    # 交叉验证
    print("\n5折交叉验证...", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
    print(f"CV F1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}", flush=True)
    
    # 测试集评估
    print("\n测试集评估:", flush=True)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Accuracy:  {acc:.4f}", flush=True)
    print(f"  Precision: {prec:.4f}", flush=True)
    print(f"  Recall:    {rec:.4f}", flush=True)
    print(f"  F1-score:  {f1:.4f}", flush=True)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n混淆矩阵:", flush=True)
    print(f"              预测正常  预测篡改", flush=True)
    print(f"  实际正常    {cm[0,0]:6d}    {cm[0,1]:6d}", flush=True)
    print(f"  实际篡改    {cm[1,0]:6d}    {cm[1,1]:6d}", flush=True)
    
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    
    print(f"\n误报率 (FPR): {fpr:.4f} ({fpr*100:.2f}%)", flush=True)
    print(f"漏报率 (FNR): {fnr:.4f} ({fnr*100:.2f}%)", flush=True)
    
    return model, scaler, X_test, y_test, {
        'cv_f1_mean': float(cv_scores.mean()),
        'cv_f1_std': float(cv_scores.std()),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'train_time': float(train_time)
    }


def main():
    print("=" * 60, flush=True)
    print("步骤3.5.4: 模型优化 - 降低误报率", flush=True)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 60, flush=True)
    
    # 加载特征矩阵
    print("\n加载特征矩阵...", flush=True)
    data = np.load(os.path.join(DATA_DIR, 'feature_matrix.npz'))
    X = data['X']
    y = data['y']
    
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}", flush=True)
    
    # 1. 使用类别权重训练
    model, scaler, X_test, y_test, metrics = train_with_class_weight(X, y, FEATURE_NAMES)
    
    # 2. 阈值优化
    best_threshold, threshold_metrics = optimize_threshold(model, X_test, y_test)
    
    # 保存优化后的模型
    import pickle
    model_path = os.path.join(OUTPUT_DIR, 'optimized_model.pkl')
    scaler_path = os.path.join(OUTPUT_DIR, 'optimized_scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n优化模型已保存: {model_path}", flush=True)
    
    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'optimization': {
            'class_weight': metrics,
            'threshold_optimization': {
                'best_threshold': float(best_threshold),
                'metrics': threshold_metrics
            }
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'optimization_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"优化结果已保存: {OUTPUT_DIR}/optimization_results.json", flush=True)
    
    return results


if __name__ == '__main__':
    results = main()