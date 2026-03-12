"""
步骤3.5.2-3.5.3: 特征相关性分析 + XGBoost模型训练
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


def analyze_correlation(X, feature_names):
    """分析特征相关性"""
    print("\n" + "=" * 60, flush=True)
    print("特征相关性分析", flush=True)
    print("=" * 60, flush=True)
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(X.T)
    
    print("\n相关性矩阵:", flush=True)
    print(f"{'':15}", end="", flush=True)
    for name in feature_names:
        print(f"{name[:8]:>8}", end="", flush=True)
    print(flush=True)
    
    for i, name in enumerate(feature_names):
        print(f"{name:15}", end="", flush=True)
        for j in range(len(feature_names)):
            print(f"{corr_matrix[i,j]:>8.2f}", end="", flush=True)
        print(flush=True)
    
    # 找出高相关特征对
    print("\n高相关特征对 (|r| > 0.7):", flush=True)
    high_corr = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.7:
                high_corr.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
                print(f"  {feature_names[i]} <-> {feature_names[j]}: r={corr_matrix[i,j]:.3f}", flush=True)
    
    if not high_corr:
        print("  无高相关特征对", flush=True)
    
    return corr_matrix, high_corr


def train_xgboost(X, y, feature_names):
    """训练XGBoost模型 (使用sklearn的GradientBoosting)"""
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    print("\n" + "=" * 60, flush=True)
    print("Gradient Boosting模型训练", flush=True)
    print("=" * 60, flush=True)
    
    # 预处理
    print("\n预处理...", flush=True)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {len(y_train)} (篡改={sum(y_train==1)}, 正常={sum(y_train==0)})", flush=True)
    print(f"测试集: {len(y_test)} (篡改={sum(y_test==1)}, 正常={sum(y_test==0)})", flush=True)
    
    # 训练Gradient Boosting
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
    model.fit(X_train, y_train)
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
    
    # 误报率和漏报率
    fpr = cm[0,1] / (cm[0,0] + cm[0,1])  # 误报率
    fnr = cm[1,0] / (cm[1,0] + cm[1,1])  # 漏报率
    print(f"\n误报率 (FPR): {fpr:.4f} ({fpr*100:.2f}%)", flush=True)
    print(f"漏报率 (FNR): {fnr:.4f} ({fnr*100:.2f}%)", flush=True)
    
    # 特征重要性
    print("\n特征重要性:", flush=True)
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}", flush=True)
    
    return model, scaler, {
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
    print("步骤3.5.2-3.5.3: 特征分析 + Gradient Boosting训练", flush=True)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 60, flush=True)
    
    # 加载特征矩阵
    print("\n加载特征矩阵...", flush=True)
    data = np.load(os.path.join(DATA_DIR, 'feature_matrix.npz'))
    X = data['X']
    y = data['y']
    
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}", flush=True)
    print(f"篡改样本: {np.sum(y==1)}, 正常样本: {np.sum(y==0)}", flush=True)
    
    # 相关性分析
    corr_matrix, high_corr = analyze_correlation(X, FEATURE_NAMES)
    
    # XGBoost训练
    model, scaler, metrics = train_xgboost(X, y, FEATURE_NAMES)
    
    # 保存模型
    import pickle
    model_path = os.path.join(OUTPUT_DIR, 'xgboost_model.pkl')
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n模型已保存: {model_path}", flush=True)
    print(f"标准化器已保存: {scaler_path}", flush=True)
    
    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'total': int(X.shape[0]),
            'tampered': int(np.sum(y==1)),
            'normal': int(np.sum(y==0)),
            'features': FEATURE_NAMES
        },
        'correlation': {
            'high_corr_pairs': [(p[0], p[1], float(p[2])) for p in high_corr]
        },
        'model': metrics
    }
    
    with open(os.path.join(OUTPUT_DIR, 'xgboost_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {OUTPUT_DIR}/xgboost_results.json", flush=True)
    
    return results


if __name__ == '__main__':
    results = main()