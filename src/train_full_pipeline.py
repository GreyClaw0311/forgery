"""
全量数据训练Pipeline
使用5731张图片重新训练模型
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import RESULTS_DIR

# 全量数据路径
FULL_DATA_DIR = '/tmp/forgery/tamper_data_full/processed'

# 导入特征模块
from features import (
    feature_ela, feature_dct, feature_cfa, feature_noise,
    feature_edge, feature_lbp, feature_histogram, feature_sift,
    feature_fft, feature_metadata,
    feature_hog, feature_color, feature_adjacency, feature_wavelet,
    feature_gradient, feature_block_dct, feature_jpeg_ghost, feature_local_noise,
    feature_resampling, feature_contrast, feature_blur, feature_saturation,
    feature_splicing
)

try:
    from features import feature_jpeg_block
    HAS_JPEG_BLOCK = True
except ImportError:
    HAS_JPEG_BLOCK = False

FEATURE_MODULES = [
    ('ela', feature_ela),
    ('dct', feature_dct),
    ('cfa', feature_cfa),
    ('noise', feature_noise),
    ('edge', feature_edge),
    ('lbp', feature_lbp),
    ('histogram', feature_histogram),
    ('sift', feature_sift),
    ('fft', feature_fft),
    ('metadata', feature_metadata),
    ('hog', feature_hog),
    ('color', feature_color),
    ('adjacency', feature_adjacency),
    ('wavelet', feature_wavelet),
    ('gradient', feature_gradient),
    ('block_dct', feature_block_dct),
    ('jpeg_ghost', feature_jpeg_ghost),
    ('local_noise', feature_local_noise),
    ('resampling', feature_resampling),
    ('contrast', feature_contrast),
    ('blur', feature_blur),
    ('saturation', feature_saturation),
    ('splicing', feature_splicing),
]

if HAS_JPEG_BLOCK:
    from features import feature_jpeg_block
    FEATURE_MODULES.append(('jpeg_block', feature_jpeg_block))

FEATURE_NAMES = [f[0] for f in FEATURE_MODULES]


def extract_feature_score(feature_name, feature_module, image_path):
    """提取单个特征分数"""
    detect_func = getattr(feature_module, f'detect_tampering_{feature_name}', None)
    if detect_func is None:
        return 0.0
    try:
        _, score = detect_func(image_path)
        return float(score)
    except:
        return 0.0


def build_feature_matrix(data_dir):
    """构建特征矩阵"""
    print("=" * 60)
    print("构建全量特征矩阵")
    print(f"数据目录: {data_dir}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 收集样本
    samples = []
    
    # Easy
    easy_dir = os.path.join(data_dir, 'easy/images')
    if os.path.exists(easy_dir):
        for f in sorted(os.listdir(easy_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(easy_dir, f),
                    'filename': f,
                    'category': 'easy',
                    'label': 1
                })
    
    # Difficult
    difficult_dir = os.path.join(data_dir, 'difficult/images')
    if os.path.exists(difficult_dir):
        for f in sorted(os.listdir(difficult_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(difficult_dir, f),
                    'filename': f,
                    'category': 'difficult',
                    'label': 1
                })
    
    # Good
    good_dir = os.path.join(data_dir, 'good')
    if os.path.exists(good_dir):
        for f in sorted(os.listdir(good_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(good_dir, f),
                    'filename': f,
                    'category': 'good',
                    'label': 0
                })
    
    print(f"总样本: {len(samples)}")
    print(f"  Easy: {sum(1 for s in samples if s['category']=='easy')}")
    print(f"  Difficult: {sum(1 for s in samples if s['category']=='difficult')}")
    print(f"  Good: {sum(1 for s in samples if s['category']=='good')}")
    
    # 提取特征
    X = []
    y = []
    
    start_time = time.time()
    for i, sample in enumerate(samples):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            print(f"进度: {i+1}/{len(samples)} ({(i+1)/len(samples)*100:.1f}%), ETA: {eta:.0f}s")
        
        features = []
        for fname, fmodule in FEATURE_MODULES:
            score = extract_feature_score(fname, fmodule, sample['path'])
            features.append(score)
        
        X.append(features)
        y.append(sample['label'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n特征矩阵: {X.shape}")
    print(f"耗时: {time.time() - start_time:.1f}s")
    
    return X, y, FEATURE_NAMES


def preprocess_features(X):
    """特征预处理"""
    print("\n特征预处理...")
    
    X_processed = X.copy()
    
    # 异常值处理
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.any(col > 1e6):
            X_processed[:, i] = np.log1p(col)
    
    return X_processed


def train_and_evaluate(X, y, feature_names):
    """训练和评估模型"""
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)
    
    # 预处理
    X_processed = preprocess_features(X)
    
    # 标准化
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {len(y_train)} (篡改={sum(y_train==1)}, 正常={sum(y_train==0)})")
    print(f"测试集: {len(y_test)} (篡改={sum(y_test==1)}, 正常={sum(y_test==0)})")
    
    # 模型
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
        'Logistic Regression': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"训练: {name}")
        print(f"{'='*40}")
        
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"训练时间: {train_time:.1f}s")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print(f"混淆矩阵:")
        print(f"  实际正常: 预测正常={cm[0,0]}, 预测篡改={cm[0,1]}")
        print(f"  实际篡改: 预测正常={cm[1,0]}, 预测篡改={cm[1,1]}")
        
        # 误报率和漏报率
        fpr = cm[0,1] / (cm[0,0] + cm[0,1])  # 误报率
        fnr = cm[1,0] / (cm[1,0] + cm[1,1])  # 漏报率
        print(f"误报率(FPR): {fpr:.4f}")
        print(f"漏报率(FNR): {fnr:.4f}")
        
        # 特征重要性
        if name == 'Random Forest':
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            print(f"\n特征重要性 Top 10:")
            for i, idx in enumerate(indices[:10]):
                print(f"  {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
        
        results[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'train_time': float(train_time)
        }
    
    # 选择最佳模型
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    print(f"\n最佳模型: {best_model_name}")
    print(f"F1-score: {results[best_model_name]['f1_score']:.4f}")
    
    return results, models, scaler


def main():
    """主函数"""
    print("=" * 60)
    print("全量数据训练Pipeline")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 构建特征矩阵
    X, y, feature_names = build_feature_matrix(FULL_DATA_DIR)
    
    # 2. 训练模型
    results, models, scaler = train_and_evaluate(X, y, feature_names)
    
    # 3. 保存结果
    output_dir = os.path.join(RESULTS_DIR, 'full')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存特征矩阵
    np.savez(os.path.join(output_dir, 'feature_matrix.npz'), X=X, y=y)
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df.to_csv(os.path.join(output_dir, 'feature_matrix.csv'), index=False)
    
    # 保存结果
    with open(os.path.join(output_dir, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果保存到: {output_dir}")
    
    return results


if __name__ == '__main__':
    results = main()