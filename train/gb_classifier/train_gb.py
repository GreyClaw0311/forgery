#!/usr/bin/env python3
"""
图像篡改检测模型训练脚本

使用方法:
    python train/gb_classifier/train_gb.py --data_dir ./data --output_dir ./release/models
"""

import os
import sys
import argparse
import json
import time
import pickle
import numpy as np

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from release.algorithms.features import FEATURE_NAMES, extract_all_features

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def collect_samples(data_dir):
    """收集所有样本"""
    samples = []
    
    # Easy (简单篡改)
    easy_dir = os.path.join(data_dir, 'easy/images')
    if os.path.exists(easy_dir):
        for f in sorted(os.listdir(easy_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(easy_dir, f),
                    'label': 1,
                    'category': 'easy'
                })
    
    # Difficult (复杂篡改)
    diff_dir = os.path.join(data_dir, 'difficult/images')
    if os.path.exists(diff_dir):
        for f in sorted(os.listdir(diff_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(diff_dir, f),
                    'label': 1,
                    'category': 'difficult'
                })
    
    # Good (正常)
    good_dir = os.path.join(data_dir, 'good')
    if os.path.exists(good_dir):
        for f in sorted(os.listdir(good_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(good_dir, f),
                    'label': 0,
                    'category': 'good'
                })
    
    return samples


def build_feature_matrix(data_dir, output_dir):
    """构建特征矩阵"""
    print("=" * 60)
    print("步骤1: 构建特征矩阵")
    print("=" * 60)
    
    # 收集样本
    print("\n收集样本...")
    samples = collect_samples(data_dir)
    total = len(samples)
    
    if total == 0:
        print("错误: 未找到任何样本!")
        return None, None
    
    print(f"总样本: {total}")
    print(f"  Easy: {sum(1 for s in samples if s['category']=='easy')}")
    print(f"  Difficult: {sum(1 for s in samples if s['category']=='difficult')}")
    print(f"  Good: {sum(1 for s in samples if s['category']=='good')}")
    
    # 提取特征
    print("\n提取特征...")
    X_list = []
    y_list = []
    
    start_time = time.time()
    for i, sample in enumerate(samples):
        try:
            features = extract_all_features(sample['path'])
            X_list.append(features)
            y_list.append(sample['label'])
        except Exception as e:
            print(f"  警告: {sample['path']} 特征提取失败: {e}")
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{total}] 速度: {speed:.1f}张/秒")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n特征矩阵: {X.shape}")
    print(f"标签分布: 篡改={np.sum(y==1)}, 正常={np.sum(y==0)}")
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'feature_matrix.npz'), X=X, y=y, feature_names=FEATURE_NAMES)
    
    return X, y


def train_model(X, y, output_dir):
    """训练模型"""
    print("\n" + "=" * 60)
    print("步骤2: 训练模型")
    print("=" * 60)
    
    # 标准化
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集: {len(y_train)} (篡改={sum(y_train==1)}, 正常={sum(y_train==0)})")
    print(f"测试集: {len(y_test)} (篡改={sum(y_test==1)}, 正常={sum(y_test==0)})")
    
    # 计算类别权重
    n_samples = len(y_train)
    n_tampered = sum(y_train == 1)
    n_normal = sum(y_train == 0)
    
    weight_tampered = n_samples / (2 * n_tampered) if n_tampered > 0 else 1
    weight_normal = n_samples / (2 * n_normal) if n_normal > 0 else 1
    
    print(f"\n类别权重:")
    print(f"  篡改样本权重: {weight_tampered:.2f}")
    print(f"  正常样本权重: {weight_normal:.2f}")
    
    sample_weights = np.where(y_train == 1, weight_tampered, weight_normal)
    
    # 训练模型
    print("\n训练 Gradient Boosting...")
    model = GradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        random_state=42
    )
    
    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    train_time = time.time() - start
    
    print(f"训练时间: {train_time:.1f}秒")
    
    # 交叉验证
    print("\n5折交叉验证...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
    print(f"CV F1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 测试集评估
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    print(f"\n测试集评估:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  FPR:       {fpr:.4f}")
    print(f"  FNR:       {fnr:.4f}")
    
    # 阈值优化
    print("\n" + "=" * 60)
    print("步骤3: 阈值优化")
    print("=" * 60)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.5
    best_score = 0
    best_metrics = None
    
    print("\n测试不同阈值:")
    print(f"{'阈值':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'FPR':>8}")
    print("-" * 50)
    
    for threshold in np.arange(0.3, 0.9, 0.05):
        y_pred_t = (y_proba >= threshold).astype(int)
        
        f1_t = f1_score(y_test, y_pred_t, zero_division=0)
        prec_t = precision_score(y_test, y_pred_t, zero_division=0)
        rec_t = recall_score(y_test, y_pred_t, zero_division=0)
        
        cm_t = confusion_matrix(y_test, y_pred_t)
        fpr_t = cm_t[0, 1] / (cm_t[0, 0] + cm_t[0, 1]) if (cm_t[0, 0] + cm_t[0, 1]) > 0 else 0
        
        print(f"{threshold:>8.2f} {f1_t:>8.4f} {prec_t:>10.4f} {rec_t:>8.4f} {fpr_t:>8.4f}")
        
        score = f1_t - fpr_t * 0.5
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {'f1': f1_t, 'precision': prec_t, 'recall': rec_t, 'fpr': fpr_t}
    
    print(f"\n最优阈值: {best_threshold:.2f}")
    
    # 保存模型
    model_path = os.path.join(output_dir, 'model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    config_path = os.path.join(output_dir, 'config.json')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    config = {
        'feature_names': FEATURE_NAMES,
        'optimal_threshold': float(best_threshold),
        'metrics': best_metrics
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n模型已保存: {model_path}")
    print(f"标准化器: {scaler_path}")
    print(f"配置文件: {config_path}")
    
    return model, scaler, best_threshold


def main():
    parser = argparse.ArgumentParser(description='训练图像篡改检测模型')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./release/models', help='输出目录')
    args = parser.parse_args()
    
    print("=" * 60)
    print("图像篡改检测模型训练")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 步骤1: 构建特征矩阵
    X, y = build_feature_matrix(args.data_dir, args.output_dir)
    
    if X is None:
        return
    
    # 步骤2+3: 训练模型并优化阈值
    train_model(X, y, args.output_dir)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
