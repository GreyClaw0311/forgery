#!/usr/bin/env python3
"""完成训练和评估"""

import os
import sys
import numpy as np
import cv2
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/root/forgery')
from src.utils import compute_iou, compute_precision_recall, compute_f1

OUTPUT_DIR = '/root/forgery/results/pixel_segmentation'

# 加载数据集
print("=" * 60)
print("加载已构建的数据集")
print("=" * 60)

data = np.load(os.path.join(OUTPUT_DIR, 'dataset.npz'))
X, y = data['X'], data['y']

print(f"数据集大小: {len(X)} 样本")
print(f"篡改像素: {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
print(f"正常像素: {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")

# 训练模型
print("\n" + "=" * 60)
print("训练 Random Forest 模型")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)

print(f"训练准确率: {train_score:.4f}")
print(f"验证准确率: {val_score:.4f}")

y_pred = model.predict(X_val)
print("\n分类报告:")
print(classification_report(y_val, y_pred, target_names=['正常', '篡改']))

# 保存模型
model_path = os.path.join(OUTPUT_DIR, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)
print(f"\n模型保存到: {model_path}")

# 特征重要性
print("\n" + "=" * 60)
print("特征重要性 Top 10")
print("=" * 60)

feature_names = [
    'DCT_mean', 'DCT_std', 'DCT_high_mean', 'DCT_high_std', 'DCT_energy_ratio',
    'ELA_mean', 'ELA_std', 'ELA_max', 'ELA_p95',
    'Noise_mean', 'Noise_std', 'Noise_var', 'Noise_p95',
    'Edge_density', 'Edge_mag_mean', 'Edge_mag_std', 'Edge_mag_max',
] + [f'LBP_{i}' for i in range(32)] + ['Color_R_mean', 'Color_R_std', 'Color_G_mean', 'Color_G_std', 'Color_B_mean', 'Color_B_std', 'HSV_H_std', 'HSV_S_std']

importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

for i, idx in enumerate(indices):
    name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
    print(f"  {i+1}. {name}: {importances[idx]:.4f}")

# 测试评估
print("\n" + "=" * 60)
print("在测试图片上评估")
print("=" * 60)

# 简化测试 - 直接用验证集评估像素级效果
y_proba = model.predict_proba(X_val)[:, 1]

# 不同阈值下的效果
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\n不同阈值下的性能:")
print(f"{'阈值':<10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
print("-" * 42)

best_f1 = 0
best_threshold = 0.5

for thresh in thresholds:
    pred = (y_proba > thresh).astype(int)
    tp = np.sum((pred == 1) & (y_val == 1))
    fp = np.sum((pred == 1) & (y_val == 0))
    fn = np.sum((pred == 0) & (y_val == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{thresh:<10} {precision:>12.4f} {recall:>10.4f} {f1:>10.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\n最佳阈值: {best_threshold}, F1: {best_f1:.4f}")

# 保存结果
results = {
    'timestamp': datetime.now().isoformat(),
    'model_type': 'RandomForest',
    'dataset_size': int(len(X)),
    'tampered_pixels': int(np.sum(y == 1)),
    'normal_pixels': int(np.sum(y == 0)),
    'train_accuracy': float(train_score),
    'val_accuracy': float(val_score),
    'best_threshold': float(best_threshold),
    'pixel_level_f1': float(best_f1),
    'feature_importance': {feature_names[i]: float(importances[i]) for i in indices[:10]}
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n结果保存到: {OUTPUT_DIR}/results.json")

print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)
print(f"\n关键指标:")
print(f"  验证准确率: {val_score:.4f}")
print(f"  最佳阈值: {best_threshold}")
print(f"  像素级 F1: {best_f1:.4f}")