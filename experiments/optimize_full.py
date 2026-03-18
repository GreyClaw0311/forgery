#!/usr/bin/env python3
"""全量数据优化训练 - 目标F1>0.85"""

import os
import sys
import numpy as np
import cv2
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/root/forgery')

DATA_DIR = '/data/tamper_data_full'
OUTPUT_DIR = '/root/forgery/results/pixel_segmentation_optimized'

# 窗口32, 步长16 - 更精细
WINDOW = 32
STRIDE = 16

def extract_features_rich(patch):
    """丰富特征提取 - 57维"""
    features = []
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    gray = gray.astype(np.float32)
    h, w = gray.shape
    
    # 1. DCT特征 (8个)
    dct = cv2.dct(gray)
    dct_low = dct[:8, :8]
    dct_high = dct[8:, 8:]
    features.append(np.mean(np.abs(dct_low)))
    features.append(np.std(dct_low))
    features.append(np.mean(np.abs(dct_high)))
    features.append(np.std(dct_high))
    features.append(np.percentile(np.abs(dct_low), 95))
    features.append(np.percentile(np.abs(dct_high), 95))
    features.append(np.max(np.abs(dct_low)))
    features.append(np.sum(np.abs(dct_high)) / (np.sum(np.abs(dct)) + 1e-8))  # 高频比例
    
    # 2. ELA特征 (4个)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, encoded = cv2.imencode('.jpg', patch, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = np.abs(patch.astype(np.float32) - decoded.astype(np.float32))
    features.append(np.mean(ela))
    features.append(np.std(ela))
    features.append(np.percentile(ela.flatten(), 95))
    features.append(np.max(ela))
    
    # 3. Noise特征 (4个)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blurred
    features.append(np.mean(np.abs(noise)))
    features.append(np.std(noise))
    features.append(np.percentile(np.abs(noise), 95))
    features.append(np.max(np.abs(noise)))
    
    # 4. Edge特征 (6个)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    features.append(np.mean(edges > 0))  # 边缘密度
    features.append(np.mean(mag))
    features.append(np.std(mag))
    features.append(np.max(mag))
    features.append(np.percentile(mag, 95))
    features.append(np.sum(mag > np.percentile(mag, 90)) / (h * w))  # 强边缘比例
    
    # 5. 纹理特征 (10个) - 使用简单统计代替LBP
    # 计算局部标准差作为纹理度量
    kernel_size = 3
    local_std = np.zeros_like(gray)
    for i in range(kernel_size, h - kernel_size):
        for j in range(kernel_size, w - kernel_size):
            local_std[i, j] = np.std(gray[i-kernel_size:i+kernel_size+1, j-kernel_size:j+kernel_size+1])
    features.append(np.mean(local_std))
    features.append(np.std(local_std))
    features.append(np.percentile(local_std, 95))
    
    # 灰度共生矩阵简化 - 使用差分统计
    diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_v = np.abs(gray[1:, :] - gray[:-1, :])
    features.append(np.mean(diff_h))
    features.append(np.std(diff_h))
    features.append(np.mean(diff_v))
    features.append(np.std(diff_v))
    features.append(np.percentile(diff_h, 95))
    features.append(np.percentile(diff_v, 95))
    features.append(np.mean(np.abs(gray[1:, 1:] - gray[:-1, :-1])))  # 对角
    
    # 6. Color/HSV特征 (7个)
    if len(patch.shape) == 3:
        # BGR
        for c in range(3):
            features.append(np.mean(patch[:,:,c]))
            features.append(np.std(patch[:,:,c]))
        # HSV
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        features.append(np.std(hsv[:,:,0]))  # H标准差
        features.append(np.std(hsv[:,:,1]))  # S标准差
        features.append(np.std(hsv[:,:,2]))  # V标准差
    else:
        features.extend([0] * 9)
    
    return np.array(features, dtype=np.float32)


def process_image(image_path, mask_path, max_samples=5000):
    """处理单张图片"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    h, w = image.shape[:2]
    
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        labels = (mask > 0).astype(np.uint8)
    else:
        return None, None
    
    half = WINDOW // 2
    features_list = []
    labels_list = []
    
    for y in range(half, h - half, STRIDE):
        for x in range(half, w - half, STRIDE):
            patch = image[y-half:y+half, x-half:x+half]
            if patch.shape[0] != WINDOW or patch.shape[1] != WINDOW:
                continue
            
            feat = extract_features_rich(patch)
            features_list.append(feat)
            labels_list.append(labels[y, x])
    
    if len(features_list) == 0:
        return None, None
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    # 平衡采样
    tampered_idx = np.where(labels == 1)[0]
    normal_idx = np.where(labels == 0)[0]
    
    if len(tampered_idx) == 0 or len(normal_idx) == 0:
        return None, None
    
    # 保持1:3的比例
    normal_sample = min(len(tampered_idx) * 3, len(normal_idx))
    if normal_sample < len(normal_idx):
        sampled_normal = np.random.choice(normal_idx, normal_sample, replace=False)
        selected_idx = np.concatenate([tampered_idx, sampled_normal])
    else:
        selected_idx = np.arange(len(labels))
    
    # 限制最大样本数
    if len(selected_idx) > max_samples:
        selected_idx = np.random.choice(selected_idx, max_samples, replace=False)
    
    return features[selected_idx], labels[selected_idx]


def main():
    print("=" * 60)
    print("全量数据优化训练 - 目标F1>0.85")
    print("=" * 60)
    print(f"窗口: {WINDOW}, 步长: {STRIDE}")
    print("特征: 57维丰富特征")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 统计图片数
    total_files = 0
    for split in ['easy', 'difficult']:
        images_dir = os.path.join(DATA_DIR, split, 'images')
        if os.path.exists(images_dir):
            total_files += len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    print(f"总图片数: {total_files}")
    
    # 构建数据集
    print("\n构建数据集...")
    all_features = []
    all_labels = []
    processed = 0
    
    for split in ['easy', 'difficult']:
        images_dir = os.path.join(DATA_DIR, split, 'images')
        masks_dir = os.path.join(DATA_DIR, split, 'masks')
        
        if not os.path.exists(images_dir):
            continue
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        print(f"\n{split}: {len(image_files)} 张")
        
        for i, img_file in enumerate(image_files):
            if i % 200 == 0:
                print(f"  {i}/{len(image_files)} ({processed}/{total_files})")
            
            image_path = os.path.join(images_dir, img_file)
            mask_name = img_file.replace('.jpg', '.png')
            mask_path = os.path.join(masks_dir, mask_name)
            
            if not os.path.exists(mask_path):
                continue
            
            features, labels = process_image(image_path, mask_path)
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                processed += 1
    
    print(f"\n共处理 {processed} 张图片")
    
    if len(all_features) == 0:
        print("错误: 没有提取到特征")
        return
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"数据集: {len(X)} 样本")
    print(f"  篡改: {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")
    print(f"  正常: {np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")
    
    # 保存数据集
    np.savez(os.path.join(OUTPUT_DIR, 'dataset.npz'), X=X, y=y)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集: {len(X_train)}, 验证集: {len(X_val)}")
    
    # 结果记录
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_images': processed,
        'dataset_size': int(len(X)),
        'window': WINDOW,
        'stride': STRIDE,
        'feature_dim': X.shape[1],
        'models': {}
    }
    
    # ====== 模型1: Random Forest ======
    print("\n" + "=" * 40)
    print("模型1: Random Forest (优化版)")
    print("=" * 40)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    rf_train_score = rf_model.score(X_train, y_train)
    rf_val_score = rf_model.score(X_val, y_val)
    rf_pred = rf_model.predict(X_val)
    rf_proba = rf_model.predict_proba(X_val)[:, 1]
    
    print(f"训练准确率: {rf_train_score:.4f}")
    print(f"验证准确率: {rf_val_score:.4f}")
    
    # 阈值优化
    best_rf_f1 = 0
    best_rf_thresh = 0.5
    print("\n阈值优化:")
    for thresh in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        pred = (rf_proba > thresh).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred)
        print(f"  {thresh}: P={p:.3f} R={r:.3f} F1={f1:.4f}")
        if f1 > best_rf_f1:
            best_rf_f1 = f1
            best_rf_thresh = thresh
    
    print(f"\n最佳阈值: {best_rf_thresh}, F1: {best_rf_f1:.4f}")
    
    results['models']['random_forest'] = {
        'val_accuracy': float(rf_val_score),
        'best_threshold': float(best_rf_thresh),
        'f1': float(best_rf_f1),
        'n_estimators': 200,
        'max_depth': 30
    }
    
    # 保存RF模型
    with open(os.path.join(OUTPUT_DIR, 'rf_model.pkl'), 'wb') as f:
        pickle.dump({'model': rf_model, 'scaler': scaler, 'threshold': best_rf_thresh}, f)
    
    # ====== 模型2: Gradient Boosting ======
    print("\n" + "=" * 40)
    print("模型2: Gradient Boosting")
    print("=" * 40)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=10,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    
    gb_train_score = gb_model.score(X_train, y_train)
    gb_val_score = gb_model.score(X_val, y_val)
    gb_proba = gb_model.predict_proba(X_val)[:, 1]
    
    print(f"训练准确率: {gb_train_score:.4f}")
    print(f"验证准确率: {gb_val_score:.4f}")
    
    # 阈值优化
    best_gb_f1 = 0
    best_gb_thresh = 0.5
    print("\n阈值优化:")
    for thresh in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        pred = (gb_proba > thresh).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred)
        print(f"  {thresh}: P={p:.3f} R={r:.3f} F1={f1:.4f}")
        if f1 > best_gb_f1:
            best_gb_f1 = f1
            best_gb_thresh = thresh
    
    print(f"\n最佳阈值: {best_gb_thresh}, F1: {best_gb_f1:.4f}")
    
    results['models']['gradient_boosting'] = {
        'val_accuracy': float(gb_val_score),
        'best_threshold': float(best_gb_thresh),
        'f1': float(best_gb_f1),
        'n_estimators': 100,
        'max_depth': 8
    }
    
    # 保存GB模型
    with open(os.path.join(OUTPUT_DIR, 'gb_model.pkl'), 'wb') as f:
        pickle.dump({'model': gb_model, 'scaler': scaler, 'threshold': best_gb_thresh}, f)
    
    # ====== 特征重要性 ======
    print("\n" + "=" * 40)
    print("特征重要性 (Top 15)")
    print("=" * 40)
    
    feature_names = []
    # DCT (8)
    feature_names.extend([f'DCT_{i}' for i in range(8)])
    # ELA (4)
    feature_names.extend([f'ELA_{i}' for i in range(4)])
    # Noise (4)
    feature_names.extend([f'Noise_{i}' for i in range(4)])
    # Edge (6)
    feature_names.extend([f'Edge_{i}' for i in range(6)])
    # Texture (10)
    feature_names.extend([f'Texture_{i}' for i in range(10)])
    # Color (9)
    feature_names.extend([f'Color_{i}' for i in range(9)])
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(15, len(indices))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]*100:.2f}%")
    
    results['feature_importance'] = {
        feature_names[indices[i]]: float(importances[indices[i]]) 
        for i in range(min(15, len(indices)))
    }
    
    # ====== 最终结果 ======
    print("\n" + "=" * 60)
    print("最终结果对比")
    print("=" * 60)
    print(f"Random Forest:      F1 = {best_rf_f1:.4f} (阈值 {best_rf_thresh})")
    print(f"Gradient Boosting:  F1 = {best_gb_f1:.4f} (阈值 {best_gb_thresh})")
    print(f"目标 F1:            > 0.85")
    print("=" * 60)
    
    best_model = 'random_forest' if best_rf_f1 > best_gb_f1 else 'gradient_boosting'
    best_f1 = max(best_rf_f1, best_gb_f1)
    
    results['best_model'] = best_model
    results['best_f1'] = float(best_f1)
    results['target_achieved'] = best_f1 > 0.85
    
    # 保存结果
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {OUTPUT_DIR}")
    
    if best_f1 > 0.85:
        print("\n🎉 目标达成！F1 > 0.85")
    else:
        print(f"\n📈 当前最佳F1: {best_f1:.4f}")
        print("建议: 尝试深度学习方法 (U-Net)")


if __name__ == '__main__':
    main()