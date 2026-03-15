#!/usr/bin/env python3
"""全量数据训练 - 优化版"""

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

DATA_DIR = '/data/tamper_data_full'
OUTPUT_DIR = '/root/forgery/results/pixel_segmentation_full'

# 简化特征提取器 - 只用快速特征
def extract_fast_features(patch):
    """提取快速特征（不包含慢速LBP）"""
    features = []
    
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    gray = gray.astype(np.float32)
    h, w = gray.shape
    
    # DCT
    dct = cv2.dct(gray)
    features.extend([
        np.mean(np.abs(dct[:8, :8])),
        np.std(dct[:8, :8]),
        np.mean(np.abs(dct[8:, 8:])),
    ])
    
    # Noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blurred
    features.extend([
        np.mean(np.abs(noise)),
        np.std(noise),
        np.percentile(np.abs(noise), 95),
    ])
    
    # Edge
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([
        np.mean(edges > 0),
        np.mean(mag),
        np.std(mag),
    ])
    
    # Color
    if len(patch.shape) == 3:
        for c in range(3):
            features.extend([np.mean(patch[:,:,c]), np.std(patch[:,:,c])])
    else:
        features.extend([0] * 6)
    
    return np.array(features, dtype=np.float32)


def process_image(image_path, mask_path, window=32, stride=16, sample_ratio=0.03):
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
    
    half = window // 2
    features_list = []
    labels_list = []
    
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            patch = image[y-half:y+half, x-half:x+half]
            if patch.shape[0] != window or patch.shape[1] != window:
                continue
            
            feat = extract_fast_features(patch)
            features_list.append(feat)
            labels_list.append(labels[y, x])
    
    if len(features_list) == 0:
        return None, None
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    # 采样
    tampered_idx = np.where(labels == 1)[0]
    normal_idx = np.where(labels == 0)[0]
    
    if len(normal_idx) > len(tampered_idx) * 10:
        sample_size = min(len(tampered_idx) * 10, int(len(normal_idx) * sample_ratio))
        if sample_size > 0:
            sampled_normal = np.random.choice(normal_idx, sample_size, replace=False)
            selected_idx = np.concatenate([tampered_idx, sampled_normal])
        else:
            selected_idx = np.arange(len(labels))
    else:
        selected_idx = np.arange(len(labels))
    
    return features[selected_idx], labels[selected_idx]


def main():
    print("=" * 60)
    print("全量数据像素级分割训练")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 构建数据集
    print("\n构建数据集...")
    all_features = []
    all_labels = []
    
    total_images = 0
    for split in ['easy', 'difficult']:
        images_dir = os.path.join(DATA_DIR, split, 'images')
        masks_dir = os.path.join(DATA_DIR, split, 'masks')
        
        if not os.path.exists(images_dir):
            continue
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        print(f"\n{split}: {len(image_files)} 张")
        
        for i, img_file in enumerate(image_files):
            if i % 50 == 0:
                print(f"  {i}/{len(image_files)}")
            
            image_path = os.path.join(images_dir, img_file)
            mask_name = img_file.replace('.jpg', '.png')
            mask_path = os.path.join(masks_dir, mask_name)
            
            if not os.path.exists(mask_path):
                continue
            
            features, labels = process_image(image_path, mask_path)
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                total_images += 1
    
    print(f"\n共处理 {total_images} 张图片")
    
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
    
    # 训练
    print("\n训练模型...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"  训练准确率: {train_score:.4f}")
    print(f"  验证准确率: {val_score:.4f}")
    
    y_pred = model.predict(X_val)
    print("\n分类报告:")
    print(classification_report(y_val, y_pred, target_names=['正常', '篡改']))
    
    # 阈值优化
    y_proba = model.predict_proba(X_val)[:, 1]
    print("\n阈值优化:")
    
    best_f1 = 0
    best_thresh = 0.5
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (y_proba > thresh).astype(int)
        tp = np.sum((pred==1) & (y_val==1))
        fp = np.sum((pred==1) & (y_val==0))
        fn = np.sum((pred==0) & (y_val==1))
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        print(f"  {thresh}: P={p:.3f} R={r:.3f} F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    # 保存
    with open(os.path.join(OUTPUT_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_images': total_images,
        'dataset_size': int(len(X)),
        'val_accuracy': float(val_score),
        'best_threshold': float(best_thresh),
        'pixel_f1': float(best_f1)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"验证准确率: {val_score:.4f}")
    print(f"最佳阈值: {best_thresh}")
    print(f"像素级 F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()