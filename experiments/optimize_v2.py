#!/usr/bin/env python3
"""高效优化版本 - 目标F1>0.85"""

import os
import sys
import numpy as np
import cv2
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data/tamper_data_full'
OUTPUT_DIR = '/root/forgery/results/optimize_v2'

# 优化参数
WINDOW = 32          # 缩小窗口
STRIDE = 16          # 更精细步长
MAX_IMAGES = 1000    # 限制图片数，加快迭代

def extract_features_optimized(patch):
    """优化特征提取 - 35维关键特征"""
    features = []
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
    gray = gray.astype(np.float32)
    
    # 1. DCT特征 (8个) - JPEG压缩痕迹
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
    features.append(np.sum(np.abs(dct_high)) / (np.sum(np.abs(dct)) + 1e-8))
    
    # 2. ELA特征 (4个) - 错误级别分析
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, encoded = cv2.imencode('.jpg', patch, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = np.abs(patch.astype(np.float32) - decoded.astype(np.float32))
    features.append(np.mean(ela))
    features.append(np.std(ela))
    features.append(np.percentile(ela.flatten(), 95))
    features.append(np.max(ela))
    
    # 3. Noise特征 (4个) - 噪声一致性
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blurred
    features.append(np.mean(np.abs(noise)))
    features.append(np.std(noise))
    features.append(np.percentile(np.abs(noise), 95))
    features.append(np.max(np.abs(noise)))
    
    # 4. Edge特征 (6个) - 边缘特征
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x**2 + sobel_y**2)
    features.append(np.mean(edges > 0))
    features.append(np.mean(mag))
    features.append(np.std(mag))
    features.append(np.max(mag))
    features.append(np.percentile(mag, 95))
    features.append(np.sum(mag > np.percentile(mag, 90)) / mag.size)
    
    # 5. 纹理特征 (8个) - 局部统计
    kernel_size = 3
    local_var = cv2.GaussianBlur(gray**2, (5,5), 0) - blurred**2
    features.append(np.mean(local_var))
    features.append(np.std(local_var))
    
    # 差分统计
    diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_v = np.abs(gray[1:, :] - gray[:-1, :])
    features.append(np.mean(diff_h))
    features.append(np.std(diff_h))
    features.append(np.mean(diff_v))
    features.append(np.std(diff_v))
    features.append(np.percentile(diff_h, 95))
    features.append(np.percentile(diff_v, 95))
    
    # 6. Color/HSV特征 (5个)
    if len(patch.shape) == 3:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        features.append(np.std(hsv[:,:,0]))  # H
        features.append(np.std(hsv[:,:,1]))  # S
        features.append(np.std(hsv[:,:,2]))  # V
        # 颜色一致性
        features.append(np.std(patch[:,:,0]))  # B
        features.append(np.std(patch[:,:,1]))  # G
    else:
        features.extend([0] * 5)
    
    return np.array(features, dtype=np.float32)


def process_image_fast(image_path, mask_path):
    """快速处理单张图片"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    h, w = image.shape[:2]
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))
    labels = (mask > 0).astype(np.uint8)
    
    half = WINDOW // 2
    features_list = []
    labels_list = []
    
    # 随机采样像素点，而不是遍历所有
    np.random.seed(42)
    y_coords = np.random.randint(half, h - half, min(500, (h - 2*half) * (w - 2*half) // STRIDE))
    x_coords = np.random.randint(half, w - half, len(y_coords))
    
    for y, x in zip(y_coords, x_coords):
        patch = image[y-half:y+half, x-half:x+half]
        if patch.shape[0] != WINDOW or patch.shape[1] != WINDOW:
            continue
        
        feat = extract_features_optimized(patch)
        features_list.append(feat)
        labels_list.append(labels[y, x])
    
    if len(features_list) == 0:
        return None, None
    
    return np.array(features_list), np.array(labels_list)


def morphological_postprocess(mask, min_area=100):
    """后处理：形态学操作 + 连通域过滤"""
    kernel = np.ones((3, 3), np.uint8)
    
    # 开运算去除噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 闭运算填充空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 连通域过滤
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = 255
    
    return result


def main():
    print("=" * 60)
    print("高效优化训练 - 目标F1>0.85")
    print("=" * 60)
    print(f"窗口: {WINDOW}x{WINDOW}, 步长: {STRIDE}")
    print(f"特征: 35维, 图片限制: {MAX_IMAGES}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        # 限制图片数量
        if len(image_files) > MAX_IMAGES // 2:
            image_files = image_files[:MAX_IMAGES // 2]
        
        print(f"\n{split}: {len(image_files)} 张")
        
        for i, img_file in enumerate(image_files):
            if i % 100 == 0:
                print(f"  {i}/{len(image_files)}")
            
            image_path = os.path.join(images_dir, img_file)
            mask_name = img_file.replace('.jpg', '.png')
            mask_path = os.path.join(masks_dir, mask_name)
            
            if not os.path.exists(mask_path):
                continue
            
            features, labels = process_image_fast(image_path, mask_path)
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
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集: {len(X_train)}, 验证集: {len(X_val)}")
    
    # ====== 训练 ======
    print("\n" + "=" * 40)
    print("训练 Random Forest (优化版)")
    print("=" * 40)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    print(f"训练准确率: {train_score:.4f}")
    print(f"验证准确率: {val_score:.4f}")
    
    # 阈值优化
    print("\n阈值优化:")
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.75, 0.05):
        pred = (y_proba > thresh).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred)
        print(f"  {thresh:.2f}: P={p:.3f} R={r:.3f} F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    print(f"\n最佳阈值: {best_thresh:.2f}, F1: {best_f1:.4f}")
    
    # 特征重要性
    print("\n特征重要性 Top 10:")
    importances = model.feature_importances_
    feature_names = (
        ['DCT_low_mean', 'DCT_low_std', 'DCT_high_mean', 'DCT_high_std', 
         'DCT_low_p95', 'DCT_high_p95', 'DCT_low_max', 'DCT_high_ratio'] +
        ['ELA_mean', 'ELA_std', 'ELA_p95', 'ELA_max'] +
        ['Noise_mean', 'Noise_std', 'Noise_p95', 'Noise_max'] +
        ['Edge_density', 'Edge_mag_mean', 'Edge_mag_std', 'Edge_mag_max', 
         'Edge_mag_p95', 'Edge_strong_ratio'] +
        ['LocalVar_mean', 'LocalVar_std', 'DiffH_mean', 'DiffH_std', 
         'DiffV_mean', 'DiffV_std', 'DiffH_p95', 'DiffV_p95'] +
        ['H_std', 'S_std', 'V_std', 'B_std', 'G_std']
    )
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(indices))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]*100:.2f}%")
    
    # 保存
    with open(os.path.join(OUTPUT_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'threshold': best_thresh}, f)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_images': processed,
        'dataset_size': int(len(X)),
        'window': WINDOW,
        'stride': STRIDE,
        'feature_dim': 35,
        'val_accuracy': float(val_score),
        'best_threshold': float(best_thresh),
        'f1': float(best_f1),
        'target_achieved': best_f1 > 0.85
    }
    
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"验证准确率: {val_score:.4f}")
    print(f"最佳阈值: {best_thresh:.2f}")
    print(f"像素级 F1: {best_f1:.4f}")
    print(f"目标达成: {'✅ 是' if best_f1 > 0.85 else '❌ 否'}")
    print(f"\n结果已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()