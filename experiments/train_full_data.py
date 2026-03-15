#!/usr/bin/env python3
"""
全量数据像素级篡改分割训练
"""

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

# 配置
DATA_DIR = '/data/tamper_data_full'
OUTPUT_DIR = '/root/forgery/results/pixel_segmentation_full'

# ============================================================
# 特征提取
# ============================================================

class PixelFeatureExtractor:
    def __init__(self, window_size=32, stride=16):
        self.window_size = window_size
        self.stride = stride
    
    def extract_dct_features(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        gray = gray.astype(np.float32)
        dct = cv2.dct(gray)
        return [
            np.mean(np.abs(dct[:8, :8])),
            np.std(dct[:8, :8]),
            np.mean(np.abs(dct[8:, 8:])),
            np.std(dct[8:, 8:]),
            np.sum(dct[:4, :4] ** 2) / (np.sum(dct ** 2) + 1e-8)
        ]
    
    def extract_ela_features(self, patch, quality=90):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', gray, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        diff = cv2.absdiff(gray, decoded)
        return [np.mean(diff), np.std(diff), np.max(diff), np.percentile(diff, 95)]
    
    def extract_noise_features(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        gray = gray.astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        return [np.mean(np.abs(noise)), np.std(noise), np.var(noise), np.percentile(np.abs(noise), 95)]
    
    def extract_edge_features(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        edges = cv2.Canny(gray, 50, 150)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return [np.mean(edges > 0), np.mean(magnitude), np.std(magnitude), np.max(magnitude)]
    
    def extract_lbp_features(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        # 简化LBP
        lbp = np.zeros_like(gray, dtype=np.float32)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                code = 0
                for k, (di, dj) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
                    if gray[i+di, j+dj] >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
        return (hist / (hist.sum() + 1e-8)).tolist()
    
    def extract_color_features(self, patch):
        if len(patch.shape) == 2:
            return [0] * 6
        features = []
        for c in range(3):
            channel = patch[:, :, c]
            features.extend([np.mean(channel), np.std(channel)])
        return features
    
    def extract_all_features(self, patch):
        features = []
        for extractor in [self.extract_dct_features, self.extract_ela_features, 
                         self.extract_noise_features, self.extract_edge_features,
                         self.extract_lbp_features, self.extract_color_features]:
            try:
                f = extractor(patch)
                features.extend(f if isinstance(f, list) else [f])
            except:
                features.extend([0] * 10)
        return np.array(features, dtype=np.float32)


# ============================================================
# 数据集构建
# ============================================================

def extract_features_from_image(image_path, mask_path, extractor, sample_ratio=0.05):
    """从单张图片提取特征"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    h, w = image.shape[:2]
    
    # 加载mask
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        labels = (mask > 0).astype(np.uint8)
    else:
        return None, None
    
    window = extractor.window_size
    stride = extractor.stride
    half = window // 2
    
    features_list = []
    labels_list = []
    
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            patch = image[y-half:y+half, x-half:x+half]
            if patch.shape[0] != window or patch.shape[1] != window:
                continue
            
            feat = extractor.extract_all_features(patch)
            features_list.append(feat)
            labels_list.append(labels[y, x])
    
    if len(features_list) == 0:
        return None, None
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    # 采样处理不平衡
    tampered_idx = np.where(labels == 1)[0]
    normal_idx = np.where(labels == 0)[0]
    
    if len(normal_idx) > len(tampered_idx) * 10:
        sample_size = min(len(tampered_idx) * 10, int(len(normal_idx) * sample_ratio))
        sampled_normal = np.random.choice(normal_idx, sample_size, replace=False)
        selected_idx = np.concatenate([tampered_idx, sampled_normal])
    else:
        selected_idx = np.arange(len(labels))
    
    return features[selected_idx], labels[selected_idx]


def build_dataset(data_dir, output_path):
    """构建数据集"""
    print("构建像素级数据集...")
    
    extractor = PixelFeatureExtractor(window_size=32, stride=16)
    all_features = []
    all_labels = []
    
    for split in ['easy', 'difficult']:
        images_dir = os.path.join(data_dir, split, 'images')
        masks_dir = os.path.join(data_dir, split, 'masks')
        
        if not os.path.exists(images_dir):
            continue
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        print(f"\n{split}: {len(image_files)} 张图片")
        
        for i, img_file in enumerate(image_files):
            if i % 100 == 0:
                print(f"  {i}/{len(image_files)}")
            
            image_path = os.path.join(images_dir, img_file)
            mask_name = img_file.replace('.jpg', '.png')
            mask_path = os.path.join(masks_dir, mask_name)
            
            if not os.path.exists(mask_path):
                continue
            
            features, labels = extract_features_from_image(image_path, mask_path, extractor)
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
    
    if len(all_features) == 0:
        print("错误: 没有提取到任何特征")
        return None, None
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print(f"\n数据集统计:")
    print(f"  总样本: {len(X)}")
    print(f"  篡改: {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
    print(f"  正常: {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, X=X, y=y)
    print(f"  保存到: {output_path}")
    
    return X, y


# ============================================================
# 训练
# ============================================================

def train_model(X, y):
    """训练模型"""
    print("\n训练 Random Forest...")
    
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
        min_samples_leaf=5,
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
    
    # 不同阈值评估
    y_proba = model.predict_proba(X_val)[:, 1]
    print("\n阈值优化:")
    print(f"{'阈值':<10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    
    best_f1 = 0
    best_thresh = 0.5
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
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
            best_thresh = thresh
    
    print(f"\n最佳阈值: {best_thresh}, F1: {best_f1:.4f}")
    
    return model, scaler, best_thresh, best_f1, val_score


def main():
    print("=" * 60)
    print("全量数据像素级篡改分割训练")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 构建数据集
    dataset_path = os.path.join(OUTPUT_DIR, 'dataset.npz')
    if os.path.exists(dataset_path):
        print(f"\n加载已有数据集: {dataset_path}")
        data = np.load(dataset_path)
        X, y = data['X'], data['y']
    else:
        X, y = build_dataset(DATA_DIR, dataset_path)
    
    if X is None:
        return
    
    # 训练
    model, scaler, best_thresh, best_f1, val_score = train_model(X, y)
    
    # 保存
    model_path = os.path.join(OUTPUT_DIR, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"\n模型保存到: {model_path}")
    
    # 结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': DATA_DIR,
        'dataset_size': int(len(X)),
        'tampered_pixels': int(np.sum(y == 1)),
        'normal_pixels': int(np.sum(y == 0)),
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