"""
特征矩阵构建脚本
提取所有24个特征的分数值，构建N×24特征矩阵
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import DATA_DIR, RESULTS_DIR

# 导入所有特征模块
from features import (
    feature_ela, feature_dct, feature_cfa, feature_noise,
    feature_edge, feature_lbp, feature_histogram, feature_sift,
    feature_fft, feature_metadata,
    feature_hog, feature_color, feature_adjacency, feature_wavelet,
    feature_gradient, feature_block_dct, feature_jpeg_ghost, feature_local_noise,
    feature_resampling, feature_contrast, feature_blur, feature_saturation,
    feature_splicing
)

# 检查是否有jpeg_block特征
try:
    from features import feature_jpeg_block
    HAS_JPEG_BLOCK = True
except ImportError:
    HAS_JPEG_BLOCK = False

# 定义所有特征
ALL_FEATURES = [
    # 核心特征
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
    # 变体特征
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
    ALL_FEATURES.append(('jpeg_block', feature_jpeg_block))


def extract_feature_score(feature_name, feature_module, image_path):
    """提取单个特征的分数"""
    detect_func = getattr(feature_module, f'detect_tampering_{feature_name}', None)
    if detect_func is None:
        return None
    
    try:
        is_tampered, score = detect_func(image_path)
        return float(score)
    except Exception as e:
        print(f"  警告: {feature_name} 提取失败 - {e}")
        return None


def build_feature_matrix():
    """构建特征矩阵"""
    print("=" * 60)
    print("构建特征矩阵")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 收集所有图像路径和标签
    samples = []
    
    # Easy数据集 (标签=1)
    easy_dir = os.path.join(DATA_DIR, 'easy', 'images')
    if os.path.exists(easy_dir):
        for img_file in sorted(os.listdir(easy_dir)):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(easy_dir, img_file),
                    'filename': img_file,
                    'category': 'easy',
                    'label': 1  # 篡改
                })
    
    # Difficult数据集 (标签=1)
    difficult_dir = os.path.join(DATA_DIR, 'difficult', 'images')
    if os.path.exists(difficult_dir):
        for img_file in sorted(os.listdir(difficult_dir)):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(difficult_dir, img_file),
                    'filename': img_file,
                    'category': 'difficult',
                    'label': 1  # 篡改
                })
    
    # Good数据集 (标签=0)
    good_dir = os.path.join(DATA_DIR, 'good')
    if os.path.exists(good_dir):
        for img_file in sorted(os.listdir(good_dir)):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(good_dir, img_file),
                    'filename': img_file,
                    'category': 'good',
                    'label': 0  # 正常
                })
    
    print(f"\n总样本数: {len(samples)}")
    print(f"  - Easy (篡改): {sum(1 for s in samples if s['category'] == 'easy')}")
    print(f"  - Difficult (篡改): {sum(1 for s in samples if s['category'] == 'difficult')}")
    print(f"  - Good (正常): {sum(1 for s in samples if s['category'] == 'good')}")
    
    # 提取特征
    feature_names = [f[0] for f in ALL_FEATURES]
    print(f"\n特征数量: {len(feature_names)}")
    print(f"特征列表: {feature_names}")
    
    # 构建特征矩阵
    X = []
    y = []
    metadata = []
    
    print("\n提取特征中...")
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['filename']}...", end=" ")
        
        feature_scores = []
        for feature_name, feature_module in ALL_FEATURES:
            score = extract_feature_score(feature_name, feature_module, sample['path'])
            if score is None:
                score = 0.0
            feature_scores.append(score)
        
        X.append(feature_scores)
        y.append(sample['label'])
        metadata.append({
            'filename': sample['filename'],
            'category': sample['category'],
            'label': sample['label']
        })
        print("完成")
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"标签分布: 篡改={np.sum(y==1)}, 正常={np.sum(y==0)}")
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    df['category'] = [m['category'] for m in metadata]
    df['filename'] = [m['filename'] for m in metadata]
    
    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 保存为CSV
    csv_path = os.path.join(RESULTS_DIR, 'feature_matrix.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n特征矩阵已保存: {csv_path}")
    
    # 保存为NPZ
    npz_path = os.path.join(RESULTS_DIR, 'feature_matrix.npz')
    np.savez(npz_path, X=X, y=y, feature_names=feature_names)
    print(f"NPZ格式已保存: {npz_path}")
    
    # 保存元数据
    meta_path = os.path.join(RESULTS_DIR, 'feature_matrix_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(samples),
            'num_features': len(feature_names),
            'feature_names': feature_names,
            'label_distribution': {
                'tampered': int(np.sum(y==1)),
                'normal': int(np.sum(y==0))
            },
            'samples': metadata,
            'created_at': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    print(f"元数据已保存: {meta_path}")
    
    # 打印特征统计
    print("\n" + "=" * 60)
    print("特征统计 (按类别)")
    print("=" * 60)
    
    tampered_mask = y == 1
    normal_mask = y == 0
    
    print(f"\n{'特征':<15} {'篡改均值':>12} {'篡改std':>12} {'正常均值':>12} {'正常std':>12} {'差异':>10}")
    print("-" * 75)
    
    for i, fname in enumerate(feature_names):
        tampered_mean = np.mean(X[tampered_mask, i])
        tampered_std = np.std(X[tampered_mask, i])
        normal_mean = np.mean(X[normal_mask, i])
        normal_std = np.std(X[normal_mask, i])
        diff = tampered_mean - normal_mean
        
        print(f"{fname:<15} {tampered_mean:>12.4f} {tampered_std:>12.4f} {normal_mean:>12.4f} {normal_std:>12.4f} {diff:>10.4f}")
    
    return X, y, feature_names, df


if __name__ == '__main__':
    X, y, feature_names, df = build_feature_matrix()