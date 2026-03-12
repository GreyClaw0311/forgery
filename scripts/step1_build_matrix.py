"""
步骤3.5.1: 构建全量特征矩阵
提取5731张图片的24个特征
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# 数据路径
DATA_DIR = '/tmp/forgery/tamper_data_full/processed'
OUTPUT_DIR = '/tmp/forgery/results/full'

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
except:
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
    FEATURE_MODULES.append(('jpeg_block', feature_jpeg_block))

FEATURE_NAMES = [f[0] for f in FEATURE_MODULES]


def collect_samples():
    """收集所有样本路径"""
    samples = []
    
    # Easy
    easy_dir = os.path.join(DATA_DIR, 'easy/images')
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
    diff_dir = os.path.join(DATA_DIR, 'difficult/images')
    if os.path.exists(diff_dir):
        for f in sorted(os.listdir(diff_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(diff_dir, f),
                    'filename': f,
                    'category': 'difficult',
                    'label': 1
                })
    
    # Good
    good_dir = os.path.join(DATA_DIR, 'good')
    if os.path.exists(good_dir):
        for f in sorted(os.listdir(good_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(good_dir, f),
                    'filename': f,
                    'category': 'good',
                    'label': 0
                })
    
    return samples


def extract_features(samples):
    """提取特征"""
    X = []
    y = []
    
    total = len(samples)
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        # 进度显示
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (total - i - 1) / speed
            print(f"进度: {i+1}/{total} ({(i+1)/total*100:.1f}%), 速度: {speed:.1f}张/秒, ETA: {eta:.0f}秒")
        
        features = []
        for fname, fmodule in FEATURE_MODULES:
            detect_func = getattr(fmodule, f'detect_tampering_{fname}', None)
            if detect_func:
                try:
                    _, score = detect_func(sample['path'])
                    features.append(float(score))
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        X.append(features)
        y.append(sample['label'])
    
    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("步骤3.5.1: 构建全量特征矩阵")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 收集样本
    print("\n收集样本...")
    samples = collect_samples()
    
    easy_count = sum(1 for s in samples if s['category'] == 'easy')
    diff_count = sum(1 for s in samples if s['category'] == 'difficult')
    good_count = sum(1 for s in samples if s['category'] == 'good')
    
    print(f"总样本: {len(samples)}")
    print(f"  Easy (篡改): {easy_count}")
    print(f"  Difficult (篡改): {diff_count}")
    print(f"  Good (正常): {good_count}")
    print(f"  篡改总计: {easy_count + diff_count}")
    
    # 提取特征
    print(f"\n提取 {len(FEATURE_NAMES)} 个特征...")
    X, y = extract_features(samples)
    
    print(f"\n特征矩阵: {X.shape}")
    print(f"标签分布: 篡改={np.sum(y==1)}, 正常={np.sum(y==0)}")
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # NPZ
    np.savez(
        os.path.join(OUTPUT_DIR, 'feature_matrix.npz'),
        X=X, y=y, feature_names=FEATURE_NAMES
    )
    
    # CSV
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['label'] = y
    df.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index=False)
    
    print(f"\n保存到: {OUTPUT_DIR}")
    print(f"  feature_matrix.npz")
    print(f"  feature_matrix.csv")
    
    # 特征统计
    print("\n特征统计:")
    print(f"{'特征':<15} {'最小值':>12} {'最大值':>12} {'均值':>12}")
    print("-" * 55)
    for i, name in enumerate(FEATURE_NAMES):
        print(f"{name:<15} {X[:,i].min():>12.4f} {X[:,i].max():>12.4f} {X[:,i].mean():>12.4f}")
    
    return X, y


if __name__ == '__main__':
    X, y = main()