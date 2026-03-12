"""
步骤3.5.1: 构建全量特征矩阵 (优化版)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, '/tmp/forgery/src')

DATA_DIR = '/tmp/forgery/tamper_data_full/processed'
OUTPUT_DIR = '/tmp/forgery/results/full'

# 特征模块
FEATURE_INFO = [
    ('ela', 'feature_ela'),
    ('dct', 'feature_dct'),
    ('cfa', 'feature_cfa'),
    ('noise', 'feature_noise'),
    ('edge', 'feature_edge'),
    ('lbp', 'feature_lbp'),
    ('histogram', 'feature_histogram'),
    ('sift', 'feature_sift'),
    ('fft', 'feature_fft'),
    ('metadata', 'feature_metadata'),
    ('hog', 'feature_hog'),
    ('color', 'feature_color'),
    ('adjacency', 'feature_adjacency'),
    ('wavelet', 'feature_wavelet'),
    ('gradient', 'feature_gradient'),
    ('block_dct', 'feature_block_dct'),
    ('jpeg_ghost', 'feature_jpeg_ghost'),
    ('local_noise', 'feature_local_noise'),
    ('resampling', 'feature_resampling'),
    ('contrast', 'feature_contrast'),
    ('blur', 'feature_blur'),
    ('saturation', 'feature_saturation'),
    ('splicing', 'feature_splicing'),
    ('jpeg_block', 'feature_jpeg_block'),
]

FEATURE_NAMES = [f[0] for f in FEATURE_INFO]


def extract_single_image(args):
    """提取单张图片的特征"""
    img_path, feature_info = args
    
    features = []
    for fname, fmodule_name in feature_info:
        try:
            module = __import__(f'features.{fmodule_name}', fromlist=[''])
            detect_func = getattr(module, f'detect_tampering_{fname}', None)
            if detect_func:
                _, score = detect_func(img_path)
                features.append(float(score))
            else:
                features.append(0.0)
        except:
            features.append(0.0)
    
    return features


def main():
    print("=" * 60)
    print("步骤3.5.1: 构建全量特征矩阵")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 收集样本
    print("\n收集样本...")
    samples = []
    
    # Easy
    easy_dir = os.path.join(DATA_DIR, 'easy/images')
    if os.path.exists(easy_dir):
        for f in sorted(os.listdir(easy_dir))[:100]:  # 先处理100张测试
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append((os.path.join(easy_dir, f), 1, 'easy'))
    
    # Difficult
    diff_dir = os.path.join(DATA_DIR, 'difficult/images')
    if os.path.exists(diff_dir):
        for f in sorted(os.listdir(diff_dir))[:100]:
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append((os.path.join(diff_dir, f), 1, 'difficult'))
    
    # Good
    good_dir = os.path.join(DATA_DIR, 'good')
    if os.path.exists(good_dir):
        for f in sorted(os.listdir(good_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append((os.path.join(good_dir, f), 0, 'good'))
    
    print(f"本次处理样本: {len(samples)}")
    print(f"  篡改: {sum(1 for s in samples if s[1]==1)}")
    print(f"  正常: {sum(1 for s in samples if s[1]==0)}")
    
    # 提取特征
    print(f"\n提取 {len(FEATURE_NAMES)} 个特征...")
    
    X = []
    start_time = time.time()
    
    for i, (img_path, label, cat) in enumerate(samples):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"进度: {i+1}/{len(samples)}, 耗时: {elapsed:.1f}s")
        
        features = extract_single_image((img_path, FEATURE_INFO))
        X.append(features)
    
    X = np.array(X)
    y = np.array([s[1] for s in samples])
    
    print(f"\n特征矩阵: {X.shape}")
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUTPUT_DIR, 'feature_matrix.npz'), X=X, y=y, feature_names=FEATURE_NAMES)
    
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['label'] = y
    df.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index=False)
    
    print(f"保存到: {OUTPUT_DIR}")
    
    return X, y


if __name__ == '__main__':
    X, y = main()