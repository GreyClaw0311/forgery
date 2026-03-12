"""
步骤3.5.1: 构建全量特征矩阵 (分批处理版)
每处理500张图片保存一次checkpoint
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '/tmp/forgery/src')

DATA_DIR = '/tmp/forgery/tamper_data_full/processed'
OUTPUT_DIR = '/tmp/forgery/results/full'
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')
BATCH_SIZE = 500

# 特征列表
FEATURE_NAMES = [
    'ela', 'dct', 'cfa', 'noise', 'edge', 'lbp', 'histogram', 'sift',
    'fft', 'metadata', 'hog', 'color', 'adjacency', 'wavelet', 'gradient',
    'block_dct', 'jpeg_ghost', 'local_noise', 'resampling', 'contrast',
    'blur', 'saturation', 'splicing', 'jpeg_block'
]


def load_checkpoint():
    """加载checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed': [], 'X': [], 'y': []}


def save_checkpoint(cp):
    """保存checkpoint"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(cp, f)


def extract_features_for_image(img_path):
    """提取单张图片的所有特征"""
    features = []
    
    for fname in FEATURE_NAMES:
        try:
            module = __import__(f'features.feature_{fname}', fromlist=[''])
            detect_func = getattr(module, f'detect_tampering_{fname}', None)
            if detect_func:
                _, score = detect_func(img_path)
                features.append(float(score))
            else:
                features.append(0.0)
        except Exception as e:
            features.append(0.0)
    
    return features


def collect_all_samples():
    """收集所有样本"""
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


def main():
    print("=" * 60)
    print("步骤3.5.1: 构建全量特征矩阵 (分批处理)")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 收集样本
    print("\n收集样本...")
    samples = collect_all_samples()
    total = len(samples)
    
    print(f"总样本: {total}")
    print(f"  Easy: {sum(1 for s in samples if s['category']=='easy')}")
    print(f"  Difficult: {sum(1 for s in samples if s['category']=='difficult')}")
    print(f"  Good: {sum(1 for s in samples if s['category']=='good')}")
    
    # 加载checkpoint
    cp = load_checkpoint()
    processed_paths = set(cp['processed'])
    X_list = cp['X']
    y_list = cp['y']
    
    # 过滤已处理的样本
    pending = [s for s in samples if s['path'] not in processed_paths]
    print(f"\n已处理: {len(processed_paths)}, 待处理: {len(pending)}")
    
    if len(pending) == 0:
        print("所有样本已处理完成!")
        X = np.array(X_list)
        y = np.array(y_list)
    else:
        # 分批处理
        start_time = time.time()
        batch_count = 0
        
        for i, sample in enumerate(pending):
            # 提取特征
            features = extract_features_for_image(sample['path'])
            
            X_list.append(features)
            y_list.append(sample['label'])
            processed_paths.add(sample['path'])
            
            # 进度显示
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                eta = (len(pending) - i - 1) / speed if speed > 0 else 0
                print(f"进度: {len(processed_paths)}/{total} ({len(processed_paths)/total*100:.1f}%), 速度: {speed:.2f}张/秒, ETA: {eta:.0f}秒")
            
            # 保存checkpoint
            if (i + 1) % BATCH_SIZE == 0:
                batch_count += 1
                cp['processed'] = list(processed_paths)
                cp['X'] = X_list
                cp['y'] = y_list
                save_checkpoint(cp)
                print(f"  Checkpoint已保存 (批次 {batch_count})")
        
        # 最终保存
        X = np.array(X_list)
        y = np.array(y_list)
    
    print(f"\n特征矩阵: {X.shape}")
    print(f"标签分布: 篡改={np.sum(y==1)}, 正常={np.sum(y==0)}")
    
    # 保存最终结果
    np.savez(
        os.path.join(OUTPUT_DIR, 'feature_matrix.npz'),
        X=X, y=y, feature_names=FEATURE_NAMES
    )
    
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['label'] = y
    df.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index=False)
    
    print(f"\n最终结果已保存到: {OUTPUT_DIR}")
    
    # 删除checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint文件已清理")
    
    return X, y


if __name__ == '__main__':
    X, y = main()