"""
步骤3.5.1: 构建全量特征矩阵 (优化版)
- 仅使用Top 10特征
- 实时输出进度
- 内存优化
"""

import os
import sys
import gc
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '/tmp/forgery/src')

DATA_DIR = '/tmp/forgery/tamper_data_full/processed'
OUTPUT_DIR = '/tmp/forgery/results/full'
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')

# Top 10 特征 (基于特征重要性)
TOP_FEATURES = [
    'jpeg_block',   # 9.47%
    'contrast',     # 9.21%
    'saturation',   # 8.87%
    'jpeg_ghost',   # 6.25%
    'fft',          # 6.10%
    'cfa',          # 5.66%
    'edge',         # 4.88%
    'color',        # 4.27%
    'resampling',   # 4.06%
    'splicing',     # 4.02%
]


def extract_features_for_image(img_path):
    """提取单张图片的特征"""
    features = []
    
    for fname in TOP_FEATURES:
        try:
            module = __import__(f'features.feature_{fname}', fromlist=[''])
            detect_func = getattr(module, f'detect_tampering_{fname}', None)
            if detect_func:
                _, score = detect_func(img_path)
                features.append(float(score))
            else:
                features.append(0.0)
        except Exception as e:
            print(f"    [警告] {fname} 提取失败: {e}")
            features.append(0.0)
    
    return features


def collect_samples():
    """收集所有样本"""
    samples = []
    
    # Easy
    easy_dir = os.path.join(DATA_DIR, 'easy/images')
    if os.path.exists(easy_dir):
        for f in sorted(os.listdir(easy_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({'path': os.path.join(easy_dir, f), 'label': 1, 'category': 'easy'})
    
    # Difficult
    diff_dir = os.path.join(DATA_DIR, 'difficult/images')
    if os.path.exists(diff_dir):
        for f in sorted(os.listdir(diff_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({'path': os.path.join(diff_dir, f), 'label': 1, 'category': 'difficult'})
    
    # Good
    good_dir = os.path.join(DATA_DIR, 'good')
    if os.path.exists(good_dir):
        for f in sorted(os.listdir(good_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({'path': os.path.join(good_dir, f), 'label': 0, 'category': 'good'})
    
    return samples


def main():
    print("=" * 60, flush=True)
    print("步骤3.5.1: 构建全量特征矩阵 (优化版)", flush=True)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"使用特征: {TOP_FEATURES}", flush=True)
    print("=" * 60, flush=True)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 收集样本
    print("\n收集样本...", flush=True)
    samples = collect_samples()
    total = len(samples)
    
    print(f"总样本: {total}", flush=True)
    print(f"  Easy: {sum(1 for s in samples if s['category']=='easy')}", flush=True)
    print(f"  Difficult: {sum(1 for s in samples if s['category']=='difficult')}", flush=True)
    print(f"  Good: {sum(1 for s in samples if s['category']=='good')}", flush=True)
    
    # 加载checkpoint
    processed = set()
    X_list = []
    y_list = []
    
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                cp = json.load(f)
                processed = set(cp.get('processed', []))
                print(f"\n从checkpoint恢复: 已处理 {len(processed)} 张", flush=True)
        except:
            pass
    
    # 过滤已处理
    pending = [s for s in samples if s['path'] not in processed]
    print(f"待处理: {len(pending)} 张", flush=True)
    
    if len(pending) == 0:
        print("所有样本已处理完成!", flush=True)
        X = np.array(X_list)
        y = np.array(y_list)
    else:
        # 提取特征
        start_time = time.time()
        last_save = 0
        
        for i, sample in enumerate(pending):
            # 提取特征
            features = extract_features_for_image(sample['path'])
            X_list.append(features)
            y_list.append(sample['label'])
            processed.add(sample['path'])
            
            # 进度显示 (每50张)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(pending) - i - 1) / speed if speed > 0 else 0
                print(f"[{i+1}/{len(pending)}] 速度: {speed:.1f}张/秒, ETA: {eta/60:.1f}分钟", flush=True)
            
            # 保存checkpoint (每200张)
            if len(processed) - last_save >= 200:
                cp = {
                    'processed': list(processed),
                    'timestamp': datetime.now().isoformat()
                }
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(cp, f)
                last_save = len(processed)
                print(f"  [Checkpoint] 已保存: {len(processed)} 张", flush=True)
                
                # 强制垃圾回收
                gc.collect()
        
        X = np.array(X_list)
        y = np.array(y_list)
    
    # 最终结果
    print(f"\n特征矩阵: {X.shape}", flush=True)
    print(f"标签分布: 篡改={np.sum(y==1)}, 正常={np.sum(y==0)}", flush=True)
    
    # 保存
    np.savez(
        os.path.join(OUTPUT_DIR, 'feature_matrix.npz'),
        X=X, y=y, feature_names=TOP_FEATURES
    )
    
    df = pd.DataFrame(X, columns=TOP_FEATURES)
    df['label'] = y
    df.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index=False)
    
    print(f"\n保存到: {OUTPUT_DIR}", flush=True)
    print(f"  feature_matrix.npz", flush=True)
    print(f"  feature_matrix.csv", flush=True)
    
    # 删除checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint已清理", flush=True)
    
    # 特征统计
    print("\n特征统计:", flush=True)
    print(f"{'特征':<15} {'最小':>10} {'最大':>10} {'均值':>10}", flush=True)
    print("-" * 50, flush=True)
    for i, name in enumerate(TOP_FEATURES):
        print(f"{name:<15} {X[:,i].min():>10.2f} {X[:,i].max():>10.2f} {X[:,i].mean():>10.2f}", flush=True)
    
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time/60:.1f} 分钟", flush=True)
    
    return X, y


if __name__ == '__main__':
    X, y = main()