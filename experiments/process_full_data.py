#!/usr/bin/env python3
"""
全量数据处理脚本 - 修正版
"""

import os
import shutil
import re
import cv2

SOURCE_DIR = '/data/my_data'
OUTPUT_DIR = '/data/tamper_data_full'

def setup_directories():
    for split in ['easy', 'difficult', 'good']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'masks'), exist_ok=True)

def copy_file(src_img, src_mask, dst_split, name):
    dst_img = os.path.join(OUTPUT_DIR, dst_split, 'images', name)
    dst_mask = os.path.join(OUTPUT_DIR, dst_split, 'masks', name.replace('.jpg', '.png'))
    
    if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)
    
    if src_mask and os.path.exists(src_mask):
        if src_mask.endswith('.jpg'):
            mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                cv2.imwrite(dst_mask, mask)
        else:
            shutil.copy2(src_mask, dst_mask)

def process_directory(subdir, dst_split, filter_func=None):
    """处理单个目录"""
    print(f"处理 {subdir} -> {dst_split}")
    src_dir = os.path.join(SOURCE_DIR, subdir)
    img_dir = os.path.join(src_dir, 'images')
    mask_dir = os.path.join(src_dir, 'masks')
    
    count = 0
    files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    for f in files:
        # 过滤函数
        if filter_func and not filter_func(f):
            continue
            
        src_img = os.path.join(img_dir, f)
        # 查找对应的mask
        mask_name = f.replace('.jpg', '.png')
        src_mask = os.path.join(mask_dir, mask_name)
        if not os.path.exists(src_mask):
            # 尝试其他扩展名
            mask_name = f.replace('.jpg', '.jpg')
            src_mask = os.path.join(mask_dir, mask_name)
        if not os.path.exists(src_mask):
            src_mask = None
        
        copy_file(src_img, src_mask, dst_split, f)
        count += 1
    
    print(f"  完成: {count}")
    return count

def has_underscore_digit(filename):
    """检查文件名是否有下划线+数字/英文"""
    return bool(re.search(r'_[0-9a-zA-Z]+\.', filename))

def is_good_file(filename):
    """检查是否是good文件"""
    return filename.lower().startswith('good_')

def main():
    print("=" * 60)
    print("全量数据处理")
    print("=" * 60)
    
    setup_directories()
    
    stats = {'easy': 0, 'difficult': 0, 'good': 0}
    
    # === Easy ===
    print("\n--- Easy ---")
    stats['easy'] += process_directory('t-sroie', 'easy')
    stats['easy'] += process_directory('doctamper-fcd', 'easy')
    stats['easy'] += process_directory('doctamper-scd', 'easy')
    stats['easy'] += process_directory('doctamper-testingset', 'easy')
    
    # tamper-id: 有下划线+数字/英文 -> easy
    src_dir = os.path.join(SOURCE_DIR, 'tamper-id')
    img_dir = os.path.join(src_dir, 'images')
    mask_dir = os.path.join(src_dir, 'masks')
    easy_count = 0
    good_count = 0
    for f in os.listdir(img_dir):
        if not f.endswith('.jpg'):
            continue
        src_img = os.path.join(img_dir, f)
        mask_name = f.replace('.jpg', '.png')
        src_mask = os.path.join(mask_dir, mask_name)
        if not os.path.exists(src_mask):
            src_mask = None
        
        if has_underscore_digit(f):
            copy_file(src_img, src_mask, 'easy', f)
            easy_count += 1
        else:
            copy_file(src_img, src_mask, 'good', f)
            good_count += 1
    print(f"tamper-id -> easy: {easy_count}, -> good: {good_count}")
    stats['easy'] += easy_count
    stats['good'] += good_count
    
    # === Difficult ===
    print("\n--- Difficult ---")
    stats['difficult'] += process_directory('competition_data', 'difficult')
    stats['difficult'] += process_directory('season3_data', 'difficult')
    
    # RTM: 非good -> difficult, good -> good
    src_dir = os.path.join(SOURCE_DIR, 'RTM')
    img_dir = os.path.join(src_dir, 'images')
    mask_dir = os.path.join(src_dir, 'masks')
    difficult_count = 0
    good_count = 0
    for f in os.listdir(img_dir):
        if not f.endswith('.jpg'):
            continue
        src_img = os.path.join(img_dir, f)
        mask_name = f.replace('.jpg', '.png')
        src_mask = os.path.join(mask_dir, mask_name)
        if not os.path.exists(src_mask):
            src_mask = None
        
        if is_good_file(f):
            copy_file(src_img, src_mask, 'good', f)
            good_count += 1
        else:
            copy_file(src_img, src_mask, 'difficult', f)
            difficult_count += 1
    print(f"RTM -> difficult: {difficult_count}, -> good: {good_count}")
    stats['difficult'] += difficult_count
    stats['good'] += good_count
    
    # === 统计 ===
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"Easy: {stats['easy']}")
    print(f"Difficult: {stats['difficult']}")
    print(f"Good: {stats['good']}")
    print(f"总计: {sum(stats.values())}")
    
    # 验证
    print("\n验证:")
    for split in ['easy', 'difficult', 'good']:
        img_count = len(os.listdir(os.path.join(OUTPUT_DIR, split, 'images')))
        mask_count = len(os.listdir(os.path.join(OUTPUT_DIR, split, 'masks')))
        print(f"  {split}: {img_count} images, {mask_count} masks")

if __name__ == '__main__':
    main()