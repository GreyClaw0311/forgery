"""
数据处理脚本
将全量数据整理为标准格式：
- easy/images/
- difficult/images/
- good/
"""

import os
import shutil
import re

# 路径配置
SOURCE_DIR = '/tmp/forgery/tamper_data_full/my_data'
TARGET_DIR = '/tmp/forgery/tamper_data_full/processed'

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def copy_image(src, dst_dir, name=None):
    """复制图片"""
    if name is None:
        name = os.path.basename(src)
    dst = os.path.join(dst_dir, name)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

def process_t_sroie():
    """处理t-sroie数据 → easy"""
    print("处理 t-sroie...")
    src_dir = os.path.join(SOURCE_DIR, 't-sroie/images')
    dst_dir = os.path.join(TARGET_DIR, 'easy/images')
    ensure_dir(dst_dir)
    
    count = 0
    for f in os.listdir(src_dir):
        if f.endswith(('.jpg', '.png', '.jpeg')):
            # 跳过子目录
            src = os.path.join(src_dir, f)
            if os.path.isfile(src):
                copy_image(src, dst_dir)
                count += 1
    
    print(f"  t-sroie -> easy: {count} 张")
    return count

def process_doctamper():
    """处理DocTamper数据 (doctamper-fcd, doctamper-scd, doctamper-testingset) → easy"""
    print("处理 DocTamper...")
    dst_dir = os.path.join(TARGET_DIR, 'easy/images')
    ensure_dir(dst_dir)
    
    total = 0
    for subdir in ['doctamper-fcd', 'doctamper-scd', 'doctamper-testingset']:
        src_dir = os.path.join(SOURCE_DIR, subdir, 'images')
        if not os.path.exists(src_dir):
            continue
        
        count = 0
        for f in os.listdir(src_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                src = os.path.join(src_dir, f)
                if os.path.isfile(src):
                    # 添加前缀避免重名
                    new_name = f"{subdir}_{f}"
                    copy_image(src, dst_dir, new_name)
                    count += 1
        
        print(f"  {subdir} -> easy: {count} 张")
        total += count
    
    return total

def process_tamper_id():
    """处理tamper-id数据
    - 有下划线+数字/英文的 → easy
    - 无下划线的 → good
    """
    print("处理 tamper-id...")
    src_dir = os.path.join(SOURCE_DIR, 'tamper-id/images')
    easy_dir = os.path.join(TARGET_DIR, 'easy/images')
    good_dir = os.path.join(TARGET_DIR, 'good')
    ensure_dir(easy_dir)
    ensure_dir(good_dir)
    
    easy_count = 0
    good_count = 0
    
    # 匹配有下划线+数字或英文的模式: xxx_1.jpg, xxx_front_gen1.jpg
    pattern = re.compile(r'_\d+\.(jpg|png|jpeg)$|_[a-z]+.*\.(jpg|png|jpeg)$', re.IGNORECASE)
    
    for f in os.listdir(src_dir):
        if not f.endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        src = os.path.join(src_dir, f)
        if not os.path.isfile(src):
            continue
        
        if pattern.search(f):
            # 有下划线+数字/英文 → easy
            copy_image(src, easy_dir, f"tamperid_{f}")
            easy_count += 1
        else:
            # 无下划线 → good
            copy_image(src, good_dir, f"tamperid_{f}")
            good_count += 1
    
    print(f"  tamper-id -> easy: {easy_count} 张")
    print(f"  tamper-id -> good: {good_count} 张")
    return easy_count, good_count

def process_rtm():
    """处理RTM数据
    - good_开头的 → good
    - 其他的 → difficult
    """
    print("处理 RTM...")
    src_dir = os.path.join(SOURCE_DIR, 'RTM/images')
    good_dir = os.path.join(TARGET_DIR, 'good')
    difficult_dir = os.path.join(TARGET_DIR, 'difficult/images')
    ensure_dir(good_dir)
    ensure_dir(difficult_dir)
    
    good_count = 0
    difficult_count = 0
    
    for f in os.listdir(src_dir):
        if not f.endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        src = os.path.join(src_dir, f)
        if not os.path.isfile(src):
            continue
        
        if f.lower().startswith('good_'):
            copy_image(src, good_dir, f"rtm_{f}")
            good_count += 1
        else:
            copy_image(src, difficult_dir, f"rtm_{f}")
            difficult_count += 1
    
    print(f"  RTM -> good: {good_count} 张")
    print(f"  RTM -> difficult: {difficult_count} 张")
    return good_count, difficult_count

def process_competition_and_season3():
    """处理competition_data和season3_data → difficult"""
    print("处理 competition_data 和 season3_data...")
    difficult_dir = os.path.join(TARGET_DIR, 'difficult/images')
    ensure_dir(difficult_dir)
    
    total = 0
    for subdir in ['competition_data', 'season3_data']:
        src_dir = os.path.join(SOURCE_DIR, subdir, 'images')
        if not os.path.exists(src_dir):
            continue
        
        count = 0
        for f in os.listdir(src_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                src = os.path.join(src_dir, f)
                if os.path.isfile(src):
                    new_name = f"{subdir}_{f}"
                    copy_image(src, difficult_dir, new_name)
                    count += 1
        
        print(f"  {subdir} -> difficult: {count} 张")
        total += count
    
    return total

def main():
    """主函数"""
    print("=" * 60)
    print("数据处理脚本")
    print("=" * 60)
    
    # 创建目标目录
    ensure_dir(TARGET_DIR)
    
    # 统计
    stats = {
        'easy': 0,
        'difficult': 0,
        'good': 0
    }
    
    # 处理各类数据
    stats['easy'] += process_t_sroie()
    stats['easy'] += process_doctamper()
    
    easy_from_tamper, good_from_tamper = process_tamper_id()
    stats['easy'] += easy_from_tamper
    stats['good'] += good_from_tamper
    
    good_from_rtm, difficult_from_rtm = process_rtm()
    stats['good'] += good_from_rtm
    stats['difficult'] += difficult_from_rtm
    
    stats['difficult'] += process_competition_and_season3()
    
    # 打印统计
    print("\n" + "=" * 60)
    print("处理完成！统计结果：")
    print("=" * 60)
    print(f"  easy (简单篡改): {stats['easy']} 张")
    print(f"  difficult (复杂篡改): {stats['difficult']} 张")
    print(f"  good (正常): {stats['good']} 张")
    print(f"  总计: {stats['easy'] + stats['difficult'] + stats['good']} 张")
    print(f"\n数据保存到: {TARGET_DIR}")
    
    return stats

if __name__ == '__main__':
    stats = main()