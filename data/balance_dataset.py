#!/usr/bin/env python3
"""
数据集平衡脚本

解决训练数据严重不平衡问题:
- 篡改样本: 44,648 (easy=30,648 + difficult=14,000) = 93.5%
- 正常样本: 3,083 (good) = 6.5%
- 原始比例: 14.5:1

支持多种平衡策略:
1. 欠采样: 减少多数类样本
2. 过采样: 复制少数类样本
3. 混合采样: 结合两种方法

使用方法:
    # 查看当前数据分布
    python balance_dataset.py --source /path/to/data --analyze-only
    
    # 欠采样到 2:1 比例
    python balance_dataset.py --source /path/to/data --output /path/to/balanced --strategy undersample --ratio 2
    
    # 混合采样到 3:1 比例
    python balance_dataset.py --source /path/to/data --output /path/to/balanced --strategy hybrid --ratio 3
    
    # 使用增强过采样 (推荐)
    python balance_dataset.py --source /path/to/data --output /path/to/balanced --strategy augment --ratio 3
"""

import os
import sys
import shutil
import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm


class DatasetBalancer:
    """数据集平衡器"""
    
    def __init__(self, 
                 source_dir: str,
                 output_dir: str,
                 strategy: str = 'hybrid',
                 target_ratio: float = 2.0,
                 seed: int = 42):
        """
        初始化平衡器
        
        Args:
            source_dir: 源数据目录 (包含 easy/, difficult/, good/)
            output_dir: 输出目录
            strategy: 平衡策略
                - undersample: 欠采样篡改样本
                - oversample: 过采样正常样本
                - hybrid: 混合采样 (推荐)
                - augment: 增强过采样 (数据增强)
            target_ratio: 目标比例 (篡改:正常)
            seed: 随机种子
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.strategy = strategy
        self.target_ratio = target_ratio
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 数据收集
        self.easy_images = []
        self.difficult_images = []
        self.good_images = []
        
        # 统计信息
        self.stats = {
            'source': str(source_dir),
            'output': str(output_dir),
            'strategy': strategy,
            'target_ratio': target_ratio,
            'original': {},
            'balanced': {},
            'timestamp': None
        }
    
    def scan_source_data(self) -> Dict:
        """扫描源数据"""
        print("=" * 60)
        print("扫描源数据")
        print("=" * 60)
        
        # 扫描 easy/images
        easy_dir = self.source_dir / 'easy' / 'images'
        if easy_dir.exists():
            self.easy_images = list(easy_dir.glob('*.jpg')) + list(easy_dir.glob('*.png'))
            print(f"  easy/images: {len(self.easy_images)} 张 (篡改)")
        
        # 扫描 difficult/images
        diff_dir = self.source_dir / 'difficult' / 'images'
        if diff_dir.exists():
            self.difficult_images = list(diff_dir.glob('*.jpg')) + list(diff_dir.glob('*.png'))
            print(f"  difficult/images: {len(self.difficult_images)} 张 (篡改)")
        
        # 扫描 good/images
        good_dir = self.source_dir / 'good' / 'images'
        if not good_dir.exists():
            good_dir = self.source_dir / 'good'
        if good_dir.exists():
            self.good_images = list(good_dir.glob('*.jpg')) + list(good_dir.glob('*.png'))
            print(f"  good: {len(self.good_images)} 张 (正常)")
        
        total_tampered = len(self.easy_images) + len(self.difficult_images)
        total_good = len(self.good_images)
        total = total_tampered + total_good
        
        self.stats['original'] = {
            'easy': len(self.easy_images),
            'difficult': len(self.difficult_images),
            'tampered': total_tampered,
            'good': total_good,
            'total': total,
            'tampered_ratio': round(total_tampered / total * 100, 2) if total > 0 else 0,
            'good_ratio': round(total_good / total * 100, 2) if total > 0 else 0,
            'tamper_good_ratio': round(total_tampered / total_good, 2) if total_good > 0 else float('inf')
        }
        
        print(f"\n原始数据统计:")
        print(f"  篡改样本: {total_tampered} ({self.stats['original']['tampered_ratio']}%)")
        print(f"  正常样本: {total_good} ({self.stats['original']['good_ratio']}%)")
        print(f"  篡改:正常 = {self.stats['original']['tamper_good_ratio']}:1")
        
        return self.stats['original']
    
    def balance(self) -> Dict:
        """执行平衡策略"""
        print("\n" + "=" * 60)
        print(f"执行平衡策略: {self.strategy}")
        print(f"目标比例 (篡改:正常) = {self.target_ratio}:1")
        print("=" * 60)
        
        total_tampered = len(self.easy_images) + len(self.difficult_images)
        total_good = len(self.good_images)
        
        if self.strategy == 'undersample':
            # 欠采样: 减少篡改样本
            target_tampered = int(total_good * self.target_ratio)
            result = self._undersample(target_tampered)
        
        elif self.strategy == 'oversample':
            # 过采样: 增加正常样本
            target_good = int(total_tampered / self.target_ratio)
            result = self._oversample(target_good)
        
        elif self.strategy == 'hybrid':
            # 混合采样: 适度欠采样 + 适度过采样
            # 目标: 篡改样本不超过正常样本的 target_ratio 倍
            # 同时保持一定的数据量
            
            # 计算平衡点
            # 设最终篡改数为 T, 正常数 G
            # T = G * target_ratio
            # 为保持数据量，取折中方案
            
            # 方案: 
            # - 如果篡改样本过多，欠采样到合理数量
            # - 同时过采样正常样本到一定数量
            
            # 计算: 保持篡改样本数为正常样本的 target_ratio 倍
            # 且尽量保留更多数据
            
            # 选项1: 以正常样本为基准
            # target_good = total_good * 3  # 过采样3倍
            # target_tampered = target_good * self.target_ratio
            
            # 选项2: 以总数据量为基准 (推荐)
            # 保持总数据量约为原来的50%
            target_total = (total_tampered + total_good) // 2
            target_good = int(target_total / (1 + self.target_ratio))
            target_tampered = int(target_good * self.target_ratio)
            
            result = self._hybrid_sample(target_tampered, target_good)
        
        elif self.strategy == 'augment':
            # 增强过采样: 使用数据增强扩展正常样本
            target_good = int(total_tampered / self.target_ratio)
            result = self._augment_oversample(target_good)
        
        elif self.strategy == 'equal':
            # 等量平衡: 每个类别都达到目标数量
            target_count = int(self.target_ratio)  # 这里 ratio 用作目标数量
            result = self._equal_balance(target_count)
        
        else:
            raise ValueError(f"未知策略: {self.strategy}")
        
        self.stats['balanced'] = result
        return result
    
    def _undersample(self, target_tampered: int) -> Dict:
        """欠采样篡改样本"""
        total_tampered = len(self.easy_images) + len(self.difficult_images)
        
        print(f"\n欠采样篡改样本:")
        print(f"  原始: {total_tampered} 张")
        print(f"  目标: {target_tampered} 张")
        print(f"  采样率: {target_tampered/total_tampered*100:.1f}%")
        
        # 按 easy/difficult 比例分配
        easy_ratio = len(self.easy_images) / total_tampered
        target_easy = int(target_tampered * easy_ratio)
        target_diff = target_tampered - target_easy
        
        # 随机采样
        sampled_easy = random.sample(self.easy_images, min(target_easy, len(self.easy_images)))
        sampled_diff = random.sample(self.difficult_images, min(target_diff, len(self.difficult_images)))
        
        print(f"\n采样结果:")
        print(f"  easy: {len(sampled_easy)} 张")
        print(f"  difficult: {len(sampled_diff)} 张")
        print(f"  good: {len(self.good_images)} 张 (不变)")
        print(f"  总计: {len(sampled_easy) + len(sampled_diff) + len(self.good_images)} 张")
        
        return {
            'easy': len(sampled_easy),
            'difficult': len(sampled_diff),
            'tampered': len(sampled_easy) + len(sampled_diff),
            'good': len(self.good_images),
            'total': len(sampled_easy) + len(sampled_diff) + len(self.good_images),
            'tamper_good_ratio': round((len(sampled_easy) + len(sampled_diff)) / len(self.good_images), 2)
        }
    
    def _oversample(self, target_good: int) -> Dict:
        """过采样正常样本"""
        print(f"\n过采样正常样本:")
        print(f"  原始: {len(self.good_images)} 张")
        print(f"  目标: {target_good} 张")
        print(f"  复制倍数: {target_good / len(self.good_images):.1f}x")
        
        total_tampered = len(self.easy_images) + len(self.difficult_images)
        
        print(f"\n采样结果:")
        print(f"  easy: {len(self.easy_images)} 张 (不变)")
        print(f"  difficult: {len(self.difficult_images)} 张 (不变)")
        print(f"  good: {target_good} 张 (过采样)")
        print(f"  总计: {total_tampered + target_good} 张")
        
        return {
            'easy': len(self.easy_images),
            'difficult': len(self.difficult_images),
            'tampered': total_tampered,
            'good': target_good,
            'total': total_tampered + target_good,
            'tamper_good_ratio': round(total_tampered / target_good, 2)
        }
    
    def _hybrid_sample(self, target_tampered: int, target_good: int) -> Dict:
        """混合采样"""
        total_tampered = len(self.easy_images) + len(self.difficult_images)
        
        print(f"\n混合采样:")
        print(f"  篡改样本: {total_tampered} → {target_tampered} (欠采样)")
        print(f"  正常样本: {len(self.good_images)} → {target_good} (过采样)")
        
        # 按 easy/difficult 比例分配
        easy_ratio = len(self.easy_images) / total_tampered
        target_easy = int(target_tampered * easy_ratio)
        target_diff = target_tampered - target_easy
        
        print(f"\n采样结果:")
        print(f"  easy: {target_easy} 张")
        print(f"  difficult: {target_diff} 张")
        print(f"  good: {target_good} 张")
        print(f"  总计: {target_tampered + target_good} 张")
        print(f"  篡改:正常 = {target_tampered}:{target_good} = {target_tampered/target_good:.1f}:1")
        
        return {
            'easy': target_easy,
            'difficult': target_diff,
            'tampered': target_tampered,
            'good': target_good,
            'total': target_tampered + target_good,
            'tamper_good_ratio': round(target_tampered / target_good, 2)
        }
    
    def _augment_oversample(self, target_good: int) -> Dict:
        """增强过采样 - 使用数据增强"""
        print(f"\n增强过采样正常样本:")
        print(f"  原始: {len(self.good_images)} 张")
        print(f"  目标: {target_good} 张")
        
        total_tampered = len(self.easy_images) + len(self.difficult_images)
        
        # 计算需要增强的数量
        copies_needed = target_good - len(self.good_images)
        augment_times = copies_needed / len(self.good_images)
        
        print(f"  需要增强: {copies_needed} 张")
        print(f"  增强倍数: {augment_times:.1f}x")
        
        print(f"\n采样结果:")
        print(f"  easy: {len(self.easy_images)} 张 (不变)")
        print(f"  difficult: {len(self.difficult_images)} 张 (不变)")
        print(f"  good: {target_good} 张 (增强过采样)")
        print(f"  总计: {total_tampered + target_good} 张")
        
        return {
            'easy': len(self.easy_images),
            'difficult': len(self.difficult_images),
            'tampered': total_tampered,
            'good': target_good,
            'total': total_tampered + target_good,
            'tamper_good_ratio': round(total_tampered / target_good, 2)
        }
    
    def _equal_balance(self, target_count: int) -> Dict:
        """
        等量平衡 - 每个类别都调整到目标数量
        
        Args:
            target_count: 每个类别的目标数量
        
        Returns:
            平衡后的统计信息
        """
        print(f"\n等量平衡 (每类 {target_count} 张):")
        
        # 统计采样方式
        easy_action = "欠采样" if len(self.easy_images) > target_count else "过采样"
        diff_action = "欠采样" if len(self.difficult_images) > target_count else "过采样"
        good_action = "欠采样" if len(self.good_images) > target_count else "过采样增强"
        
        print(f"  easy: {len(self.easy_images)} → {target_count} ({easy_action})")
        print(f"  difficult: {len(self.difficult_images)} → {target_count} ({diff_action})")
        print(f"  good: {len(self.good_images)} → {target_count} ({good_action})")
        
        total_tampered = target_count * 2  # easy + difficult
        total = target_count * 3
        
        print(f"\n最终分布:")
        print(f"  easy: {target_count} 张")
        print(f"  difficult: {target_count} 张")
        print(f"  good: {target_count} 张")
        print(f"  总计: {total} 张")
        print(f"  篡改:正常 = {total_tampered}:{target_count} = {total_tampered/target_count:.0f}:1")
        
        return {
            'easy': target_count,
            'difficult': target_count,
            'tampered': total_tampered,
            'good': target_count,
            'total': total,
            'tamper_good_ratio': 2.0
        }
    
    def export_dataset(self):
        """导出平衡后的数据集"""
        print("\n" + "=" * 60)
        print("导出数据集")
        print("=" * 60)
        
        balanced = self.stats['balanced']
        
        # 创建输出目录结构
        for split in ['easy', 'difficult', 'good']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        
        # 计算采样数量
        total_tampered = len(self.easy_images) + len(self.difficult_images)
        easy_ratio = len(self.easy_images) / total_tampered if total_tampered > 0 else 0.5
        
        target_easy = balanced['easy']
        target_diff = balanced['difficult']
        target_good = balanced['good']
        
        # 采样篡改样本
        sampled_easy = random.sample(self.easy_images, min(target_easy, len(self.easy_images)))
        sampled_diff = random.sample(self.difficult_images, min(target_diff, len(self.difficult_images)))
        
        # 复制篡改样本
        print("\n复制篡改样本...")
        self._copy_images(sampled_easy, 'easy')
        self._copy_images(sampled_diff, 'difficult')
        
        # 处理正常样本
        if self.strategy == 'equal':
            # 等量平衡模式
            print(f"\n处理各类别 (目标各 {target_good} 张)...")
            
            # easy: 欠采样或过采样
            self._copy_with_balance(self.easy_images, 'easy', target_easy)
            
            # difficult: 欠采样或过采样
            self._copy_with_balance(self.difficult_images, 'difficult', target_diff)
            
            # good: 增强过采样
            self._copy_with_oversample(self.good_images, 'good', target_good, use_augment=True)
            
        elif self.strategy in ['oversample', 'hybrid', 'augment']:
            # 过采样
            print(f"\n复制正常样本 (过采样到 {target_good} 张)...")
            self._copy_with_oversample(self.good_images, 'good', target_good, use_augment=(self.strategy=='augment'))
        else:
            # 直接复制
            print(f"\n复制正常样本...")
            self._copy_images(self.good_images, 'good')
        
        # 保存统计信息
        self.stats['timestamp'] = datetime.now().isoformat()
        stats_file = self.output_dir / 'balance_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n统计信息已保存: {stats_file}")
        print("\n" + "=" * 60)
        print("导出完成")
        print("=" * 60)
        
        # 打印最终统计
        print(f"\n最终数据分布:")
        print(f"  easy/images: {balanced['easy']} 张")
        print(f"  difficult/images: {balanced['difficult']} 张")
        print(f"  good: {balanced['good']} 张")
        print(f"  总计: {balanced['total']} 张")
        print(f"  篡改:正常 = {balanced['tamper_good_ratio']}:1")
    
    def _copy_images(self, images: List[Path], category: str):
        """复制图片到输出目录"""
        output_dir = self.output_dir / category / 'images'
        
        for img_path in tqdm(images, desc=f"复制 {category}"):
            if img_path.exists():
                shutil.copy2(img_path, output_dir / img_path.name)
    
    def _copy_with_balance(self, images: List[Path], category: str, target_count: int):
        """
        平衡复制 - 支持欠采样和过采样
        
        Args:
            images: 图片路径列表
            category: 类别名称
            target_count: 目标数量
        """
        output_dir = self.output_dir / category / 'images'
        
        if len(images) == 0:
            return
        
        if len(images) >= target_count:
            # 欠采样
            sampled = random.sample(images, target_count)
            print(f"  {category}: 欠采样 {len(images)} → {target_count}")
            for img_path in tqdm(sampled, desc=f"复制 {category}"):
                if img_path.exists():
                    shutil.copy2(img_path, output_dir / img_path.name)
        else:
            # 过采样 (简单复制)
            print(f"  {category}: 过采样 {len(images)} → {target_count}")
            
            # 先复制所有原始图片
            for img_path in tqdm(images, desc=f"复制原始 {category}"):
                if img_path.exists():
                    shutil.copy2(img_path, output_dir / img_path.name)
            
            # 额外复制
            copies_needed = target_count - len(images)
            idx = 0
            while idx < copies_needed:
                for img_path in images:
                    if idx >= copies_needed:
                        break
                    if not img_path.exists():
                        continue
                    
                    stem = img_path.stem
                    suffix = img_path.suffix
                    new_name = f"{stem}_dup{idx:04d}{suffix}"
                    shutil.copy2(img_path, output_dir / new_name)
                    idx += 1
    
    def _copy_with_oversample(self, images: List[Path], category: str, target_count: int, use_augment: bool = False):
        """过采样复制图片"""
        output_dir = self.output_dir / category / 'images' if category != 'good' else self.output_dir / category
        
        if len(images) == 0:
            return
        
        # 先复制所有原始图片
        for img_path in tqdm(images, desc=f"复制原始 {category}"):
            if img_path.exists():
                shutil.copy2(img_path, output_dir / img_path.name)
        
        # 计算需要额外复制的数量
        copies_needed = target_count - len(images)
        if copies_needed <= 0:
            return
        
        # 循环复制/增强
        print(f"  过采样 {copies_needed} 张...")
        
        idx = 0
        while idx < copies_needed:
            for img_path in images:
                if idx >= copies_needed:
                    break
                
                if not img_path.exists():
                    continue
                
                # 生成新文件名
                stem = img_path.stem
                suffix = img_path.suffix
                new_name = f"{stem}_aug{idx:04d}{suffix}"
                
                if use_augment:
                    # 数据增强
                    self._copy_with_augment(img_path, output_dir / new_name)
                else:
                    # 直接复制
                    shutil.copy2(img_path, output_dir / new_name)
                
                idx += 1
    
    def _copy_with_augment(self, src_path: Path, dst_path: Path):
        """复制并增强图片"""
        img = cv2.imread(str(src_path))
        if img is None:
            shutil.copy2(src_path, dst_path)
            return
        
        # 随机选择增强方式
        aug_type = random.choice(['flip_h', 'flip_v', 'rotate', 'brightness', 'none'])
        
        if aug_type == 'flip_h':
            img = cv2.flip(img, 1)
        elif aug_type == 'flip_v':
            img = cv2.flip(img, 0)
        elif aug_type == 'rotate':
            angle = random.choice([90, 180, 270])
            if angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            else:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif aug_type == 'brightness':
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        # 'none' 则不做增强
        
        cv2.imwrite(str(dst_path), img)
    
    def process(self, analyze_only: bool = False):
        """执行完整处理流程"""
        # 扫描数据
        self.scan_source_data()
        
        if analyze_only:
            print("\n[仅分析模式] 不执行平衡操作")
            return
        
        # 平衡数据
        self.balance()
        
        # 导出数据集
        self.export_dataset()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='数据集平衡脚本 - 解决训练数据不平衡问题',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 仅分析数据分布
    python balance_dataset.py --source /path/to/data --analyze-only
    
    # 等量平衡 (推荐) - 每个类别3000张
    python balance_dataset.py --source /path/to/data --output /path/to/balanced \\
        --strategy equal --ratio 3000
    
    # 混合采样到 3:1 比例
    python balance_dataset.py --source /path/to/data --output /path/to/balanced \\
        --strategy hybrid --ratio 3
    
    # 增强过采样到 4:1 比例
    python balance_dataset.py --source /path/to/data --output /path/to/balanced \\
        --strategy augment --ratio 4
        """
    )
    
    parser.add_argument('--source', type=str, required=True, 
                        help='源数据目录 (包含 easy/, difficult/, good/)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认: 源目录_balanced)')
    parser.add_argument('--strategy', type=str, default='equal',
                        choices=['undersample', 'oversample', 'hybrid', 'augment', 'equal'],
                        help='平衡策略 (default: equal)')
    parser.add_argument('--ratio', type=float, default=3000,
                        help='目标比例或数量: ratio策略时为比例(如3表示3:1), equal策略时为每类数量(如3000) (default: 3000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='仅分析数据分布，不执行平衡')
    
    args = parser.parse_args()
    
    # 默认输出目录
    if args.output is None:
        args.output = str(Path(args.source).parent / (Path(args.source).name + '_balanced'))
    
    balancer = DatasetBalancer(
        source_dir=args.source,
        output_dir=args.output,
        strategy=args.strategy,
        target_ratio=args.ratio,
        seed=args.seed
    )
    
    balancer.process(analyze_only=args.analyze_only)


if __name__ == '__main__':
    main()