#!/usr/bin/env python3
"""
图像篡改数据处理脚本

功能：
1. 从原始数据目录整理数据
2. 统一文件命名和格式
3. 划分训练/验证/测试集
4. 生成数据统计报告
"""

import os
import sys
import shutil
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


class DataProcessor:
    """图像篡改数据处理器"""
    
    def __init__(self, 
                 source_dir: str,
                 output_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42):
        """
        初始化数据处理器
        
        Args:
            source_dir: 原始数据目录
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'total_masks': 0,
            'categories': {},
            'splits': {'train': 0, 'val': 0, 'test': 0}
        }
    
    def scan_source_data(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        扫描源数据目录，收集所有图片和对应mask
        
        Returns:
            {category: [(image_path, mask_path), ...]}
        """
        data = {}
        
        # 遍历源目录
        for category_dir in self.source_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            images_dir = category_dir / 'images'
            masks_dir = category_dir / 'masks'
            
            if not images_dir.exists():
                continue
            
            pairs = []
            for img_file in images_dir.glob('*.jpg'):
                # 查找对应的mask
                mask_file = masks_dir / (img_file.stem + '.png')
                if mask_file.exists():
                    pairs.append((str(img_file), str(mask_file)))
                elif masks_dir.exists():
                    # 尝试其他扩展名
                    for ext in ['.png', '.jpg', '.jpeg']:
                        alt_mask = masks_dir / (img_file.stem + ext)
                        if alt_mask.exists():
                            pairs.append((str(img_file), str(alt_mask)))
                            break
            
            if pairs:
                data[category_name] = pairs
                self.stats['categories'][category_name] = len(pairs)
                print(f"  {category_name}: {len(pairs)} 对")
        
        return data
    
    def split_data(self, data: Dict[str, List[Tuple[str, str]]]) -> Dict[str, List[Tuple[str, str]]]:
        """
        划分数据集
        
        Args:
            data: 原始数据
            
        Returns:
            {split: [(image_path, mask_path), ...]}
        """
        all_pairs = []
        for pairs in data.values():
            all_pairs.extend(pairs)
        
        random.shuffle(all_pairs)
        
        n_total = len(all_pairs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        splits = {
            'train': all_pairs[:n_train],
            'val': all_pairs[n_train:n_train + n_val],
            'test': all_pairs[n_train + n_val:]
        }
        
        for split_name, pairs in splits.items():
            self.stats['splits'][split_name] = len(pairs)
        
        return splits
    
    def copy_and_rename(self, splits: Dict[str, List[Tuple[str, str]]]):
        """
        复制并重命名文件到输出目录
        
        Args:
            splits: 划分后的数据
        """
        for split_name, pairs in splits.items():
            images_dir = self.output_dir / split_name / 'images'
            masks_dir = self.output_dir / split_name / 'masks'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"处理{split_name}")):
                # 新文件名: split_000001.jpg
                new_name = f"{split_name}_{idx:06d}"
                
                # 复制图片
                shutil.copy(img_path, images_dir / (new_name + '.jpg'))
                
                # 复制并确保mask为PNG
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # 二值化
                    mask = (mask > 127).astype(np.uint8) * 255
                    cv2.imwrite(str(masks_dir / (new_name + '.png')), mask)
                    self.stats['total_masks'] += 1
                
                self.stats['total_images'] += 1
    
    def process(self):
        """执行完整处理流程"""
        print("=" * 60)
        print("图像篡改数据处理")
        print("=" * 60)
        print(f"源目录: {self.source_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"划分比例: 训练{self.train_ratio*100:.0f}% / 验证{self.val_ratio*100:.0f}% / 测试{self.test_ratio*100:.0f}%")
        
        # 1. 扫描源数据
        print("\n[1/3] 扫描源数据...")
        data = self.scan_source_data()
        
        if not data:
            print("错误: 未找到有效数据")
            return
        
        # 2. 划分数据集
        print("\n[2/3] 划分数据集...")
        splits = self.split_data(data)
        
        for split_name, pairs in splits.items():
            print(f"  {split_name}: {len(pairs)} 张")
        
        # 3. 复制文件
        print("\n[3/3] 复制文件...")
        self.copy_and_rename(splits)
        
        # 4. 保存统计信息
        stats_file = self.output_dir / 'stats.json'
        self.stats['timestamp'] = datetime.now().isoformat()
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # 5. 打印摘要
        print("\n" + "=" * 60)
        print("处理完成")
        print("=" * 60)
        print(f"总图片数: {self.stats['total_images']}")
        print(f"总Mask数: {self.stats['total_masks']}")
        print(f"\n数据集统计:")
        for split, count in self.stats['splits'].items():
            print(f"  {split}: {count}")
        print(f"\n分类统计:")
        for cat, count in self.stats['categories'].items():
            print(f"  {cat}: {count}")
        print(f"\n统计文件: {stats_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='图像篡改数据处理')
    parser.add_argument('--source', type=str, default='/data/my_data',
                        help='源数据目录')
    parser.add_argument('--output', type=str, default='/data/tamper_data_processed',
                        help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    processor = DataProcessor(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    processor.process()


if __name__ == '__main__':
    main()