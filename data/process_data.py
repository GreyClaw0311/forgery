#!/usr/bin/env python3
"""
图像篡改数据处理脚本

功能：
1. 从原始数据目录整理数据
2. 统一文件命名和格式
3. 为good数据集（正常图片）生成全黑mask
4. 划分训练/验证/测试集
5. 生成数据统计报告
"""

import os
import sys
import shutil
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
            'splits': {'train': 0, 'val': 0, 'test': 0},
            'good_images': 0,
            'tampered_images': 0
        }
    
    def scan_source_data(self) -> Dict[str, List[Tuple[str, Optional[str]]]]:
        """扫描源数据目录，收集所有图片和对应mask"""
        data = {}
        
        for category_dir in self.source_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name.lower()
            images_dir = category_dir / 'images'
            masks_dir = category_dir / 'masks'
            
            if category_name == 'good':
                images_dir = category_dir
                masks_dir = None
            
            if not images_dir.exists():
                continue
            
            pairs = []
            is_good_category = category_name == 'good'
            
            for img_file in images_dir.glob('*.jpg'):
                if is_good_category:
                    pairs.append((str(img_file), None))
                else:
                    if masks_dir and masks_dir.exists():
                        mask_file = masks_dir / (img_file.stem + '.png')
                        if mask_file.exists():
                            pairs.append((str(img_file), str(mask_file)))
            
            if pairs:
                data[category_dir.name] = pairs
                self.stats['categories'][category_dir.name] = len(pairs)
                
                if is_good_category:
                    self.stats['good_images'] = len(pairs)
                    print(f"  {category_dir.name}: {len(pairs)} 张 (正常图片)")
                else:
                    self.stats['tampered_images'] += len(pairs)
                    print(f"  {category_dir.name}: {len(pairs)} 张 (篡改图片)")
        
        return data
    
    def split_data(self, data: Dict[str, List[Tuple[str, Optional[str]]]]) -> Dict[str, List[Tuple[str, Optional[str]]]]:
        """划分数据集"""
        splits = {'train': [], 'val': [], 'test': []}
        
        for category, pairs in data.items():
            random.shuffle(pairs)
            
            n_total = len(pairs)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            
            splits['train'].extend(pairs[:n_train])
            splits['val'].extend(pairs[n_train:n_train + n_val])
            splits['test'].extend(pairs[n_train + n_val:])
        
        for split_name in splits:
            random.shuffle(splits[split_name])
            self.stats['splits'][split_name] = len(splits[split_name])
        
        return splits
    
    def copy_and_rename(self, splits: Dict[str, List[Tuple[str, Optional[str]]]]):
        """复制并重命名文件"""
        for split_name, pairs in splits.items():
            images_dir = self.output_dir / split_name / 'images'
            masks_dir = self.output_dir / split_name / 'masks'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"处理{split_name}")):
                new_name = f"{split_name}_{idx:06d}"
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                shutil.copy(img_path, images_dir / (new_name + '.jpg'))
                
                if mask_path is None:
                    mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        mask = np.zeros((h, w), dtype=np.uint8)
                    else:
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask, (w, h))
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
        
        print("\n[1/3] 扫描源数据...")
        data = self.scan_source_data()
        
        if not data:
            print("错误: 未找到有效数据")
            return
        
        print("\n[2/3] 划分数据集...")
        splits = self.split_data(data)
        
        for split_name, pairs in splits.items():
            print(f"  {split_name}: {len(pairs)} 张")
        
        print("\n[3/3] 复制文件...")
        self.copy_and_rename(splits)
        
        stats_file = self.output_dir / 'stats.json'
        self.stats['timestamp'] = datetime.now().isoformat()
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print("\n" + "=" * 60)
        print("处理完成")
        print("=" * 60)
        print(f"总图片数: {self.stats['total_images']}")
        print(f"统计文件: {stats_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图像篡改数据处理')
    parser.add_argument('--source', type=str, required=True, help='源数据目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    
    args = parser.parse_args()
    
    processor = DataProcessor(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    processor.process()


if __name__ == '__main__':
    main()