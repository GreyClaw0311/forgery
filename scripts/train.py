#!/usr/bin/env python3
"""
图像篡改像素级分割训练脚本

核心功能：
1. 滑动窗口特征提取
2. Random Forest模型训练
3. 阈值优化
4. 模型保存与评估
"""

import os
import sys
import pickle
import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler


# ============== 配置 ==============
class Config:
    """训练配置"""
    # 窗口参数
    WINDOW_SIZE = 32          # 滑动窗口大小
    STRIDE = 16               # 滑动步长
    
    # 采样参数
    MAX_SAMPLES_PER_IMAGE = 5000  # 每张图片最大采样数
    BALANCE_RATIO = 3             # 正常:篡改 采样比例
    
    # 模型参数
    N_ESTIMATORS = 200        # 随机森林树数量
    MAX_DEPTH = 30            # 最大深度
    MIN_SAMPLES_SPLIT = 5     # 分裂最小样本数
    MIN_SAMPLES_LEAF = 2      # 叶子节点最小样本数
    
    # 训练参数
    TEST_SIZE = 0.2           # 验证集比例
    RANDOM_STATE = 42         # 随机种子


# ============== 特征提取 ==============
class FeatureExtractor:
    """像素级特征提取器"""
    
    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self.half = window_size // 2
    
    def extract(self, patch: np.ndarray) -> np.ndarray:
        """
        从图像块提取特征
        
        Args:
            patch: 图像块 (H, W, C) 或 (H, W)
            
        Returns:
            特征向量 (35,)
        """
        features = []
        
        # 转灰度
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        gray = gray.astype(np.float32)
        
        # 1. DCT特征 (8个) - JPEG压缩痕迹
        dct = cv2.dct(gray)
        dct_low = dct[:8, :8]
        dct_high = dct[8:, 8:]
        
        features.append(np.mean(np.abs(dct_low)))
        features.append(np.std(dct_low))
        features.append(np.mean(np.abs(dct_high)))
        features.append(np.std(dct_high))
        features.append(np.percentile(np.abs(dct_low), 95))
        features.append(np.percentile(np.abs(dct_high), 95))
        features.append(np.max(np.abs(dct_low)))
        features.append(np.sum(np.abs(dct_high)) / (np.sum(np.abs(dct)) + 1e-8))
        
        # 2. ELA特征 (4个) - 错误级别分析
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded = cv2.imencode('.jpg', patch, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        ela = np.abs(patch.astype(np.float32) - decoded.astype(np.float32))
        
        features.append(np.mean(ela))
        features.append(np.std(ela))
        features.append(np.percentile(ela.flatten(), 95))
        features.append(np.max(ela))
        
        # 3. Noise特征 (4个) - 噪声一致性
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        
        features.append(np.mean(np.abs(noise)))
        features.append(np.std(noise))
        features.append(np.percentile(np.abs(noise), 95))
        features.append(np.max(np.abs(noise)))
        
        # 4. Edge特征 (6个) - 边缘特征
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.append(np.mean(edges > 0))  # 边缘密度
        features.append(np.mean(mag))
        features.append(np.std(mag))
        features.append(np.max(mag))
        features.append(np.percentile(mag, 95))
        features.append(np.sum(mag > np.percentile(mag, 90)) / mag.size)
        
        # 5. 纹理特征 (8个) - 局部统计
        diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
        diff_v = np.abs(gray[1:, :] - gray[:-1, :])
        
        features.append(np.mean(diff_h))
        features.append(np.std(diff_h))
        features.append(np.mean(diff_v))
        features.append(np.std(diff_v))
        features.append(np.percentile(diff_h, 95))
        features.append(np.percentile(diff_v, 95))
        features.append(np.percentile(np.abs(gray[1:, 1:] - gray[:-1, :-1]), 95))
        features.append(np.std(gray))
        
        # 6. Color/HSV特征 (5个)
        if len(patch.shape) == 3:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            features.append(np.std(hsv[:,:,0]))  # H
            features.append(np.std(hsv[:,:,1]))  # S
            features.append(np.std(hsv[:,:,2]))  # V
            features.append(np.std(patch[:,:,0]))  # B
            features.append(np.std(patch[:,:,1]))  # G
        else:
            features.extend([0] * 5)
        
        return np.array(features, dtype=np.float32)


# ============== 数据处理 ==============
class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.extractor = FeatureExtractor(config.WINDOW_SIZE)
    
    def process_image(self, 
                      image_path: str, 
                      mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单张图片，提取特征和标签
        
        Args:
            image_path: 图片路径
            mask_path: Mask路径
            
        Returns:
            (features, labels)
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        h, w = image.shape[:2]
        
        # 读取Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        
        # 二值化mask (0=正常, 1=篡改)
        labels = (mask > 127).astype(np.uint8)
        
        # 检查是否为正常图片（全黑mask）
        is_normal_image = np.sum(labels) == 0
        
        # 滑动窗口提取特征
        half = self.config.WINDOW_SIZE // 2
        features_list = []
        labels_list = []
        
        for y in range(half, h - half, self.config.STRIDE):
            for x in range(half, w - half, self.config.STRIDE):
                patch = image[y-half:y+half, x-half:x+half]
                if patch.shape[0] != self.config.WINDOW_SIZE or \
                   patch.shape[1] != self.config.WINDOW_SIZE:
                    continue
                
                feat = self.extractor.extract(patch)
                features_list.append(feat)
                labels_list.append(labels[y, x])
        
        if len(features_list) == 0:
            return None, None
        
        features = np.array(features_list)
        labels = np.array(labels_list)
        
        # 平衡采样
        tampered_idx = np.where(labels == 1)[0]
        normal_idx = np.where(labels == 0)[0]
        
        # 处理正常图片（全黑mask，无篡改像素）
        if len(tampered_idx) == 0:
            # 正常图片：只采样正常像素
            if len(normal_idx) == 0:
                return None, None
            # 限制采样数量
            n_samples = min(len(normal_idx), self.config.MAX_SAMPLES_PER_IMAGE)
            selected_idx = np.random.choice(normal_idx, n_samples, replace=False)
            return features[selected_idx], labels[selected_idx]
        
        # 处理篡改图片
        if len(normal_idx) == 0:
            # 只有篡改像素的情况（极少见）
            n_samples = min(len(tampered_idx), self.config.MAX_SAMPLES_PER_IMAGE)
            selected_idx = np.random.choice(tampered_idx, n_samples, replace=False)
            return features[selected_idx], labels[selected_idx]
        
        # 正常情况：保持篡改:正常 = 1:BALANCE_RATIO
        n_tampered = len(tampered_idx)
        n_normal = min(n_tampered * self.config.BALANCE_RATIO, len(normal_idx))
        
        if n_normal < len(normal_idx):
            sampled_normal = np.random.choice(normal_idx, n_normal, replace=False)
            selected_idx = np.concatenate([tampered_idx, sampled_normal])
        else:
            selected_idx = np.arange(len(labels))
        
        # 限制总样本数
        if len(selected_idx) > self.config.MAX_SAMPLES_PER_IMAGE:
            selected_idx = np.random.choice(selected_idx, 
                                            self.config.MAX_SAMPLES_PER_IMAGE, 
                                            replace=False)
        
        return features[selected_idx], labels[selected_idx]
    
    def build_dataset(self, 
                      data_dir: str, 
                      split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        构建数据集
        
        Args:
            data_dir: 数据目录
            split: 数据集划分 (train/val/test)
            
        Returns:
            (X, y)
        """
        data_dir = Path(data_dir)
        images_dir = data_dir / split / 'images'
        masks_dir = data_dir / split / 'masks'
        
        if not images_dir.exists():
            print(f"错误: 目录不存在 {images_dir}")
            return None, None
        
        all_features = []
        all_labels = []
        
        # 统计
        normal_images = 0
        tampered_images = 0
        
        image_files = list(images_dir.glob('*.jpg'))
        print(f"处理 {split} 集: {len(image_files)} 张图片")
        
        for img_file in tqdm(image_files):
            mask_name = img_file.stem + '.png'
            mask_file = masks_dir / mask_name
            
            if not mask_file.exists():
                continue
            
            # 检查是否为正常图片
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            is_normal = mask is not None and np.sum(mask > 127) == 0
            
            if is_normal:
                normal_images += 1
            else:
                tampered_images += 1
            
            features, labels = self.process_image(str(img_file), str(mask_file))
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
        
        if not all_features:
            return None, None
        
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        print(f"\n  数据集统计:")
        print(f"    正常图片: {normal_images} 张")
        print(f"    篡改图片: {tampered_images} 张")
        print(f"    总样本数: {len(X)}")
        print(f"    篡改像素: {np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")
        print(f"    正常像素: {np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")
        
        return X, y


# ============== 训练器 ==============
class Trainer:
    """模型训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
        self.best_threshold = 0.5
        self.feature_importance = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            
        Returns:
            训练结果
        """
        print("\n" + "=" * 40)
        print("开始训练")
        print("=" * 40)
        
        # 标准化
        print("\n标准化特征...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和验证集
        print(f"划分数据集 (验证集 {self.config.TEST_SIZE*100:.0f}%)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, 
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        print(f"  训练集: {len(X_train)}")
        print(f"  验证集: {len(X_val)}")
        
        # 训练模型
        print("\n训练 Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=self.config.MAX_DEPTH,
            min_samples_split=self.config.MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.config.MIN_SAMPLES_LEAF,
            n_jobs=-1,
            random_state=self.config.RANDOM_STATE,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        print(f"\n训练准确率: {train_score:.4f}")
        print(f"验证准确率: {val_score:.4f}")
        
        # 阈值优化
        print("\n阈值优化...")
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in np.arange(0.3, 0.75, 0.05):
            pred = (y_proba > thresh).astype(int)
            f1 = f1_score(y_val, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        self.best_threshold = best_thresh
        
        # 最终评估
        y_pred = (y_proba > self.best_threshold).astype(int)
        
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"\n最佳阈值: {self.best_threshold:.2f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        
        # 特征重要性
        self.feature_importance = self.model.feature_importances_
        
        return {
            'train_accuracy': float(train_score),
            'val_accuracy': float(val_score),
            'best_threshold': float(self.best_threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def save(self, output_path: str):
        """保存模型"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.best_threshold,
            'config': {
                'window_size': self.config.WINDOW_SIZE,
                'stride': self.config.STRIDE,
                'feature_dim': 35
            }
        }
        
        model_file = output_path / 'model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n模型已保存: {model_file}")
        
        return model_file


# ============== 主函数 ==============
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图像篡改像素级分割训练')
    parser.add_argument('--data-dir', type=str, default='/data/tamper_data_full',
                        help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./results/model',
                        help='输出目录')
    parser.add_argument('--window-size', type=int, default=32,
                        help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=16,
                        help='滑动步长')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='随机森林树数量')
    
    args = parser.parse_args()
    
    # 配置
    config = Config()
    config.WINDOW_SIZE = args.window_size
    config.STRIDE = args.stride
    config.N_ESTIMATORS = args.n_estimators
    
    print("=" * 60)
    print("图像篡改像素级分割训练")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"窗口大小: {config.WINDOW_SIZE}")
    print(f"滑动步长: {config.STRIDE}")
    
    # 构建数据集
    print("\n[1/2] 构建数据集...")
    builder = DatasetBuilder(config)
    X, y = builder.build_dataset(args.data_dir, 'train')
    
    if X is None:
        print("错误: 无法构建数据集")
        return
    
    # 训练模型
    print("\n[2/2] 训练模型...")
    trainer = Trainer(config)
    results = trainer.train(X, y)
    
    # 保存模型
    trainer.save(args.output_dir)
    
    # 保存结果
    results['timestamp'] = datetime.now().isoformat()
    results['config'] = {
        'window_size': config.WINDOW_SIZE,
        'stride': config.STRIDE,
        'n_estimators': config.N_ESTIMATORS
    }
    
    results_file = Path(args.output_dir) / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"模型文件: {args.output_dir}/model.pkl")


if __name__ == '__main__':
    main()