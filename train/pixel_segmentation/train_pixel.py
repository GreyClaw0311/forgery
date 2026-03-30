#!/usr/bin/env python3
"""
图像篡改像素级分割训练脚本 - 最终版

核心特点:
1. 49维特征 (移除无效LBP特征)
2. GPU加速训练 (XGBoost CUDA)
3. 多进程并行特征提取
4. 数据集缓存 (避免重复构建)
5. 智能采样解决数据不平衡
6. 预设配置 (针对不同场景)
7. 阈值优化 + 连通域后处理评估
8. 实时进度显示

使用方法:
# 推荐配置 (GPU加速)
python train_pixel.py --data_dir /path/to/data --preset balanced --model-type xgb

# 保存数据集缓存
python train_pixel.py --data_dir /path/to/data --cache-dataset

# 从缓存加载
python train_pixel.py --data_dir /path/to/data --load-cache ./cache.npz

# 高召回配置
python train_pixel.py --data_dir /path/to/data --preset high_recall

预设配置:
- default:    基础配置
- balanced:   平衡配置 (推荐)
- high_recall: 高召回配置
- aggressive: 激进配置 (数据严重不平衡)

模型类型:
- xgb:   XGBoost (GPU加速，推荐)
- lgb:   LightGBM (CPU，快速)
- rf:    Random Forest (CPU，稳定)
- ensemble: 集成模型 (加权投票)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import pickle
import json
import time
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 skimage (LBP 加速，但本版本不使用)
try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ML库
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# 更快的模型
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ============== 预设配置 ==============
PRESETS = {
    'default': {
        'MAX_SAMPLES_PER_IMAGE': 3000,
        'TAMPER_OVERSAMPLE_RATIO': 3,
        'NORMAL_UNDERSAMPLE_RATIO': 2,
        'SCALE_POS_WEIGHT': 10,
        'N_ESTIMATORS': 300,
        'LEARNING_RATE': 0.05,
        'NUM_LEAVES': 63,
        'MAX_DEPTH': -1,
        'MIN_CHILD_WEIGHT': 1,
    },
    
    'balanced': {
        'MAX_SAMPLES_PER_IMAGE': 5000,
        'TAMPER_OVERSAMPLE_RATIO': 5,
        'NORMAL_UNDERSAMPLE_RATIO': 1.5,
        'SCALE_POS_WEIGHT': 20,
        'N_ESTIMATORS': 500,
        'LEARNING_RATE': 0.03,
        'NUM_LEAVES': 127,
        'MAX_DEPTH': 12,
        'MIN_CHILD_WEIGHT': 3,
    },
    
    'high_recall': {
        'MAX_SAMPLES_PER_IMAGE': 6000,
        'TAMPER_OVERSAMPLE_RATIO': 8,
        'NORMAL_UNDERSAMPLE_RATIO': 1.2,
        'SCALE_POS_WEIGHT': 30,
        'N_ESTIMATORS': 600,
        'LEARNING_RATE': 0.02,
        'NUM_LEAVES': 127,
        'MAX_DEPTH': 15,
        'MIN_CHILD_WEIGHT': 1,
    },
    
    'aggressive': {
        'MAX_SAMPLES_PER_IMAGE': 8000,
        'TAMPER_OVERSAMPLE_RATIO': 10,
        'NORMAL_UNDERSAMPLE_RATIO': 1,
        'SCALE_POS_WEIGHT': 50,
        'N_ESTIMATORS': 800,
        'LEARNING_RATE': 0.02,
        'NUM_LEAVES': 255,
        'MAX_DEPTH': 15,
        'MIN_CHILD_WEIGHT': 1,
    },
}


# ============== 配置类 ==============
class Config:
    """训练配置"""
    # 窗口参数
    WINDOW_SIZE = 32
    STRIDE = 16
    
    # 采样参数
    MAX_SAMPLES_PER_IMAGE = 3000
    TAMPER_OVERSAMPLE_RATIO = 3
    NORMAL_UNDERSAMPLE_RATIO = 2
    
    # 特征参数 - 固定49维 (无LBP)
    FEATURE_DIM = 49
    
    # 模型参数
    MODEL_TYPE = 'xgb'
    N_ESTIMATORS = 300
    
    # 训练参数
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # 优化参数
    SKIP_SOLID_BLOCKS = True
    SOLID_THRESHOLD = 5.0
    
    # 动态参数
    SCALE_POS_WEIGHT = 10
    LEARNING_RATE = 0.05
    NUM_LEAVES = 63
    MAX_DEPTH = -1
    MIN_CHILD_WEIGHT = 1
    
    def apply_preset(self, preset_name: str):
        """应用预设配置"""
        if preset_name in PRESETS:
            preset = PRESETS[preset_name]
            for key, value in preset.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            return True
        return False
    
    def get_lgb_params(self):
        """获取 LightGBM 参数"""
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': self.NUM_LEAVES,
            'learning_rate': self.LEARNING_RATE,
            'max_depth': self.MAX_DEPTH,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
            'scale_pos_weight': self.SCALE_POS_WEIGHT,
            'min_child_weight': self.MIN_CHILD_WEIGHT,
        }
    
    def get_xgb_params(self):
        """获取 XGBoost 参数 (GPU加速)"""
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': self.MAX_DEPTH if self.MAX_DEPTH > 0 else 10,
            'learning_rate': self.LEARNING_RATE,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'scale_pos_weight': self.SCALE_POS_WEIGHT,
            'min_child_weight': self.MIN_CHILD_WEIGHT,
            # GPU 加速参数
            'tree_method': 'hist',
            'device': 'cuda:0',
        }


# ============== 49维特征提取器 ==============
class FeatureExtractor49D:
    """
    49维特征提取器 (移除无效LBP特征)
    
    特征组成:
    - DCT特征: 8维
    - ELA特征: 4维
    - Noise特征: 6维
    - Edge特征: 6维
    - 纹理特征: 8维
    - Color特征: 5维
    - 频域特征: 6维
    - 局部对比度: 6维
    总计: 49维
    """
    
    def __init__(self, window_size: int = 32, skip_solid: bool = True, solid_threshold: float = 5.0):
        self.window_size = window_size
        self.half = window_size // 2
        self.skip_solid = skip_solid
        self.solid_threshold = solid_threshold
    
    def is_solid_block(self, patch: np.ndarray) -> bool:
        """快速判断是否为纯色块"""
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        return np.std(gray) < self.solid_threshold
    
    def extract(self, patch: np.ndarray) -> np.ndarray:
        """提取49维特征"""
        features = []
        
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        gray = gray.astype(np.float32)
        
        # ===== 1. DCT特征 (8维) =====
        dct = cv2.dct(gray)
        dct_low = dct[:8, :8]
        dct_high = dct[8:, 8:]
        dct_low_abs = np.abs(dct_low)
        dct_high_abs = np.abs(dct_high)
        dct_abs = np.abs(dct)
        
        features.append(np.mean(dct_low_abs))
        features.append(np.std(dct_low_abs))
        features.append(np.mean(dct_high_abs))
        features.append(np.std(dct_high_abs))
        features.append(np.percentile(dct_low_abs, 95))
        features.append(np.percentile(dct_high_abs, 95))
        features.append(np.max(dct_low_abs))
        features.append(np.sum(dct_high_abs) / (np.sum(dct_abs) + 1e-8))
        
        # ===== 2. ELA特征 (4维) =====
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded = cv2.imencode('.jpg', patch, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        ela = np.abs(patch.astype(np.float32) - decoded.astype(np.float32))
        ela_flat = ela.flatten()
        
        features.append(np.mean(ela_flat))
        features.append(np.std(ela_flat))
        features.append(np.percentile(ela_flat, 95))
        features.append(np.max(ela_flat))
        
        # ===== 3. Noise特征 (6维) =====
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        noise_abs = np.abs(noise)
        noise_flat = noise_abs.flatten()
        
        features.append(np.mean(noise_abs))
        features.append(np.std(noise))
        features.append(np.percentile(noise_flat, 95))
        features.append(np.max(noise_abs))
        features.append(np.percentile(noise_flat, 99))
        features.append(np.median(noise_abs))
        
        # ===== 4. Edge特征 (6维) =====
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.append(np.mean(edges > 0))
        features.append(np.mean(mag))
        features.append(np.std(mag))
        features.append(np.max(mag))
        features.append(np.percentile(mag, 95))
        features.append(np.sum(mag > np.percentile(mag, 90)) / mag.size)
        
        # ===== 5. 纹理特征 (8维) =====
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
        
        # ===== 6. Color特征 (5维) =====
        if len(patch.shape) == 3:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            features.append(np.std(hsv[:,:,0]))
            features.append(np.std(hsv[:,:,1]))
            features.append(np.std(hsv[:,:,2]))
            features.append(np.std(patch[:,:,0]))
            features.append(np.std(patch[:,:,1]))
        else:
            features.extend([0] * 5)
        
        # ===== 7. 频域特征 (6维) =====
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        low_freq = magnitude[center_h-radius:center_h+radius, center_w-radius:center_w+radius]
        features.append(np.mean(low_freq))
        features.append(np.std(low_freq))
        
        high_freq_mask = np.ones_like(magnitude, dtype=bool)
        high_freq_mask[center_h-radius:center_h+radius, center_w-radius:center_w+radius] = False
        high_freq = magnitude[high_freq_mask]
        features.append(np.mean(high_freq))
        features.append(np.std(high_freq))
        features.append(np.sum(low_freq) / (np.sum(magnitude) + 1e-8))
        features.append(np.percentile(high_freq, 95))
        
        # ===== 8. 局部对比度特征 (6维) =====
        local_mean = cv2.blur(gray, (8, 8))
        local_var = cv2.blur((gray - local_mean)**2, (8, 8))
        
        features.append(np.mean(local_var))
        features.append(np.std(local_var))
        features.append(np.percentile(local_var, 95))
        
        local_contrast = np.sqrt(local_var)
        features.append(np.mean(local_contrast))
        features.append(np.std(local_contrast))
        features.append(np.percentile(local_contrast, 95))
        
        return np.array(features, dtype=np.float32)


# ============== 智能采样器 ==============
class SmartSampler:
    """智能采样器 - 解决数据不平衡"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def sample(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """智能采样"""
        tampered_idx = np.where(labels == 1)[0]
        normal_idx = np.where(labels == 0)[0]
        
        if len(tampered_idx) == 0 or len(normal_idx) == 0:
            return features, labels
        
        # 篡改像素过采样
        n_tamper = len(tampered_idx)
        n_tamper_oversample = min(
            int(n_tamper * self.config.TAMPER_OVERSAMPLE_RATIO),
            self.config.MAX_SAMPLES_PER_IMAGE // 2
        )
        
        if n_tamper_oversample > n_tamper:
            oversample_times = n_tamper_oversample // n_tamper
            oversample_remainder = n_tamper_oversample % n_tamper
            
            tampered_sampled = tampered_idx.copy()
            for _ in range(oversample_times - 1):
                tampered_sampled = np.concatenate([tampered_sampled, tampered_idx])
            if oversample_remainder > 0:
                extra = np.random.choice(tampered_idx, oversample_remainder, replace=False)
                tampered_sampled = np.concatenate([tampered_sampled, extra])
        else:
            tampered_sampled = np.random.choice(tampered_idx, n_tamper_oversample, replace=False)
        
        # 正常像素欠采样
        n_normal = min(
            len(tampered_sampled) * self.config.NORMAL_UNDERSAMPLE_RATIO,
            len(normal_idx),
            self.config.MAX_SAMPLES_PER_IMAGE - len(tampered_sampled)
        )
        normal_sampled = np.random.choice(normal_idx, int(n_normal), replace=False)
        
        selected_idx = np.concatenate([tampered_sampled, normal_sampled])
        np.random.shuffle(selected_idx)
        
        return features[selected_idx], labels[selected_idx]


# ============== 进度显示器 ==============
class ProgressTracker:
    """实时进度追踪"""
    
    def __init__(self, total: int):
        self.total = total
        self.start_time = time.time()
    
    def print_progress(self, completed: int):
        """打印进度"""
        elapsed = time.time() - self.start_time
        speed = completed / elapsed if elapsed > 0 else 0
        eta = (self.total - completed) / speed if speed > 0 else 0
        
        bar_length = 30
        filled = int(bar_length * completed / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        elapsed_str = f"{elapsed/60:.1f}min" if elapsed < 3600 else f"{elapsed/3600:.1f}h"
        eta_str = f"{eta/60:.1f}min" if eta < 3600 else f"{eta/3600:.1f}h"
        
        print(f"\r  [{bar}] {completed}/{self.total} ({completed/self.total*100:.1f}%) | "
              f"速度: {speed:.1f} 张/s | 已用: {elapsed_str} | 预计: {eta_str}", end='', flush=True)


# ============== 工作进程 ==============
def process_single_image_worker(args):
    """工作进程处理单张图片"""
    image_path, mask_path, config_dict, skip_solid, solid_threshold = args
    
    window_size = config_dict['window_size']
    stride = config_dict['stride']
    max_samples = config_dict['max_samples']
    tamper_ratio = config_dict['tamper_ratio']
    normal_ratio = config_dict['normal_ratio']
    
    extractor = FeatureExtractor49D(window_size, skip_solid, solid_threshold)
    sampler = SmartSampler.__new__(SmartSampler)
    sampler.config = type('Config', (), {
        'MAX_SAMPLES_PER_IMAGE': max_samples,
        'TAMPER_OVERSAMPLE_RATIO': tamper_ratio,
        'NORMAL_UNDERSAMPLE_RATIO': normal_ratio
    })()
    
    image = cv2.imread(image_path)
    if image is None:
        return None, None, False
    
    h, w = image.shape[:2]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None, False
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))
    
    labels = (mask > 127).astype(np.uint8)
    
    half = window_size // 2
    features_list = []
    labels_list = []
    
    for y in range(half, h - half, stride):
        for x in range(half, w - half, stride):
            patch = image[y-half:y+half, x-half:x+half]
            if patch.shape[0] != window_size or patch.shape[1] != window_size:
                continue
            if skip_solid and extractor.is_solid_block(patch):
                continue
            feat = extractor.extract(patch)
            features_list.append(feat)
            labels_list.append(labels[y, x])
    
    if len(features_list) == 0:
        return None, None, False
    
    features = np.array(features_list)
    labels_arr = np.array(labels_list)
    features, labels_arr = sampler.sample(features, labels_arr)
    
    is_tampered = np.sum(labels_arr) > 0
    
    return features, labels_arr, is_tampered


# ============== 数据集构建器 ==============
class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def build_dataset(self, data_dir: str, split: str = 'train', num_workers: int = 8):
        data_dir = Path(data_dir)
        images_dir = data_dir / split / 'images'
        masks_dir = data_dir / split / 'masks'
        
        if not images_dir.exists():
            print(f"错误: 目录不存在 {images_dir}")
            return None, None
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f"\n处理 {split} 集: {len(image_files)} 张图片 ({num_workers} 进程)")
        print(f"窗口: {self.config.WINDOW_SIZE}, 步长: {self.config.STRIDE}")
        print(f"采样: MAX={self.config.MAX_SAMPLES_PER_IMAGE}, "
              f"篡改过采样={self.config.TAMPER_OVERSAMPLE_RATIO}x, "
              f"正常欠采样={self.config.NORMAL_UNDERSAMPLE_RATIO}x")
        print("-" * 60)
        
        tasks = []
        for img_file in image_files:
            mask_name = img_file.stem + '.png'
            mask_file = masks_dir / mask_name
            if not mask_file.exists():
                mask_name = img_file.stem + '.jpg'
                mask_file = masks_dir / mask_name
            if mask_file.exists():
                tasks.append((
                    str(img_file), str(mask_file),
                    {
                        'window_size': self.config.WINDOW_SIZE,
                        'stride': self.config.STRIDE,
                        'max_samples': self.config.MAX_SAMPLES_PER_IMAGE,
                        'tamper_ratio': self.config.TAMPER_OVERSAMPLE_RATIO,
                        'normal_ratio': self.config.NORMAL_UNDERSAMPLE_RATIO
                    },
                    self.config.SKIP_SOLID_BLOCKS,
                    self.config.SOLID_THRESHOLD
                ))
        
        tracker = ProgressTracker(len(tasks))
        all_features = []
        all_labels = []
        normal_count = 0
        tampered_count = 0
        
        start_time = time.time()
        
        with mp.Pool(num_workers) as pool:
            for i, (features, labels, is_tampered) in enumerate(
                pool.imap_unordered(process_single_image_worker, tasks, chunksize=10)
            ):
                if features is not None and len(features) > 0:
                    all_features.append(features)
                    all_labels.append(labels)
                    if is_tampered:
                        tampered_count += 1
                    else:
                        normal_count += 1
                
                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    tracker.print_progress(i + 1)
        
        print()
        
        if not all_features:
            return None, None
        
        print("\n合并数据集...")
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("数据集统计:")
        print(f"{'='*60}")
        print(f"  处理时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        print(f"  正常图片: {normal_count} 张")
        print(f"  篡改图片: {tampered_count} 张")
        print(f"  总样本数: {len(X):,}")
        print(f"  篡改像素: {np.sum(y==1):,} ({np.mean(y==1)*100:.1f}%)")
        print(f"  正常像素: {np.sum(y==0):,} ({np.mean(y==0)*100:.1f}%)")
        print(f"  特征维度: {X.shape[1]}")
        
        return X, y
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, cache_path: str):
        """保存数据集缓存"""
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"数据集缓存已保存: {cache_path}")
    
    def load_dataset(self, cache_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据集缓存"""
        data = np.load(cache_path)
        print(f"数据集缓存已加载: {cache_path}")
        print(f"  样本数: {len(data['X']):,}")
        print(f"  特征维度: {data['X'].shape[1]}")
        return data['X'], data['y']


# ============== 模型训练器 ==============
class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
        self.best_threshold = 0.5
        self.feature_importance = None
        self.models_dict = {}  # 保存多个模型用于 Ensemble
    
    def _create_model(self):
        """创建模型"""
        model_type = self.config.MODEL_TYPE
        
        if model_type == 'lgb' and HAS_LGB:
            print("使用 LightGBM 模型")
            return lgb.LGBMClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                **self.config.get_lgb_params()
            )
        
        elif model_type == 'xgb' and HAS_XGB:
            print("使用 XGBoost 模型 (GPU加速)")
            return xgb.XGBClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                **self.config.get_xgb_params()
            )
        
        elif model_type == 'ensemble' and HAS_LGB and HAS_XGB:
            print("使用集成模型 (XGBoost + LightGBM + RF)")
            lgb_model = lgb.LGBMClassifier(n_estimators=200, **self.config.get_lgb_params())
            xgb_model = xgb.XGBClassifier(n_estimators=200, **self.config.get_xgb_params())
            rf_model = RandomForestClassifier(
                n_estimators=200, max_depth=20, n_jobs=-1, class_weight='balanced'
            )
            return VotingClassifier(
                estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('rf', rf_model)],
                voting='soft'
            )
        
        else:
            print("使用 Random Forest 模型")
            return RandomForestClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                max_depth=30, min_samples_split=5, min_samples_leaf=2,
                n_jobs=-1, class_weight='balanced',
                random_state=self.config.RANDOM_STATE
            )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """训练模型"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        print("\n标准化特征...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"划分数据集 (验证集 {self.config.TEST_SIZE*100:.0f}%)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        print(f"  训练集: {len(X_train):,}")
        print(f"  验证集: {len(X_val):,}")
        
        self.model = self._create_model()
        
        print("\n训练模型...")
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        print(f"训练耗时: {train_time:.1f} 秒")
        
        # 基础评估
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_f1 = f1_score(y_train, train_pred)
        val_f1 = f1_score(y_val, val_pred)
        
        print(f"\n训练集 F1: {train_f1:.4f}")
        print(f"验证集 F1: {val_f1:.4f}")
        
        # 阈值优化
        print("\n阈值优化...")
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_val)[:, 1]
            
            best_f1 = 0
            best_thresh = 0.5
            
            for thresh in np.arange(0.2, 0.9, 0.02):
                pred = (y_proba > thresh).astype(int)
                f1 = f1_score(y_val, pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            self.best_threshold = best_thresh
            y_pred = (y_proba > self.best_threshold).astype(int)
            
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
        else:
            y_pred = val_pred
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
        
        print(f"\n最佳阈值: {self.best_threshold:.2f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        
        # 连通域分析 (与检测框准确率相关)
        self._analyze_connected_components(y_val, y_pred)
        
        self._compute_feature_importance()
        
        return {
            'train_f1': float(train_f1),
            'val_f1': float(val_f1),
            'best_threshold': float(self.best_threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'train_time': train_time
        }
    
    def _analyze_connected_components(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       sample_size: int = 100000):
        """分析连通域 (与检测框准确率相关)
        
        服务端评估使用检测框准确率，需要关注:
        1. 预测的连通域数量
        2. 连通域大小分布
        3. 噪点过滤效果
        """
        # 采样分析 (避免内存问题)
        if len(y_true) > sample_size:
            idx = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sample = y_true[idx]
            y_pred_sample = y_pred[idx]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
        
        # 重塑为2D (假设方形)
        size = int(np.sqrt(len(y_true_sample)))
        if size * size == len(y_true_sample):
            y_true_2d = y_true_sample[:size*size].reshape(size, size)
            y_pred_2d = y_pred_sample[:size*size].reshape(size, size)
            
            # 统计连通域
            true_components = self._count_components(y_true_2d)
            pred_components = self._count_components(y_pred_2d)
            
            print(f"\n连通域分析 (检测框相关):")
            print(f"  真实连通域数: {true_components}")
            print(f"  预测连通域数: {pred_components}")
            
            if pred_components > 0:
                avg_size, noise_count = self._analyze_component_sizes(y_pred_2d)
                print(f"  预测平均大小: {avg_size:.1f}")
                print(f"  小噪点数 (面积<100): {noise_count}")
    
    def _count_components(self, binary: np.ndarray) -> int:
        """统计连通域数量"""
        binary_uint8 = (binary > 0).astype(np.uint8) * 255
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_uint8)
        return num_labels - 1
    
    def _analyze_component_sizes(self, binary: np.ndarray) -> Tuple[float, int]:
        """分析连通域大小"""
        binary_uint8 = (binary > 0).astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_uint8)
        
        sizes = []
        noise_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            sizes.append(area)
            if area < 100:
                noise_count += 1
        
        return np.mean(sizes) if sizes else 0, noise_count
    
    def _compute_feature_importance(self):
        """计算特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            importances = []
            for name, est in self.model.estimators:
                if hasattr(est, 'feature_importances_'):
                    importances.append(est.feature_importances_)
            if importances:
                self.feature_importance = np.mean(importances, axis=0)
    
    def save(self, output_path: str):
        """保存模型"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.best_threshold,
            'feature_importance': self.feature_importance,
            'config': {
                'window_size': self.config.WINDOW_SIZE,
                'stride': self.config.STRIDE,
                'feature_dim': self.config.FEATURE_DIM,
                'model_type': self.config.MODEL_TYPE,
                'use_lbp': False,  # 49维不使用LBP
            }
        }
        
        model_file = output_path / 'model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n模型已保存: {model_file}")
        
        if self.feature_importance is not None:
            self._save_feature_importance(output_path)
        
        return model_file
    
    def _save_feature_importance(self, output_path: Path):
        """保存特征重要性报告"""
        feature_names = [
            'dct_low_mean', 'dct_low_std', 'dct_high_mean', 'dct_high_std',
            'dct_low_p95', 'dct_high_p95', 'dct_low_max', 'dct_high_ratio',
            'ela_mean', 'ela_std', 'ela_p95', 'ela_max',
            'noise_mean', 'noise_std', 'noise_p95', 'noise_max', 'noise_p99', 'noise_median',
            'edge_density', 'edge_mag_mean', 'edge_mag_std', 'edge_mag_max', 'edge_mag_p95', 'edge_strong_ratio',
            'diff_h_mean', 'diff_h_std', 'diff_v_mean', 'diff_v_std',
            'diff_h_p95', 'diff_v_p95', 'diag_p95', 'gray_std',
            'hsv_h_std', 'hsv_s_std', 'hsv_v_std', 'b_std', 'g_std',
            'low_freq_mean', 'low_freq_std', 'high_freq_mean', 'high_freq_std',
            'freq_ratio', 'high_freq_p95',
            'local_var_mean', 'local_var_std', 'local_var_p95',
            'local_contrast_mean', 'local_contrast_std', 'local_contrast_p95'
        ]
        
        importance_data = sorted(
            zip(feature_names, self.feature_importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        report_path = output_path / 'feature_importance.txt'
        with open(report_path, 'w') as f:
            f.write("特征重要性报告 (49维)\n")
            f.write("=" * 50 + "\n\n")
            for i, (name, importance) in enumerate(importance_data, 1):
                f.write(f"{i:2d}. {name:25s} {importance*100:.2f}%\n")
        
        print(f"特征重要性报告: {report_path}")


# ============== 主函数 ==============
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图像篡改像素级分割训练 - 最终版')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./release/models/pixel_segmentation')
    parser.add_argument('--preset', type=str, default='balanced',
                        choices=['default', 'balanced', 'high_recall', 'aggressive'],
                        help='参数预设')
    parser.add_argument('--model_type', type=str, default='xgb',
                        choices=['rf', 'lgb', 'xgb', 'ensemble'],
                        help='模型类型')
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--skip_solid', action='store_true', default=True)
    parser.add_argument('--no_skip_solid', action='store_false', dest='skip_solid')
    parser.add_argument('--cache_dataset', type=str, default=None, help='保存数据集缓存路径')
    parser.add_argument('--load_cache', type=str, default=None, help='加载数据集缓存路径')
    
    args = parser.parse_args()
    
    config = Config()
    config.apply_preset(args.preset)
    config.WINDOW_SIZE = args.window_size
    config.STRIDE = args.stride
    config.MODEL_TYPE = args.model_type
    config.SKIP_SOLID_BLOCKS = args.skip_solid
    
    print("=" * 60)
    print("图像篡改像素级分割训练 - 最终版")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"预设: {args.preset}")
    print(f"模型类型: {config.MODEL_TYPE}")
    print(f"窗口/步长: {config.WINDOW_SIZE}/{config.STRIDE}")
    print(f"特征维度: {config.FEATURE_DIM} (无LBP)")
    print(f"采样: MAX={config.MAX_SAMPLES_PER_IMAGE}, "
          f"篡改={config.TAMPER_OVERSAMPLE_RATIO}x, "
          f"正常={config.NORMAL_UNDERSAMPLE_RATIO}x")
    print(f"类别权重: {config.SCALE_POS_WEIGHT}")
    print(f"进程数: {args.num_workers}")
    
    builder = DatasetBuilder(config)
    
    # 加载或构建数据集
    if args.load_cache and os.path.exists(args.load_cache):
        print(f"\n[1/2] 从缓存加载数据集...")
        X, y = builder.load_dataset(args.load_cache)
    else:
        print(f"\n[1/2] 构建数据集...")
        X, y = builder.build_dataset(args.data_dir, 'train', args.num_workers)
        
        # 保存缓存
        if args.cache_dataset:
            builder.save_dataset(X, y, args.cache_dataset)
    
    if X is None:
        print("错误: 无法构建数据集")
        return
    
    print("\n[2/2] 训练模型...")
    trainer = ModelTrainer(config)
    results = trainer.train(X, y)
    
    trainer.save(args.output_dir)
    
    results['timestamp'] = datetime.now().isoformat()
    results['preset'] = args.preset
    results['config'] = {
        'window_size': config.WINDOW_SIZE,
        'stride': config.STRIDE,
        'model_type': config.MODEL_TYPE,
        'feature_dim': config.FEATURE_DIM,
        'use_lbp': False,
    }
    
    results_file = Path(args.output_dir) / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"最佳阈值: {results['best_threshold']:.2f}")
    print(f"模型文件: {args.output_dir}/model.pkl")


if __name__ == '__main__':
    main()