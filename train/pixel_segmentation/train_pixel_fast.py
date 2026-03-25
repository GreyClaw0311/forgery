#!/usr/bin/env python3
"""
图像篡改像素级分割训练脚本 - 极速优化版

优化策略:
1. 实时进度显示 (已完成/总数/预计时间/速度)
2. LBP 向量化 (skimage 替代 Python 循环, 50x 加速)
3. percentile 合并计算 (减少排序次数)
4. imap_unordered 批处理 (减少内存占用)
5. 纯色块跳过 (智能采样)
6. 特征提取向量化

预期提升:
- 构建数据集速度: 5-10x
- 内存占用: 降低 50%
"""

import os
import sys

# 添加项目根目录
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
from multiprocessing import Manager
import warnings
warnings.filterwarnings('ignore')

# 尝试导入快速 LBP
try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("警告: 未安装 skimage, LBP 将使用慢速实现。建议: pip install scikit-image")

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


# ============== 配置 ==============
class Config:
    """训练配置"""
    # 窗口参数
    WINDOW_SIZE = 32
    STRIDE = 16
    
    # 采样参数
    MAX_SAMPLES_PER_IMAGE = 3000
    TAMPER_OVERSAMPLE_RATIO = 3
    NORMAL_UNDERSAMPLE_RATIO = 2
    
    # 特征参数
    FEATURE_DIM = 57
    
    # 模型参数
    MODEL_TYPE = 'lgb'
    N_ESTIMATORS = 300
    
    # LightGBM参数
    LGB_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'scale_pos_weight': 10,
    }
    
    # XGBoost参数
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'scale_pos_weight': 10,
    }
    
    # 训练参数
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # 优化参数
    SKIP_SOLID_BLOCKS = True  # 跳过纯色块
    SOLID_THRESHOLD = 5.0     # 纯色块方差阈值


# ============== 快速特征提取器 ==============
class FastFeatureExtractor:
    """快速特征提取器 - 向量化优化版"""
    
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
        
        # 快速方差计算
        return np.std(gray) < self.solid_threshold
    
    def extract(self, patch: np.ndarray) -> np.ndarray:
        """提取特征 (57维) - 优化版"""
        features = []
        
        # 转换灰度图
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        gray = gray.astype(np.float32)
        
        # ========== 1. DCT特征 (8个) ==========
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
        
        # ========== 2. ELA特征 (4个) - 主要瓶颈 ==========
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded = cv2.imencode('.jpg', patch, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        ela = np.abs(patch.astype(np.float32) - decoded.astype(np.float32))
        ela_flat = ela.flatten()
        
        # 合并 percentile 计算
        ela_p95 = np.percentile(ela_flat, 95)
        features.append(np.mean(ela_flat))
        features.append(np.std(ela_flat))
        features.append(ela_p95)
        features.append(np.max(ela_flat))
        
        # ========== 3. Noise特征 (6个) ==========
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        noise_abs = np.abs(noise)
        noise_flat = noise_abs.flatten()
        
        # 合并 percentile 计算
        noise_p95, noise_p99 = np.percentile(noise_flat, [95, 99])
        features.append(np.mean(noise_abs))
        features.append(np.std(noise))
        features.append(noise_p95)
        features.append(np.max(noise_abs))
        features.append(noise_p99)
        features.append(np.median(noise_abs))
        
        # ========== 4. Edge特征 (6个) ==========
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
        
        # ========== 5. 纹理特征 (8个) ==========
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
        
        # ========== 6. Color特征 (5个) ==========
        if len(patch.shape) == 3:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            features.append(np.std(hsv[:,:,0]))
            features.append(np.std(hsv[:,:,1]))
            features.append(np.std(hsv[:,:,2]))
            features.append(np.std(patch[:,:,0]))
            features.append(np.std(patch[:,:,1]))
        else:
            features.extend([0] * 5)
        
        # ========== 7. LBP特征 (8个) - 快速实现 ==========
        if HAS_SKIMAGE:
            # 使用 skimage 向量化实现 (50x 加速)
            lbp = local_binary_pattern(gray.astype(np.uint8), P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=8, range=(0, 256))
            lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-8)
        else:
            # 慢速备用实现
            lbp = self._compute_lbp_slow(gray.astype(np.uint8))
            lbp_hist, _ = np.histogram(lbp, bins=8, range=(0, 256))
            lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-8)
        features.extend(lbp_hist.tolist())
        
        # ========== 8. 频域特征 (6个) ==========
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
        
        # ========== 9. 局部对比度特征 (6个) ==========
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
    
    def _compute_lbp_slow(self, gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """慢速 LBP 实现 (备用)"""
        h, w = gray.shape
        lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                code = 0
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if gray[x, y] >= center:
                        code |= (1 << p)
                lbp[i - radius, j - radius] = code
        
        return lbp


# ============== 智能采样器 ==============
class SmartSampler:
    """智能采样器"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def sample(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """智能采样解决数据不平衡"""
        tampered_idx = np.where(labels == 1)[0]
        normal_idx = np.where(labels == 0)[0]
        
        if len(tampered_idx) == 0 or len(normal_idx) == 0:
            return features, labels
        
        # 篡改像素过采样
        n_tamper = len(tampered_idx)
        n_tamper_oversample = min(
            int(n_tamper * self.config.TAMPER_OVERSAMPLE_RATIO),
            self.config.MAX_SAMPLES_PER_IMAGE // 3
        )
        
        if n_tamper_oversample > n_tamper:
            oversample_idx = np.random.choice(tampered_idx, 
                                              n_tamper_oversample - n_tamper, 
                                              replace=True)
            tampered_idx = np.concatenate([tampered_idx, oversample_idx])
        
        # 正常像素欠采样
        n_normal = min(
            len(tampered_idx) * self.config.NORMAL_UNDERSAMPLE_RATIO,
            len(normal_idx),
            self.config.MAX_SAMPLES_PER_IMAGE - len(tampered_idx)
        )
        normal_sampled = np.random.choice(normal_idx, n_normal, replace=False)
        
        selected_idx = np.concatenate([tampered_idx, normal_sampled])
        np.random.shuffle(selected_idx)
        
        return features[selected_idx], labels[selected_idx]


# ============== 进度显示器 ==============
class ProgressTracker:
    """实时进度追踪"""
    
    def __init__(self, total: int, update_interval: int = 100):
        self.total = total
        self.count = mp.Value('i', 0)
        self.start_time = mp.Value('d', time.time())
        self.update_interval = update_interval
    
    def update(self, n: int = 1):
        """更新计数"""
        with self.count.get_lock():
            self.count.value += n
    
    def get_progress(self) -> dict:
        """获取当前进度"""
        with self.count.get_lock():
            count = self.count.value
        
        elapsed = time.time() - self.start_time.value
        speed = count / elapsed if elapsed > 0 else 0
        eta = (self.total - count) / speed if speed > 0 else 0
        
        return {
            'completed': count,
            'total': self.total,
            'percent': count / self.total * 100 if self.total > 0 else 0,
            'speed': speed,
            'elapsed': elapsed,
            'eta': eta
        }
    
    def print_progress(self, completed: int, total: int, elapsed: float):
        """打印进度"""
        speed = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / speed if speed > 0 else 0
        
        # 格式化时间
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta)
        
        # 进度条
        bar_length = 30
        filled = int(bar_length * completed / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r  [{bar}] {completed}/{total} ({completed/total*100:.1f}%) | "
              f"速度: {speed:.1f} 张/s | 已用: {elapsed_str} | 预计: {eta_str}", end='', flush=True)
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            return f"{seconds/3600:.1f}h"


def process_single_image_worker(args):
    """工作进程处理单张图片"""
    image_path, mask_path, config_dict, skip_solid, solid_threshold = args
    
    # 重建配置
    window_size = config_dict['window_size']
    stride = config_dict['stride']
    max_samples = config_dict['max_samples']
    tamper_ratio = config_dict['tamper_ratio']
    normal_ratio = config_dict['normal_ratio']
    
    # 创建提取器
    extractor = FastFeatureExtractor(window_size, skip_solid, solid_threshold)
    sampler = SmartSampler.__new__(SmartSampler)
    sampler.config = type('Config', (), {
        'MAX_SAMPLES_PER_IMAGE': max_samples,
        'TAMPER_OVERSAMPLE_RATIO': tamper_ratio,
        'NORMAL_UNDERSAMPLE_RATIO': normal_ratio
    })()
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        return None, None, False
    
    h, w = image.shape[:2]
    
    # 读取 mask
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
            
            # 跳过纯色块
            if skip_solid and extractor.is_solid_block(patch):
                continue
            
            feat = extractor.extract(patch)
            features_list.append(feat)
            labels_list.append(labels[y, x])
    
    if len(features_list) == 0:
        return None, None, False
    
    features = np.array(features_list)
    labels_arr = np.array(labels_list)
    
    # 智能采样
    features, labels_arr = sampler.sample(features, labels_arr)
    
    # 判断是否为篡改图片
    is_tampered = np.sum(labels_arr) > 0
    
    return features, labels_arr, is_tampered


# ============== 数据集构建器 ==============
class FastDatasetBuilder:
    """快速数据集构建器"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def build_dataset(self, data_dir: str, split: str = 'train', 
                      num_workers: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """构建数据集 (带实时进度)"""
        data_dir = Path(data_dir)
        images_dir = data_dir / split / 'images'
        masks_dir = data_dir / split / 'masks'
        
        if not images_dir.exists():
            print(f"错误: 目录不存在 {images_dir}")
            return None, None
        
        # 收集任务
        image_files = list(images_dir.glob('*.jpg'))
        print(f"\n处理 {split} 集: {len(image_files)} 张图片 ({num_workers} 进程)")
        print(f"窗口大小: {self.config.WINDOW_SIZE}, 步长: {self.config.STRIDE}")
        print("-" * 60)
        
        tasks = []
        for img_file in image_files:
            mask_name = img_file.stem + '.png'
            mask_file = masks_dir / mask_name
            if mask_file.exists():
                tasks.append((
                    str(img_file), 
                    str(mask_file),
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
        
        # 进度追踪
        tracker = ProgressTracker(len(tasks))
        
        # 结果收集
        all_features = []
        all_labels = []
        normal_count = 0
        tampered_count = 0
        
        # 多进程处理
        start_time = time.time()
        last_update_time = start_time
        
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
                
                # 更新进度 (每 50 张或最后一张)
                current_time = time.time()
                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    tracker.print_progress(i + 1, len(tasks), current_time - start_time)
        
        print()  # 换行
        
        if not all_features:
            return None, None
        
        # 合并数据
        print("\n合并数据集...")
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        # 统计
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("数据集统计:")
        print(f"{'='*60}")
        print(f"  处理时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        print(f"  处理速度: {len(tasks)/total_time:.2f} 张/秒")
        print(f"  正常图片: {normal_count} 张")
        print(f"  篡改图片: {tampered_count} 张")
        print(f"  总样本数: {len(X):,}")
        print(f"  篡改像素: {np.sum(y==1):,} ({np.mean(y==1)*100:.1f}%)")
        print(f"  正常像素: {np.sum(y==0):,} ({np.mean(y==0)*100:.1f}%)")
        print(f"  特征维度: {X.shape[1]}")
        
        return X, y


# ============== 模型训练器 ==============
class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
        self.best_threshold = 0.5
        self.feature_importance = None
    
    def _create_model(self):
        """创建模型"""
        model_type = self.config.MODEL_TYPE
        
        if model_type == 'lgb' and HAS_LGB:
            print("使用 LightGBM 模型")
            return lgb.LGBMClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                **self.config.LGB_PARAMS
            )
        
        elif model_type == 'xgb' and HAS_XGB:
            print("使用 XGBoost 模型")
            return xgb.XGBClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                **self.config.XGB_PARAMS
            )
        
        elif model_type == 'ensemble' and HAS_LGB and HAS_XGB:
            print("使用集成模型 (LightGBM + XGBoost + RF)")
            lgb_model = lgb.LGBMClassifier(n_estimators=200, **self.config.LGB_PARAMS)
            xgb_model = xgb.XGBClassifier(n_estimators=200, **self.config.XGB_PARAMS)
            rf_model = RandomForestClassifier(
                n_estimators=200, max_depth=20, n_jobs=-1, class_weight='balanced'
            )
            return VotingClassifier(
                estimators=[('lgb', lgb_model), ('xgb', xgb_model), ('rf', rf_model)],
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
        
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_f1 = f1_score(y_train, train_pred)
        val_f1 = f1_score(y_val, val_pred)
        
        print(f"\n训练集 F1: {train_f1:.4f}")
        print(f"验证集 F1: {val_f1:.4f}")
        
        print("\n阈值优化...")
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_val)[:, 1]
            
            best_f1 = 0
            best_thresh = 0.5
            
            for thresh in np.arange(0.3, 0.8, 0.02):
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
                'model_type': self.config.MODEL_TYPE
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
            'lbp_0', 'lbp_1', 'lbp_2', 'lbp_3', 'lbp_4', 'lbp_5', 'lbp_6', 'lbp_7',
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
            f.write("特征重要性报告\n")
            f.write("=" * 50 + "\n\n")
            
            for i, (name, importance) in enumerate(importance_data, 1):
                f.write(f"{i:2d}. {name:25s} {importance*100:.2f}%\n")
        
        print(f"特征重要性报告: {report_path}")


# ============== 主函数 ==============
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图像篡改像素级分割训练 - 极速优化版')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./release/models/pixel_segmentation',
                        help='输出目录')
    parser.add_argument('--model-type', type=str, default='lgb',
                        choices=['rf', 'lgb', 'xgb', 'ensemble'],
                        help='模型类型')
    parser.add_argument('--window-size', type=int, default=32,
                        help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=16,
                        help='滑动步长')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='多进程数量')
    parser.add_argument('--skip-solid', action='store_true', default=True,
                        help='跳过纯色块 (加速)')
    parser.add_argument('--no-skip-solid', action='store_false', dest='skip_solid',
                        help='不跳过纯色块')
    
    args = parser.parse_args()
    
    config = Config()
    config.WINDOW_SIZE = args.window_size
    config.STRIDE = args.stride
    config.MODEL_TYPE = args.model_type
    config.SKIP_SOLID_BLOCKS = args.skip_solid
    
    print("=" * 60)
    print("图像篡改像素级分割训练 - 极速优化版")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型类型: {config.MODEL_TYPE}")
    print(f"窗口大小: {config.WINDOW_SIZE}")
    print(f"滑动步长: {config.STRIDE}")
    print(f"进程数: {args.num_workers}")
    print(f"跳过纯色块: {config.SKIP_SOLID_BLOCKS}")
    print(f"skimage 可用: {HAS_SKIMAGE}")
    
    print("\n[1/2] 构建数据集...")
    builder = FastDatasetBuilder(config)
    X, y = builder.build_dataset(args.data_dir, 'train', args.num_workers)
    
    if X is None:
        print("错误: 无法构建数据集")
        return
    
    print("\n[2/2] 训练模型...")
    trainer = ModelTrainer(config)
    results = trainer.train(X, y)
    
    trainer.save(args.output_dir)
    
    results['timestamp'] = datetime.now().isoformat()
    results['config'] = {
        'window_size': config.WINDOW_SIZE,
        'stride': config.STRIDE,
        'model_type': config.MODEL_TYPE,
        'feature_dim': config.FEATURE_DIM,
        'skip_solid_blocks': config.SKIP_SOLID_BLOCKS
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
    print(f"模型文件: {args.output_dir}/model.pkl")


if __name__ == '__main__':
    main()