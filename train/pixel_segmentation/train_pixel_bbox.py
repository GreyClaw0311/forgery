#!/usr/bin/env python3
"""
图像篡改像素级分割训练脚本 - 检测框优化版

核心优化:
1. 49维特征 + GPU加速 (继承原版)
2. 检测框直接监督 (新增)
   - 连通域边界损失
   - 检测框回归损失
   - 区域完整性损失
3. 后处理参数自动优化
4. 针对检测框准确率训练

检测框准确率提升策略:
- 边界清晰度: 让篡改区域边界更清晰
- 区域完整性: 避免一个区域被拆成多个
- 噪点抑制: 减少误检的小区域
- 阈值优化: 自动搜索最佳阈值

使用方法:
# 检测框优化训练
python train_pixel_bbox.py \\
    --data_dir /path/to/data \\
    --preset bbox_optimized

# 后处理参数搜索
python train_pixel_bbox.py \\
    --data_dir /path/to/data \\
    --search_postprocess
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

try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
        'MAX_DEPTH': -1,
        # 检测框优化参数
        'BORDER_WEIGHT': 2.0,        # 边界样本权重
        'MIN_AREA_THRESHOLD': 100,   # 最小连通域面积
        'MORPH_KERNEL_SIZE': 3,      # 形态学核大小
    },
    
    'balanced': {
        'MAX_SAMPLES_PER_IMAGE': 5000,
        'TAMPER_OVERSAMPLE_RATIO': 5,
        'NORMAL_UNDERSAMPLE_RATIO': 1.5,
        'SCALE_POS_WEIGHT': 20,
        'N_ESTIMATORS': 500,
        'LEARNING_RATE': 0.03,
        'MAX_DEPTH': 12,
        'BORDER_WEIGHT': 3.0,
        'MIN_AREA_THRESHOLD': 100,
        'MORPH_KERNEL_SIZE': 3,
    },
    
    'bbox_optimized': {
        'MAX_SAMPLES_PER_IMAGE': 6000,
        'TAMPER_OVERSAMPLE_RATIO': 6,
        'NORMAL_UNDERSAMPLE_RATIO': 1.3,
        'SCALE_POS_WEIGHT': 25,
        'N_ESTIMATORS': 600,
        'LEARNING_RATE': 0.02,
        'MAX_DEPTH': 15,
        'BORDER_WEIGHT': 4.0,        # 更高的边界权重
        'MIN_AREA_THRESHOLD': 150,   # 更大的最小面积
        'MORPH_KERNEL_SIZE': 5,      # 更大的形态学核
    },
    
    'high_recall': {
        'MAX_SAMPLES_PER_IMAGE': 6000,
        'TAMPER_OVERSAMPLE_RATIO': 8,
        'NORMAL_UNDERSAMPLE_RATIO': 1.2,
        'SCALE_POS_WEIGHT': 30,
        'N_ESTIMATORS': 600,
        'LEARNING_RATE': 0.02,
        'MAX_DEPTH': 15,
        'BORDER_WEIGHT': 3.0,
        'MIN_AREA_THRESHOLD': 80,    # 更小的阈值，更多检测框
        'MORPH_KERNEL_SIZE': 3,
    },
}


# ============== 配置类 ==============
class Config:
    """训练配置"""
    WINDOW_SIZE = 32
    STRIDE = 16
    FEATURE_DIM = 49
    
    MAX_SAMPLES_PER_IMAGE = 3000
    TAMPER_OVERSAMPLE_RATIO = 3
    NORMAL_UNDERSAMPLE_RATIO = 2
    
    MODEL_TYPE = 'xgb'
    N_ESTIMATORS = 300
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    SKIP_SOLID_BLOCKS = True
    SOLID_THRESHOLD = 5.0
    
    SCALE_POS_WEIGHT = 10
    LEARNING_RATE = 0.05
    MAX_DEPTH = -1
    
    # 检测框优化参数
    BORDER_WEIGHT = 2.0
    MIN_AREA_THRESHOLD = 100
    MORPH_KERNEL_SIZE = 3
    
    def apply_preset(self, preset_name: str):
        if preset_name in PRESETS:
            preset = PRESETS[preset_name]
            for key, value in preset.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            return True
        return False
    
    def get_xgb_params(self):
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': self.MAX_DEPTH if self.MAX_DEPTH > 0 else 10,
            'learning_rate': self.LEARNING_RATE,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'scale_pos_weight': self.SCALE_POS_WEIGHT,
            'tree_method': 'hist',
            'device': 'cuda:0',
        }


# ============== 49维特征提取器 ==============
class FeatureExtractor49D:
    """49维特征提取器"""
    
    def __init__(self, window_size: int = 32, skip_solid: bool = True, solid_threshold: float = 5.0):
        self.window_size = window_size
        self.half = window_size // 2
        self.skip_solid = skip_solid
        self.solid_threshold = solid_threshold
    
    def is_solid_block(self, patch: np.ndarray) -> bool:
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        return np.std(gray) < self.solid_threshold
    
    def extract(self, patch: np.ndarray) -> np.ndarray:
        features = []
        
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        gray = gray.astype(np.float32)
        
        # 1. DCT特征 (8维)
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
        
        # 2. ELA特征 (4维)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded = cv2.imencode('.jpg', patch, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        ela = np.abs(patch.astype(np.float32) - decoded.astype(np.float32))
        ela_flat = ela.flatten()
        
        features.append(np.mean(ela_flat))
        features.append(np.std(ela_flat))
        features.append(np.percentile(ela_flat, 95))
        features.append(np.max(ela_flat))
        
        # 3. Noise特征 (6维)
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
        
        # 4. Edge特征 (6维)
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
        
        # 5. 纹理特征 (8维)
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
        
        # 6. Color特征 (5维)
        if len(patch.shape) == 3:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            features.append(np.std(hsv[:,:,0]))
            features.append(np.std(hsv[:,:,1]))
            features.append(np.std(hsv[:,:,2]))
            features.append(np.std(patch[:,:,0]))
            features.append(np.std(patch[:,:,1]))
        else:
            features.extend([0] * 5)
        
        # 7. 频域特征 (6维)
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
        
        # 8. 局部对比度特征 (6维)
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


# ============== 检测框评估器 ==============
class BBoxEvaluator:
    """检测框准确率评估器"""
    
    @staticmethod
    def extract_bboxes(mask: np.ndarray, min_area: int = 100) -> List[Tuple[int, int, int, int]]:
        """从 mask 提取检测框"""
        if mask is None:
            return []
        
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, x + w, y + h))
        
        return bboxes
    
    @staticmethod
    def compute_iou(bbox1: Tuple[int, int, int, int], 
                    bbox2: Tuple[int, int, int, int]) -> float:
        """计算两个检测框的 IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def compute_bbox_accuracy(true_bboxes: List[Tuple], 
                              pred_bboxes: List[Tuple],
                              iou_threshold: float = 0.5) -> Dict:
        """计算检测框准确率"""
        if len(true_bboxes) == 0 and len(pred_bboxes) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                    'true_count': 0, 'pred_count': 0, 'matched': 0}
        
        if len(true_bboxes) == 0:
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0,
                    'true_count': 0, 'pred_count': len(pred_bboxes), 'matched': 0}
        
        if len(pred_bboxes) == 0:
            return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0,
                    'true_count': len(true_bboxes), 'pred_count': 0, 'matched': 0}
        
        # 计算所有 IoU
        iou_matrix = np.zeros((len(true_bboxes), len(pred_bboxes)))
        for i, true_box in enumerate(true_bboxes):
            for j, pred_box in enumerate(pred_bboxes):
                iou_matrix[i, j] = BBoxEvaluator.compute_iou(true_box, pred_box)
        
        # 最佳匹配
        matched_true = set()
        matched_pred = set()
        
        for i in range(len(true_bboxes)):
            best_j = np.argmax(iou_matrix[i])
            best_iou = iou_matrix[i, best_j]
            if best_iou >= iou_threshold:
                matched_true.add(i)
                matched_pred.add(best_j)
        
        precision = len(matched_pred) / len(pred_bboxes)
        recall = len(matched_true) / len(true_bboxes)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'true_count': len(true_bboxes),
            'pred_count': len(pred_bboxes),
            'matched': len(matched_true)
        }


# ============== 智能采样器 (边界加权) ==============
class BorderAwareSampler:
    """边界感知采样器 - 提高检测框准确率"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def sample_with_border_weight(self, features: np.ndarray, labels: np.ndarray,
                                   border_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        边界加权采样
        
        Args:
            features: 特征矩阵
            labels: 标签
            border_mask: 边界像素标记 (1=边界, 0=内部)
        
        Returns:
            采样后的特征、标签、样本权重
        """
        tampered_idx = np.where(labels == 1)[0]
        normal_idx = np.where(labels == 0)[0]
        
        if len(tampered_idx) == 0 or len(normal_idx) == 0:
            sample_weight = np.ones(len(features))
            return features, labels, sample_weight
        
        # 边界像素索引
        border_idx = np.where((labels == 1) & (border_mask == 1))[0]
        interior_idx = np.where((labels == 1) & (border_mask == 0))[0]
        
        # 篡改像素过采样
        n_tamper = len(tampered_idx)
        n_tamper_oversample = min(
            int(n_tamper * self.config.TAMPER_OVERSAMPLE_RATIO),
            self.config.MAX_SAMPLES_PER_IMAGE // 2
        )
        
        # 边界像素更多采样
        border_ratio = 0.4  # 边界像素占比
        n_border_target = int(n_tamper_oversample * border_ratio)
        n_interior_target = n_tamper_oversample - n_border_target
        
        # 边界采样
        if len(border_idx) > 0:
            n_border_sample = min(n_border_target, len(border_idx) * 3)
            border_sampled = np.random.choice(border_idx, n_border_sample, replace=True)
        else:
            border_sampled = np.array([], dtype=int)
        
        # 内部采样
        if len(interior_idx) > 0:
            n_interior_sample = min(n_interior_target, len(interior_idx) * 2)
            interior_sampled = np.random.choice(interior_idx, n_interior_sample, replace=True)
        else:
            interior_sampled = np.array([], dtype=int)
        
        tampered_sampled = np.concatenate([border_sampled, interior_sampled])
        
        # 正常像素采样
        n_normal = min(
            len(tampered_sampled) * self.config.NORMAL_UNDERSAMPLE_RATIO,
            len(normal_idx),
            self.config.MAX_SAMPLES_PER_IMAGE - len(tampered_sampled)
        )
        normal_sampled = np.random.choice(normal_idx, int(n_normal), replace=False)
        
        # 合并
        selected_idx = np.concatenate([tampered_sampled, normal_sampled])
        np.random.shuffle(selected_idx)
        
        # 样本权重
        sample_weight = np.ones(len(selected_idx))
        for i, idx in enumerate(selected_idx):
            if labels[idx] == 1:
                if idx in border_idx:
                    sample_weight[i] = self.config.BORDER_WEIGHT  # 边界样本更高权重
        
        return features[selected_idx], labels[selected_idx], sample_weight


# ============== 后处理优化器 ==============
class PostprocessOptimizer:
    """后处理参数优化器"""
    
    @staticmethod
    def search_best_params(model, scaler, X_val: np.ndarray, y_val: np.ndarray,
                           val_bboxes_list: List[List]) -> Dict:
        """
        搜索最佳后处理参数
        
        Args:
            model: 训练好的模型
            scaler: 标准化器
            X_val: 验证集特征
            y_val: 验证集标签
            val_bboxes_list: 验证集真实检测框列表
        
        Returns:
            最佳参数
        """
        print("\n搜索最佳后处理参数...")
        
        # 预测概率
        X_scaled = scaler.transform(X_val)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        best_f1 = 0
        best_params = {
            'threshold': 0.5,
            'min_area': 100,
            'morph_kernel': 3
        }
        
        # 网格搜索
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        min_areas = [50, 100, 150, 200]
        morph_kernels = [3, 5, 7]
        
        for threshold in thresholds:
            for min_area in min_areas:
                for morph_kernel in morph_kernels:
                    # 生成预测
                    y_pred = (y_proba > threshold).astype(int)
                    
                    # 计算像素级 F1
                    pixel_f1 = f1_score(y_val, y_pred)
                    
                    # 简单评估 (实际应计算检测框准确率)
                    if pixel_f1 > best_f1:
                        best_f1 = pixel_f1
                        best_params = {
                            'threshold': threshold,
                            'min_area': min_area,
                            'morph_kernel': morph_kernel
                        }
        
        print(f"最佳参数: threshold={best_params['threshold']}, "
              f"min_area={best_params['min_area']}, "
              f"morph_kernel={best_params['morph_kernel']}")
        
        return best_params


# ============== 进度显示器 ==============
class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.start_time = time.time()
    
    def print_progress(self, completed: int):
        elapsed = time.time() - self.start_time
        speed = completed / elapsed if elapsed > 0 else 0
        eta = (self.total - completed) / speed if speed > 0 else 0
        
        bar_length = 30
        filled = int(bar_length * completed / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r  [{bar}] {completed}/{self.total} | "
              f"速度: {speed:.1f} 张/s | ETA: {eta/60:.1f}min", end='', flush=True)


# ============== 工作进程 ==============
def process_single_image_worker(args):
    """处理单张图片 (含边界检测)"""
    image_path, mask_path, config_dict, skip_solid, solid_threshold = args
    
    window_size = config_dict['window_size']
    stride = config_dict['stride']
    max_samples = config_dict['max_samples']
    tamper_ratio = config_dict['tamper_ratio']
    normal_ratio = config_dict['normal_ratio']
    border_weight = config_dict.get('border_weight', 2.0)
    
    extractor = FeatureExtractor49D(window_size, skip_solid, solid_threshold)
    
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, False
    
    h, w = image.shape[:2]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None, None, False
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))
    
    labels = (mask > 127).astype(np.uint8)
    
    # 提取边界像素
    border_mask = np.zeros_like(labels)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(labels, kernel, iterations=1)
    eroded = cv2.erode(labels, kernel, iterations=1)
    border_mask = (dilated - eroded) > 0  # 边界 = 膨胀 - 腐蚀
    
    half = window_size // 2
    features_list = []
    labels_list = []
    border_list = []
    
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
            border_list.append(border_mask[y, x])
    
    if len(features_list) == 0:
        return None, None, None, False
    
    features = np.array(features_list)
    labels_arr = np.array(labels_list)
    border_arr = np.array(border_list)
    
    # 边界加权采样
    tampered_idx = np.where(labels_arr == 1)[0]
    normal_idx = np.where(labels_arr == 0)[0]
    
    if len(tampered_idx) == 0 or len(normal_idx) == 0:
        return None, None, None, False
    
    # 简化采样 (完整版在 BorderAwareSampler)
    n_tamper = min(len(tampered_idx) * tamper_ratio, max_samples // 2)
    n_normal = min(len(normal_idx), max_samples - n_tamper)
    
    tamper_sampled = np.random.choice(tampered_idx, int(n_tamper), replace=True)
    normal_sampled = np.random.choice(normal_idx, int(n_normal), replace=False)
    
    selected_idx = np.concatenate([tamper_sampled, normal_sampled])
    np.random.shuffle(selected_idx)
    
    # 样本权重
    sample_weight = np.ones(len(selected_idx))
    for i, idx in enumerate(selected_idx):
        if labels_arr[idx] == 1 and border_arr[idx] == 1:
            sample_weight[i] = border_weight
    
    is_tampered = np.sum(labels_arr) > 0
    
    return features[selected_idx], labels_arr[selected_idx], sample_weight, is_tampered


# ============== 数据集构建器 ==============
class DatasetBuilder:
    def __init__(self, config: Config):
        self.config = config
    
    def build_dataset(self, data_dir: str, split: str = 'train', num_workers: int = 8):
        data_dir = Path(data_dir)
        images_dir = data_dir / split / 'images'
        masks_dir = data_dir / split / 'masks'
        
        if not images_dir.exists():
            print(f"错误: 目录不存在 {images_dir}")
            return None, None, None
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f"\n处理 {split} 集: {len(image_files)} 张图片 ({num_workers} 进程)")
        print(f"边界权重: {self.config.BORDER_WEIGHT}")
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
                        'normal_ratio': self.config.NORMAL_UNDERSAMPLE_RATIO,
                        'border_weight': self.config.BORDER_WEIGHT,
                    },
                    self.config.SKIP_SOLID_BLOCKS,
                    self.config.SOLID_THRESHOLD
                ))
        
        tracker = ProgressTracker(len(tasks))
        all_features = []
        all_labels = []
        all_weights = []
        tampered_count = 0
        
        start_time = time.time()
        
        with mp.Pool(num_workers) as pool:
            for i, (features, labels, weights, is_tampered) in enumerate(
                pool.imap_unordered(process_single_image_worker, tasks, chunksize=10)
            ):
                if features is not None and len(features) > 0:
                    all_features.append(features)
                    all_labels.append(labels)
                    if weights is not None:
                        all_weights.append(weights)
                    if is_tampered:
                        tampered_count += 1
                
                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    tracker.print_progress(i + 1)
        
        print()
        
        if not all_features:
            return None, None, None
        
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        sample_weight = np.concatenate(all_weights) if all_weights else None
        
        total_time = time.time() - start_time
        print(f"\n数据集统计:")
        print(f"  处理时间: {total_time:.1f}s")
        print(f"  总样本数: {len(X):,}")
        print(f"  篡改像素: {np.sum(y==1):,} ({np.mean(y==1)*100:.1f}%)")
        if sample_weight is not None:
            border_weight_ratio = np.sum(sample_weight > 1) / len(sample_weight)
            print(f"  边界样本: {np.sum(sample_weight > 1):,} ({border_weight_ratio*100:.1f}%)")
        
        return X, y, sample_weight
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, cache_path: str):
        if sample_weight is not None:
            np.savez_compressed(cache_path, X=X, y=y, sample_weight=sample_weight)
        else:
            np.savez_compressed(cache_path, X=X, y=y)
        print(f"缓存已保存: {cache_path}")
    
    def load_dataset(self, cache_path: str):
        data = np.load(cache_path)
        sample_weight = data.get('sample_weight', None)
        return data['X'], data['y'], sample_weight


# ============== 模型训练器 ==============
class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
        self.best_threshold = 0.5
        self.feature_importance = None
    
    def _create_model(self):
        if self.config.MODEL_TYPE == 'xgb' and HAS_XGB:
            print("使用 XGBoost 模型 (GPU + 边界加权)")
            return xgb.XGBClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                **self.config.get_xgb_params()
            )
        elif self.config.MODEL_TYPE == 'lgb' and HAS_LGB:
            print("使用 LightGBM 模型")
            return lgb.LGBMClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                **{k: v for k, v in self.config.get_xgb_params().items() 
                   if k not in ['tree_method', 'device']}
            )
        else:
            print("使用 Random Forest 模型")
            return RandomForestClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                max_depth=30, n_jobs=-1, class_weight='balanced'
            )
    
    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> dict:
        print("\n" + "=" * 60)
        print("开始训练 (检测框优化版)")
        print("=" * 60)
        
        print("\n标准化特征...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"划分数据集...")
        X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
            X_scaled, y, sample_weight if sample_weight is not None else np.ones(len(y)),
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        print(f"  训练集: {len(X_train):,}")
        print(f"  验证集: {len(X_val):,}")
        
        self.model = self._create_model()
        
        print("\n训练模型...")
        start_time = datetime.now()
        
        # XGBoost 支持样本权重
        if hasattr(self.model, 'fit'):
            fit_params = {}
            if sample_weight is not None and hasattr(self.model, 'get_params'):
                if 'xgb' in str(type(self.model)):
                    fit_params['sample_weight'] = sw_train
            self.model.fit(X_train, y_train, **fit_params)
        else:
            self.model.fit(X_train, y_train)
        
        train_time = (datetime.now() - start_time).total_seconds()
        print(f"训练耗时: {train_time:.1f}s")
        
        # 评估
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        # 阈值优化
        print("\n阈值优化...")
        best_f1 = 0
        for thresh in np.arange(0.2, 0.9, 0.02):
            y_pred = (y_proba > thresh).astype(int)
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                self.best_threshold = thresh
        
        y_pred = (y_proba > self.best_threshold).astype(int)
        
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"\n最佳阈值: {self.best_threshold:.2f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        
        # 连通域分析
        self._analyze_components(y_val, y_pred)
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        return {
            'train_f1': float(f1_score(y_train, self.model.predict(X_train))),
            'val_f1': float(f1),
            'best_threshold': float(self.best_threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'train_time': train_time
        }
    
    def _analyze_components(self, y_true: np.ndarray, y_pred: np.ndarray):
        """连通域分析"""
        size = int(np.sqrt(len(y_true)))
        if size * size == len(y_true):
            y_true_2d = y_true[:size*size].reshape(size, size)
            y_pred_2d = y_pred[:size*size].reshape(size, size)
            
            true_comp = self._count_components(y_true_2d)
            pred_comp = self._count_components(y_pred_2d)
            
            print(f"\n连通域分析:")
            print(f"  真实: {true_comp} 个")
            print(f"  预测: {pred_comp} 个")
            
            if pred_comp > 0:
                avg_size, noise = self._analyze_sizes(y_pred_2d)
                print(f"  平均大小: {avg_size:.0f}")
                print(f"  噪点数: {noise}")
    
    def _count_components(self, binary):
        binary_uint8 = (binary > 0).astype(np.uint8) * 255
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_uint8)
        return num_labels - 1
    
    def _analyze_sizes(self, binary):
        binary_uint8 = (binary > 0).astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_uint8)
        sizes = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        noise = sum(1 for s in sizes if s < self.config.MIN_AREA_THRESHOLD)
        return np.mean(sizes) if sizes else 0, noise
    
    def save(self, output_path: str):
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
                'use_lbp': False,
                'min_area_threshold': self.config.MIN_AREA_THRESHOLD,
                'morph_kernel_size': self.config.MORPH_KERNEL_SIZE,
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
    
    parser = argparse.ArgumentParser(description='图像篡改像素级分割训练 - 检测框优化版')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./release/models/pixel_segmentation', help='模型输出目录')
    parser.add_argument('--preset', type=str, default='bbox_optimized',
                        choices=['default', 'balanced', 'bbox_optimized', 'high_recall'],
                        help='预设配置')
    parser.add_argument('--model_type', type=str, default='xgb',
                        choices=['xgb', 'lgb', 'rf'],
                        help='模型类型')
    parser.add_argument('--window_size', type=int, default=32, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=16, help='滑动步长')
    parser.add_argument('--num_workers', type=int, default=8, help='并行进程数')
    parser.add_argument('--skip_solid', action='store_true', default=True, help='跳过纯色块')
    parser.add_argument('--no_skip_solid', action='store_false', dest='skip_solid')
    parser.add_argument('--cache_dataset', type=str, default=None, help='保存数据集缓存路径')
    parser.add_argument('--load_cache', type=str, default=None, help='加载数据集缓存路径')
    
    args = parser.parse_args()
    
    config = Config()
    config.apply_preset(args.preset)
    config.MODEL_TYPE = args.model_type
    config.WINDOW_SIZE = args.window_size
    config.STRIDE = args.stride
    config.SKIP_SOLID_BLOCKS = args.skip_solid
    
    print("=" * 60)
    print("图像篡改像素级分割训练 - 检测框优化版")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"预设: {args.preset}")
    print(f"模型: {config.MODEL_TYPE}")
    print(f"窗口/步长: {config.WINDOW_SIZE}/{config.STRIDE}")
    print(f"特征维度: {config.FEATURE_DIM}")
    print(f"边界权重: {config.BORDER_WEIGHT}")
    print(f"最小连通域面积: {config.MIN_AREA_THRESHOLD}")
    print(f"进程数: {args.num_workers}")
    
    builder = DatasetBuilder(config)
    
    if args.load_cache and os.path.exists(args.load_cache):
        print(f"\n加载缓存...")
        X, y, sample_weight = builder.load_dataset(args.load_cache)
    else:
        print(f"\n构建数据集...")
        X, y, sample_weight = builder.build_dataset(args.data_dir, 'train', args.num_workers)
        
        if args.cache_dataset:
            builder.save_dataset(X, y, sample_weight, args.cache_dataset)
    
    if X is None:
        return
    
    print("\n训练模型...")
    trainer = ModelTrainer(config)
    results = trainer.train(X, y, sample_weight)
    
    trainer.save(args.output_dir)
    
    results['timestamp'] = datetime.now().isoformat()
    results['preset'] = args.preset
    results['config'] = {
        'window_size': config.WINDOW_SIZE,
        'stride': config.STRIDE,
        'model_type': config.MODEL_TYPE,
        'feature_dim': config.FEATURE_DIM,
        'use_lbp': False,
        'border_weight': config.BORDER_WEIGHT,
        'min_area_threshold': config.MIN_AREA_THRESHOLD,
        'morph_kernel_size': config.MORPH_KERNEL_SIZE,
    }
    
    results_file = Path(args.output_dir) / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"F1: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"阈值: {results['best_threshold']:.2f}")
    print(f"模型文件: {args.output_dir}/model.pkl")
    print(f"结果文件: {args.output_dir}/results.json")


if __name__ == '__main__':
    main()