#!/usr/bin/env python3
"""
图像篡改二分类训练脚本 - 优化版

核心优化:
1. 预设配置系统 - 支持多种训练场景
2. 多模型支持 - XGBoost (GPU) / LightGBM / GradientBoosting
3. 改进类别权重 - 针对数据不平衡优化
4. 数据集缓存 - 加速迭代
5. 多进程特征提取 - 8进程并行
6. 置信度分析 - 诊断模型问题
7. FPR核心评估 - 降低误报率

使用方法:
    # 默认配置
    python train_gb.py --data_dir /path/to/data --output_dir ./models
    
    # 使用预设配置 (推荐)
    python train_gb.py --data_dir /path/to/data --preset balanced
    
    # 高精确率配置 (降低误报)
    python train_gb.py --data_dir /path/to/data --preset high_precision
    
    # 使用缓存加速
    python train_gb.py --data_dir /path/to/data --cache_dataset ./cache.npz
    python train_gb.py --data_dir /path/to/data --load_cache ./cache.npz
"""

import os
import sys
import argparse
import json
import time
import pickle
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 特征提取
from release.algorithms.features import FEATURE_NAMES, extract_all_features

# 模型
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# 可选模型
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# GPU 支持
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False


# ============== 预设配置 ==============
PRESETS = {
    'default': {
        'description': '默认配置',
        'model_type': 'gb',
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'class_weight_multiplier': 1.0,  # 权重放大系数
        'subsample': 0.8,
    },
    
    'balanced': {
        'description': '平衡配置 - 推荐',
        'model_type': 'xgb',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 8,
        'class_weight_multiplier': 2.0,  # 正常样本权重放大2倍
        'subsample': 0.8,
    },
    
    'high_precision': {
        'description': '高精确率配置 - 降低误报',
        'model_type': 'xgb',
        'n_estimators': 500,
        'learning_rate': 0.03,
        'max_depth': 10,
        'class_weight_multiplier': 3.0,  # 正常样本权重放大3倍
        'subsample': 0.8,
    },
    
    'aggressive': {
        'description': '激进配置 - 极端不平衡数据',
        'model_type': 'xgb',
        'n_estimators': 600,
        'learning_rate': 0.02,
        'max_depth': 12,
        'class_weight_multiplier': 5.0,  # 正常样本权重放大5倍
        'subsample': 0.8,
    },
    
    'fast': {
        'description': '快速配置 - 验证调试用',
        'model_type': 'xgb',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'class_weight_multiplier': 2.0,
        'subsample': 0.8,
    },
}


# ============== 配置类 ==============
class Config:
    """训练配置"""
    
    def __init__(self):
        # 模型参数
        self.model_type = 'xgb'
        self.n_estimators = 300
        self.learning_rate = 0.05
        self.max_depth = 8
        self.subsample = 0.8
        self.random_state = 42
        
        # 类别权重
        self.class_weight_multiplier = 2.0
        
        # 数据划分
        self.test_size = 0.2
        self.cv_folds = 5
        
        # 并行
        self.num_workers = 8
        
    def apply_preset(self, preset_name: str) -> bool:
        """应用预设配置"""
        if preset_name not in PRESETS:
            print(f"警告: 未知预设 '{preset_name}'，使用默认配置")
            return False
        
        preset = PRESETS[preset_name]
        print(f"应用预设: {preset_name} - {preset['description']}")
        
        for key, value in preset.items():
            if key != 'description' and hasattr(self, key):
                setattr(self, key, value)
        
        return True
    
    def get_model_params(self) -> dict:
        """获取模型参数"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'random_state': self.random_state,
        }


# ============== 进度显示 ==============
class ProgressTracker:
    """进度追踪器"""
    
    def __init__(self, total: int, desc: str = "处理"):
        self.total = total
        self.desc = desc
        self.start_time = time.time()
        self.completed = 0
    
    def update(self, n: int = 1):
        self.completed += n
        elapsed = time.time() - self.start_time
        speed = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / speed if speed > 0 else 0
        
        bar_length = 30
        filled = int(bar_length * self.completed / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r  {self.desc}: [{bar}] {self.completed}/{self.total} | "
              f"{speed:.1f}张/s | ETA: {eta/60:.1f}min", end='', flush=True)
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\n  完成: {self.total} 张, 耗时: {elapsed:.1f}s")


# ============== 特征提取 ==============
def extract_features_single(image_path: str) -> Tuple[np.ndarray, bool]:
    """提取单张图片特征"""
    try:
        features = extract_all_features(image_path)
        return features, True
    except Exception as e:
        return None, False


def extract_features_worker(args):
    """多进程工作函数"""
    sample, = args
    features, success = extract_features_single(sample['path'])
    return features, sample['label'], sample['category'], success


def build_feature_matrix_multiprocess(samples: List[Dict], num_workers: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """多进程构建特征矩阵"""
    print(f"\n提取特征 ({num_workers} 进程)...")
    
    tracker = ProgressTracker(len(samples), "特征提取")
    
    X_list = []
    y_list = []
    failed = 0
    category_count = {'easy': 0, 'difficult': 0, 'good': 0}
    
    tasks = [(sample,) for sample in samples]
    
    with mp.Pool(num_workers) as pool:
        for features, label, category, success in pool.imap_unordered(extract_features_worker, tasks):
            tracker.update()
            
            if success and features is not None:
                X_list.append(features)
                y_list.append(label)
                category_count[category] += 1
            else:
                failed += 1
    
    tracker.finish()
    
    if not X_list:
        print("错误: 没有成功提取任何特征!")
        return None, None
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n特征矩阵: {X.shape}")
    print(f"标签分布:")
    print(f"  Easy: {category_count['easy']}")
    print(f"  Difficult: {category_count['difficult']}")
    print(f"  Good: {category_count['good']}")
    print(f"  篡改: {np.sum(y==1)}, 正常: {np.sum(y==0)}")
    print(f"  失败: {failed}")
    
    return X, y


# ============== 数据收集 ==============
def collect_samples(data_dir: str, max_tampered: int = None, max_normal: int = None) -> List[Dict]:
    """
    收集样本
    
    Args:
        data_dir: 数据目录
        max_tampered: 篡改样本最大数量 (None表示不限制)
        max_normal: 正常样本最大数量 (None表示不限制)
    """
    samples = []
    
    # Easy (简单篡改)
    easy_dir = os.path.join(data_dir, 'easy/images')
    if os.path.exists(easy_dir):
        for f in sorted(os.listdir(easy_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(easy_dir, f),
                    'label': 1,
                    'category': 'easy'
                })
    
    # Difficult (复杂篡改)
    diff_dir = os.path.join(data_dir, 'difficult/images')
    if os.path.exists(diff_dir):
        for f in sorted(os.listdir(diff_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(diff_dir, f),
                    'label': 1,
                    'category': 'difficult'
                })
    
    # Good (正常)
    good_dir = os.path.join(data_dir, 'good/images')
    if not os.path.exists(good_dir):
        good_dir = os.path.join(data_dir, 'good')
    if os.path.exists(good_dir):
        for f in sorted(os.listdir(good_dir)):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                samples.append({
                    'path': os.path.join(good_dir, f),
                    'label': 0,
                    'category': 'good'
                })
    
    # 采样控制
    if max_tampered or max_normal:
        tampered = [s for s in samples if s['label'] == 1]
        normal = [s for s in samples if s['label'] == 0]
        
        import random
        random.seed(42)
        
        if max_tampered and len(tampered) > max_tampered:
            tampered = random.sample(tampered, max_tampered)
        if max_normal and len(normal) > max_normal:
            normal = random.sample(normal, max_normal)
        
        samples = tampered + normal
        random.shuffle(samples)
    
    return samples


# ============== 模型训练器 ==============
class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
        self.best_threshold = 0.5
        self.feature_importance = None
    
    def _create_model(self, scale_pos_weight: float = 1.0):
        """创建模型"""
        
        if self.config.model_type == 'xgb' and HAS_XGB:
            print(f"使用 XGBoost 模型 (GPU: {'启用' if HAS_TORCH else '禁用'})")
            
            params = {
                'n_estimators': self.config.n_estimators,
                'learning_rate': self.config.learning_rate,
                'max_depth': self.config.max_depth,
                'subsample': self.config.subsample,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'scale_pos_weight': scale_pos_weight,
                'n_jobs': -1,
                'random_state': self.config.random_state,
            }
            
            if HAS_TORCH:
                params['tree_method'] = 'hist'
                params['device'] = 'cuda:0'
            
            return xgb.XGBClassifier(**params)
        
        elif self.config.model_type == 'lgb' and HAS_LGB:
            print("使用 LightGBM 模型")
            
            return lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                subsample=self.config.subsample,
                class_weight={0: scale_pos_weight, 1: 1},
                n_jobs=-1,
                random_state=self.config.random_state,
                verbose=-1,
            )
        
        else:
            print("使用 GradientBoosting 模型")
            
            # GB 使用 sample_weight 方式
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                subsample=self.config.subsample,
                random_state=self.config.random_state,
            )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """训练模型"""
        print("\n" + "=" * 60)
        print("训练模型")
        print("=" * 60)
        
        # 标准化
        print("\n标准化特征...")
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分数据集
        print("划分数据集...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        n_tampered = np.sum(y_train == 1)
        n_normal = np.sum(y_train == 0)
        
        print(f"  训练集: {len(y_train)} (篡改={n_tampered}, 正常={n_normal})")
        print(f"  测试集: {len(y_test)} (篡改={np.sum(y_test==1)}, 正常={np.sum(y_test==0)})")
        
        # 计算类别权重
        # 关键优化: 使用更激进的权重
        base_weight = n_tampered / n_normal if n_normal > 0 else 1.0
        scale_pos_weight = base_weight * self.config.class_weight_multiplier
        
        print(f"\n类别权重:")
        print(f"  基础比例: {base_weight:.2f}")
        print(f"  放大系数: {self.config.class_weight_multiplier}")
        print(f"  最终权重: {scale_pos_weight:.2f}")
        
        # 创建模型
        self.model = self._create_model(scale_pos_weight)
        
        # 训练
        print("\n训练中...")
        start_time = time.time()
        
        if self.config.model_type == 'gb':
            # GB 需要单独设置 sample_weight
            sample_weight = np.where(y_train == 0, scale_pos_weight, 1.0)
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"训练耗时: {train_time:.1f}s")
        
        # 交叉验证
        print(f"\n{self.config.cv_folds}折交叉验证...")
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='f1')
        print(f"CV F1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 评估
        results = self._evaluate(X_train, y_train, X_test, y_test)
        results['train_time'] = train_time
        results['cv_f1'] = float(cv_scores.mean())
        results['cv_f1_std'] = float(cv_scores.std())
        
        # 置信度分析
        self._analyze_confidence(X_test, y_test)
        
        return results
    
    def _evaluate(self, X_train, y_train, X_test, y_test) -> Dict:
        """评估模型"""
        print("\n" + "=" * 60)
        print("评估模型")
        print("=" * 60)
        
        # 获取概率
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 阈值优化 - 重点优化 FPR
        print("\n阈值优化 (目标: 降低FPR)...")
        best_threshold, best_metrics = self._optimize_threshold(y_test, y_proba)
        self.best_threshold = best_threshold
        
        # 使用最优阈值预测
        y_pred = (y_proba >= best_threshold).astype(int)
        
        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        
        # 训练集指标
        train_pred = self.model.predict(X_train)
        train_f1 = f1_score(y_train, train_pred)
        
        print(f"\n测试集评估:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  FPR (误报率): {fpr:.4f} ⚠️" if fpr > 0.1 else f"  FPR (误报率): {fpr:.4f}")
        print(f"  FNR (漏报率): {fnr:.4f}")
        print(f"  最优阈值: {best_threshold:.2f}")
        
        # 混淆矩阵
        print(f"\n混淆矩阵:")
        print(f"  TP={cm[1,1]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TN={cm[0,0]}")
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
            self._print_feature_importance()
        
        return {
            'train_f1': float(train_f1),
            'test_f1': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'accuracy': float(acc),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'best_threshold': float(best_threshold),
        }
    
    def _optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict]:
        """
        阈值优化 - 重点优化 FPR
        
        目标函数: score = F1 - FPR * penalty
        """
        best_threshold = 0.5
        best_score = 0
        best_metrics = None
        
        print(f"\n{'阈值':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'FPR':>8} {'Score':>8}")
        print("-" * 60)
        
        # FPR 惩罚系数 - FPR 越高惩罚越大
        fpr_penalty = 1.0
        
        for threshold in np.arange(0.3, 0.95, 0.05):
            y_pred = (y_proba >= threshold).astype(int)
            
            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            
            cm = confusion_matrix(y_true, y_pred)
            fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            
            # 综合得分: F1 - FPR惩罚
            score = f1 - fpr * fpr_penalty
            
            print(f"{threshold:>8.2f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {fpr:>8.4f} {score:>8.4f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {'f1': f1, 'precision': prec, 'recall': rec, 'fpr': fpr}
        
        print(f"\n最优阈值: {best_threshold:.2f} (Score: {best_score:.4f})")
        
        return best_threshold, best_metrics
    
    def _analyze_confidence(self, X: np.ndarray, y: np.ndarray):
        """置信度分析 - 诊断模型问题"""
        print("\n" + "=" * 60)
        print("置信度分析")
        print("=" * 60)
        
        y_proba = self.model.predict_proba(X)[:, 1]
        
        tampered_proba = y_proba[y == 1]
        normal_proba = y_proba[y == 0]
        
        print(f"\n篡改样本置信度:")
        print(f"  均值: {tampered_proba.mean():.4f}")
        print(f"  标准差: {tampered_proba.std():.4f}")
        print(f"  范围: [{tampered_proba.min():.4f}, {tampered_proba.max():.4f}]")
        
        print(f"\n正常样本置信度:")
        print(f"  均值: {normal_proba.mean():.4f}")
        print(f"  标准差: {normal_proba.std():.4f}")
        print(f"  范围: [{normal_proba.min():.4f}, {normal_proba.max():.4f}]")
        
        # 诊断
        diff = abs(tampered_proba.mean() - normal_proba.mean())
        if diff < 0.1:
            print(f"\n⚠️ 警告: 置信度差异过小 ({diff:.4f})，模型未能有效区分篡改/正常样本!")
            print("建议: 增大 class_weight_multiplier 或使用更平衡的数据集")
        elif diff < 0.3:
            print(f"\n⚠️ 注意: 置信度差异较小 ({diff:.4f})，模型区分能力有限")
        else:
            print(f"\n✓ 置信度差异: {diff:.4f}，模型具有一定区分能力")
        
        # 计算分离度
        if tampered_proba.std() > 0 and normal_proba.std() > 0:
            separation = diff / np.sqrt(tampered_proba.std()**2 + normal_proba.std()**2)
            print(f"  分离度: {separation:.2f} (>1.0 表示较好)")
    
    def _print_feature_importance(self):
        """打印特征重要性"""
        if self.feature_importance is None:
            return
        
        print("\n特征重要性 Top 10:")
        indices = np.argsort(self.feature_importance)[::-1]
        for i, idx in enumerate(indices[:10]):
            if idx < len(FEATURE_NAMES):
                print(f"  {i+1}. {FEATURE_NAMES[idx]}: {self.feature_importance[idx]:.4f}")
    
    def save(self, output_dir: str):
        """保存模型"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = output_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # 保存标准化器
        scaler_path = output_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存配置
        config = {
            'feature_names': FEATURE_NAMES,
            'optimal_threshold': float(self.best_threshold),
            'model_type': self.config.model_type,
            'n_estimators': self.config.n_estimators,
            'learning_rate': self.config.learning_rate,
            'max_depth': self.config.max_depth,
            'class_weight_multiplier': self.config.class_weight_multiplier,
        }
        config_path = output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n模型已保存:")
        print(f"  {model_path}")
        print(f"  {scaler_path}")
        print(f"  {config_path}")


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(
        description='图像篡改二分类训练 - 优化版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预设配置:
  default        默认配置
  balanced       平衡配置 - 推荐
  high_precision 高精确率配置 - 降低误报
  aggressive     激进配置 - 极端不平衡数据
  fast           快速配置 - 验证调试用

示例:
  # 使用预设配置
  python train_gb.py --data_dir /path/to/data --preset balanced
  
  # 使用缓存加速
  python train_gb.py --data_dir /path/to/data --cache_dataset ./cache.npz
  python train_gb.py --data_dir /path/to/data --load_cache ./cache.npz
  
  # 采样控制
  python train_gb.py --data_dir /path/to/data --max_tampered 6000 --max_normal 3000
        """
    )
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./release/models/gb_classifier', help='输出目录')
    
    # 预设配置
    parser.add_argument('--preset', type=str, default='balanced',
                        choices=list(PRESETS.keys()), help='预设配置')
    
    # 模型参数 (覆盖预设)
    parser.add_argument('--model_type', type=str, choices=['xgb', 'lgb', 'gb'], help='模型类型')
    parser.add_argument('--n_estimators', type=int, help='树数量')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--max_depth', type=int, help='最大深度')
    parser.add_argument('--class_weight_multiplier', type=float, help='类别权重放大系数')
    
    # 采样参数
    parser.add_argument('--max_tampered', type=int, help='篡改样本最大数量')
    parser.add_argument('--max_normal', type=int, help='正常样本最大数量')
    
    # 缓存参数
    parser.add_argument('--cache_dataset', type=str, help='保存数据集缓存路径')
    parser.add_argument('--load_cache', type=str, help='加载数据集缓存路径')
    
    # 并行参数
    parser.add_argument('--num_workers', type=int, default=8, help='并行进程数')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    config.apply_preset(args.preset)
    config.num_workers = args.num_workers
    
    # 覆盖预设参数
    if args.model_type:
        config.model_type = args.model_type
    if args.n_estimators:
        config.n_estimators = args.n_estimators
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_depth:
        config.max_depth = args.max_depth
    if args.class_weight_multiplier:
        config.class_weight_multiplier = args.class_weight_multiplier
    
    # 打印配置
    print("=" * 60)
    print("图像篡改二分类训练 - 优化版")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"预设: {args.preset}")
    print(f"模型: {config.model_type}")
    print(f"树数量: {config.n_estimators}")
    print(f"学习率: {config.learning_rate}")
    print(f"最大深度: {config.max_depth}")
    print(f"权重放大: {config.class_weight_multiplier}")
    print(f"进程数: {config.num_workers}")
    print("=" * 60)
    
    # 加载或构建数据集
    if args.load_cache and os.path.exists(args.load_cache):
        print(f"\n加载缓存: {args.load_cache}")
        data = np.load(args.load_cache)
        X, y = data['X'], data['y']
        print(f"特征矩阵: {X.shape}")
        print(f"标签分布: 篡改={np.sum(y==1)}, 正常={np.sum(y==0)}")
    else:
        # 收集样本
        print("\n收集样本...")
        samples = collect_samples(args.data_dir, args.max_tampered, args.max_normal)
        
        if not samples:
            print("错误: 未找到任何样本!")
            return
        
        tampered = sum(1 for s in samples if s['label'] == 1)
        normal = sum(1 for s in samples if s['label'] == 0)
        print(f"总样本: {len(samples)} (篡改={tampered}, 正常={normal})")
        
        # 构建特征矩阵
        X, y = build_feature_matrix_multiprocess(samples, config.num_workers)
        
        if X is None:
            return
        
        # 保存缓存
        if args.cache_dataset:
            print(f"\n保存缓存: {args.cache_dataset}")
            os.makedirs(os.path.dirname(args.cache_dataset) or '.', exist_ok=True)
            np.savez(args.cache_dataset, X=X, y=y, feature_names=FEATURE_NAMES)
    
    # 训练模型
    trainer = ModelTrainer(config)
    results = trainer.train(X, y)
    
    # 保存模型
    trainer.save(args.output_dir)
    
    # 保存结果
    results['timestamp'] = datetime.now().isoformat()
    results['config'] = {
        'preset': args.preset,
        'model_type': config.model_type,
        'n_estimators': config.n_estimators,
        'learning_rate': config.learning_rate,
        'max_depth': config.max_depth,
        'class_weight_multiplier': config.class_weight_multiplier,
    }
    
    results_path = Path(args.output_dir) / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 总结
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"F1: {results['test_f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"FPR (误报率): {results['fpr']:.4f}")
    print(f"最优阈值: {results['best_threshold']:.2f}")
    print(f"模型文件: {args.output_dir}/model.pkl")
    print(f"结果文件: {args.output_dir}/results.json")


if __name__ == '__main__':
    main()