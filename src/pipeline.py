"""
图像篡改检测Pipeline
融合多个特征进行综合判断
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FEATURES, DEFAULT_THRESHOLDS
from src.features import (
    feature_ela, feature_dct, feature_cfa, feature_noise,
    feature_edge, feature_lbp, feature_histogram, feature_sift,
    feature_fft, feature_metadata
)

class TamperDetectionPipeline:
    """图像篡改检测Pipeline"""
    
    def __init__(self, features=None, thresholds=None, weights=None):
        """
        初始化Pipeline
        
        Args:
            features: 使用的特征列表，默认使用推荐特征
            thresholds: 各特征的阈值
            weights: 各特征的权重
        """
        # 推荐特征组合（基于实验结果）
        self.recommended_features = ['dct', 'fft', 'noise', 'ela']
        
        self.features = features or self.recommended_features
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.weights = weights or {f: 1.0 for f in self.features}
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # 特征模块映射
        self.feature_modules = {
            'ela': feature_ela,
            'dct': feature_dct,
            'cfa': feature_cfa,
            'noise': feature_noise,
            'edge': feature_edge,
            'lbp': feature_lbp,
            'histogram': feature_histogram,
            'sift': feature_sift,
            'fft': feature_fft,
            'metadata': feature_metadata
        }
    
    def extract_features(self, image_path):
        """
        提取所有特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            features: 特征字典 {特征名: (is_tampered, score)}
        """
        features = {}
        
        for feat_name in self.features:
            module = self.feature_modules.get(feat_name)
            if module:
                detect_func = getattr(module, f'detect_tampering_{feat_name}', None)
                if detect_func:
                    try:
                        is_tampered, score = detect_func(image_path)
                        features[feat_name] = (is_tampered, score)
                    except Exception as e:
                        features[feat_name] = (False, 0)
        
        return features
    
    def predict_voting(self, image_path):
        """
        使用投票法预测
        
        Args:
            image_path: 图像路径
            
        Returns:
            is_tampered: 是否篡改
            confidence: 置信度
        """
        features = self.extract_features(image_path)
        
        if not features:
            return False, 0
        
        # 投票
        votes = [1 if is_tampered else 0 for is_tampered, _ in features.values()]
        total = len(votes)
        positive = sum(votes)
        
        is_tampered = positive > total / 2
        confidence = positive / total
        
        return is_tampered, confidence
    
    def predict_weighted(self, image_path):
        """
        使用加权法预测
        
        Args:
            image_path: 图像路径
            
        Returns:
            is_tampered: 是否篡改
            confidence: 置信度
        """
        features = self.extract_features(image_path)
        
        if not features:
            return False, 0
        
        # 加权求和
        weighted_score = 0
        total_weight = 0
        
        for feat_name, (is_tampered, score) in features.items():
            weight = self.weights.get(feat_name, 1.0)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        # 判断阈值
        is_tampered = weighted_score > 0.5
        
        return is_tampered, weighted_score
    
    def predict(self, image_path, method='weighted'):
        """
        预测图像是否篡改
        
        Args:
            image_path: 图像路径
            method: 预测方法 ('voting' 或 'weighted')
            
        Returns:
            is_tampered: 是否篡改
            confidence: 置信度
            features: 各特征结果
        """
        features = self.extract_features(image_path)
        
        if method == 'voting':
            is_tampered, confidence = self.predict_voting(image_path)
        else:
            is_tampered, confidence = self.predict_weighted(image_path)
        
        return is_tampered, confidence, features


def optimize_thresholds(pipeline, data_dir, categories=['easy', 'difficult', 'good']):
    """
    优化各特征的阈值
    
    Args:
        pipeline: Pipeline实例
        data_dir: 数据目录
        categories: 数据类别
        
    Returns:
        best_thresholds: 最优阈值
        best_score: 最优分数
    """
    from sklearn.model_selection import ParameterGrid
    
    # 定义参数搜索空间
    param_grid = {
        'ela': [5, 10, 15, 20, 25, 30],
        'dct': [0.3, 0.4, 0.5, 0.6, 0.7],
        'noise': [0.3, 0.4, 0.5, 0.6, 0.7],
        'fft': [0.2, 0.3, 0.4, 0.5]
    }
    
    best_score = 0
    best_thresholds = {}
    
    # 简化：逐个特征优化
    for feat_name in pipeline.features:
        if feat_name not in param_grid:
            continue
            
        print(f"优化 {feat_name} 阈值...")
        best_feat_score = 0
        best_feat_threshold = DEFAULT_THRESHOLDS[feat_name]
        
        for threshold in param_grid[feat_name]:
            pipeline.thresholds[feat_name] = threshold
            
            # 计算分数
            correct = 0
            total = 0
            
            for category in categories:
                if category == 'good':
                    img_dir = os.path.join(data_dir, category)
                else:
                    img_dir = os.path.join(data_dir, category, 'images')
                
                if not os.path.exists(img_dir):
                    continue
                
                for img_file in os.listdir(img_dir):
                    if not img_file.endswith(('.jpg', '.png')):
                        continue
                    
                    img_path = os.path.join(img_dir, img_file)
                    features = pipeline.extract_features(img_path)
                    
                    if feat_name in features:
                        is_tampered, score = features[feat_name]
                        # good类应该不被检测为篡改
                        if category == 'good':
                            if not is_tampered:
                                correct += 1
                        else:
                            if is_tampered:
                                correct += 1
                        total += 1
            
            if total > 0:
                score = correct / total
                if score > best_feat_score:
                    best_feat_score = score
                    best_feat_threshold = threshold
        
        best_thresholds[feat_name] = best_feat_threshold
        print(f"  最优阈值: {best_feat_threshold}, 分数: {best_feat_score:.2%}")
    
    return best_thresholds, best_score


if __name__ == '__main__':
    # 测试Pipeline
    pipeline = TamperDetectionPipeline()
    
    # 测试单张图片
    test_img = '/tmp/forgery/tamper_data/easy/images/0037_3.jpg'
    is_tampered, confidence, features = pipeline.predict(test_img)
    
    print(f"图像: {test_img}")
    print(f"是否篡改: {is_tampered}")
    print(f"置信度: {confidence:.4f}")
    print(f"各特征结果:")
    for feat_name, (detected, score) in features.items():
        print(f"  {feat_name}: 检测={detected}, 分数={score:.4f}")