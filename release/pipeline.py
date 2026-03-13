#!/usr/bin/env python3
"""
图像篡改检测 Pipeline - 发布版本

使用方法:
    from release.pipeline import ForgeryDetector
    
    detector = ForgeryDetector()
    result = detector.predict('image.jpg')
    print(f"是否篡改: {result['is_tampered']}")
    print(f"置信度: {result['confidence']:.4f}")
"""

import os
import sys
import pickle
import numpy as np

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train.features import FEATURE_NAMES, extract_all_features


class ForgeryDetector:
    """
    图像篡改检测器
    
    用于检测图像是否经过拼接、复制粘贴、修饰等篡改操作。
    
    Example:
        >>> detector = ForgeryDetector()
        >>> result = detector.predict('test.jpg')
        >>> print(result)
        {
            'is_tampered': True,
            'confidence': 0.92,
            'probability': 0.92
        }
    """
    
    def __init__(self, model_path=None, scaler_path=None, threshold=None):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径 (默认使用内置模型)
            scaler_path: 标准化器文件路径 (默认使用内置标准化器)
            threshold: 分类阈值 (默认从配置读取，或0.85)
        """
        self.model = None
        self.scaler = None
        self.threshold = threshold
        self.feature_names = FEATURE_NAMES
        
        # 默认路径
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'release', 'models', 'model.pkl')
        if scaler_path is None:
            scaler_path = os.path.join(PROJECT_ROOT, 'release', 'models', 'scaler.pkl')
        
        # 加载模型
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载标准化器
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
        
        # 加载配置
        if self.threshold is None:
            config_path = os.path.join(PROJECT_ROOT, 'release', 'models', 'config.json')
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.threshold = config.get('optimal_threshold', 0.85)
            else:
                self.threshold = 0.85
    
    def predict(self, image_path):
        """
        预测图片是否被篡改
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: {
                'is_tampered': bool,    # 是否篡改
                'confidence': float,    # 置信度
                'probability': float    # 篡改概率
            }
        """
        # 提取特征
        features = extract_all_features(image_path)
        
        # 标准化
        features_scaled = self.scaler.transform([features])
        
        # 预测概率
        prob = self.model.predict_proba(features_scaled)[0][1]
        
        # 应用阈值
        is_tampered = prob >= self.threshold
        
        return {
            'is_tampered': bool(is_tampered),
            'confidence': float(prob),
            'probability': float(prob)
        }
    
    def predict_batch(self, image_paths):
        """
        批量预测图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result['image_path'] = path
                result['error'] = None
            except Exception as e:
                result = {
                    'image_path': path,
                    'is_tampered': None,
                    'confidence': None,
                    'probability': None,
                    'error': str(e)
                }
            results.append(result)
        return results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='图像篡改检测')
    parser.add_argument('image', type=str, help='图片路径')
    parser.add_argument('--threshold', type=float, default=None, help='分类阈值')
    args = parser.parse_args()
    
    detector = ForgeryDetector(threshold=args.threshold)
    result = detector.predict(args.image)
    
    print(f"\n检测结果:")
    print(f"  图片: {args.image}")
    print(f"  是否篡改: {'是' if result['is_tampered'] else '否'}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  篡改概率: {result['probability']:.4f}")


if __name__ == '__main__':
    main()
