#!/usr/bin/env python3
"""
机器学习串联检测器

流程:
1. GradientBoost 分类器判断图片是否篡改
2. 如果篡改，使用像素级 ML 检测篡改区域

支持 GPU 推理加速
"""

import os
import sys
import pickle
import numpy as np
import cv2
from typing import Dict, Tuple, Optional

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# GPU 支持
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MLDetector:
    """
    机器学习串联检测器
    
    流程:
    1. GradientBoost 判断是否篡改
    2. 像素级 ML 定位篡改区域
    """
    
    def __init__(self,
                 gb_model_path: str = None,
                 gb_scaler_path: str = None,
                 pixel_model_path: str = None,
                 device: str = 'cuda:0'):
        """
        初始化检测器
        
        Args:
            gb_model_path: GB 分类器模型路径
            gb_scaler_path: GB 标准化器路径
            pixel_model_path: 像素级模型路径
            device: GPU 设备 (cuda:0 / cuda:1 / cpu)
        """
        self.device = device
        self.gb_model = None
        self.gb_scaler = None
        self.pixel_model = None
        self.pixel_scaler = None
        self.pixel_threshold = 0.5
        
        # 默认路径
        if gb_model_path is None:
            gb_model_path = os.path.join(PROJECT_ROOT, 'models', 'gb_classifier', 'model.pkl')
        if gb_scaler_path is None:
            gb_scaler_path = os.path.join(PROJECT_ROOT, 'models', 'gb_classifier', 'scaler.pkl')
        if pixel_model_path is None:
            pixel_model_path = os.path.join(PROJECT_ROOT, 'models', 'pixel_segmentation', 'model.pkl')
        
        # 加载 GB 分类器
        self._load_gb_model(gb_model_path, gb_scaler_path)
        
        # 加载像素级模型
        self._load_pixel_model(pixel_model_path)
    
    def _load_gb_model(self, model_path: str, scaler_path: str):
        """加载 GB 分类器"""
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.gb_model = pickle.load(f)
            print(f"GB 分类器已加载: {model_path}")
        else:
            print(f"警告: GB 模型不存在: {model_path}")
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.gb_scaler = pickle.load(f)
    
    def _load_pixel_model(self, model_path: str):
        """加载像素级模型"""
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            self.pixel_model = data.get('model')
            self.pixel_scaler = data.get('scaler')
            self.pixel_threshold = data.get('threshold', 0.5)
            print(f"像素级模型已加载: {model_path}")
        else:
            print(f"警告: 像素级模型不存在: {model_path}")
    
    def _extract_gb_features(self, image_path: str) -> np.ndarray:
        """提取 GB 分类器所需的特征"""
        from algorithms.features import extract_all_features
        return extract_all_features(image_path)
    
    def _extract_pixel_features(self, image: np.ndarray) -> Tuple[np.ndarray, list]:
        """提取像素级特征 (GPU 加速)"""
        h, w = image.shape[:2]
        window_size = 32
        stride = 16
        half = window_size // 2
        
        features_list = []
        positions = []
        
        for y in range(half, h - half, stride):
            for x in range(half, w - half, stride):
                patch = image[y-half:y+half, x-half:x+half]
                if patch.shape[0] != window_size or patch.shape[1] != window_size:
                    continue
                
                feat = self._extract_patch_features(patch)
                features_list.append(feat)
                positions.append((y, x))
        
        if not features_list:
            return np.array([]), []
        
        return np.array(features_list), positions
    
    def _extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """从图像块提取特征"""
        from algorithms.features import PixelFeatureExtractor
        extractor = PixelFeatureExtractor(32)
        return extractor.extract(patch)
    
    def predict(self, image_path: str) -> Dict:
        """
        检测图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            {
                'is_tampered': bool,      # 是否篡改
                'confidence': float,      # 置信度
                'mask': np.ndarray,       # 篡改区域掩码
                'mask_base64': str        # 掩码 Base64
            }
        """
        result = {
            'is_tampered': False,
            'confidence': 0.0,
            'mask': None,
            'mask_base64': None
        }
        
        # 1. GB 分类器判断是否篡改
        if self.gb_model is None:
            # 模型未加载，跳过分类直接检测
            result['is_tampered'] = True
            result['confidence'] = 1.0
        else:
            try:
                features = self._extract_gb_features(image_path)
                if features is not None and self.gb_scaler is not None:
                    features_scaled = self.gb_scaler.transform([features])
                    proba = self.gb_model.predict_proba(features_scaled)[0, 1]
                    result['is_tampered'] = proba > 0.5
                    result['confidence'] = float(proba)
                else:
                    result['is_tampered'] = True
                    result['confidence'] = 0.5
            except Exception as e:
                print(f"GB 分类错误: {e}")
                result['is_tampered'] = True
                result['confidence'] = 0.5
        
        # 2. 如果篡改，定位区域
        if result['is_tampered'] and self.pixel_model is not None:
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    features, positions = self._extract_pixel_features(image)
                    
                    if len(features) > 0:
                        features_scaled = self.pixel_scaler.transform(features)
                        proba = self.pixel_model.predict_proba(features_scaled)[:, 1]
                        
                        # 生成掩码
                        h, w = image.shape[:2]
                        mask = self._generate_mask(proba, positions, (h, w))
                        result['mask'] = mask
                        
                        # 转换为 Base64
                        result['mask_base64'] = self._mask_to_base64(mask)
            except Exception as e:
                print(f"像素级检测错误: {e}")
        
        return result
    
    def _generate_mask(self, proba: np.ndarray, positions: list, 
                       shape: Tuple[int, int]) -> np.ndarray:
        """生成篡改区域掩码"""
        h, w = shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        stride = 16
        half_stride = stride // 2
        
        for (y, x), p in zip(positions, proba):
            y1 = max(0, y - half_stride)
            y2 = min(h, y + half_stride)
            x1 = max(0, x - half_stride)
            x2 = min(w, x + half_stride)
            
            heatmap[y1:y2, x1:x2] += p
            count_map[y1:y2, x1:x2] += 1
        
        # 归一化
        mask = count_map > 0
        heatmap[mask] = heatmap[mask] / count_map[mask]
        
        # 二值化
        binary_mask = (heatmap > self.pixel_threshold).astype(np.uint8) * 255
        
        # 后处理
        binary_mask = self._postprocess(binary_mask)
        
        return binary_mask
    
    def _postprocess(self, mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """后处理"""
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        result = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                result[labels == i] = 255
        
        return result
    
    def _mask_to_base64(self, mask: np.ndarray) -> str:
        """掩码转 Base64"""
        import base64
        _, buffer = cv2.imencode('.png', mask)
        return base64.b64encode(buffer).decode('utf-8')


# GPU 加速版本
class MLDetectorGPU(MLDetector):
    """GPU 加速版 ML 检测器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if HAS_TORCH and torch.cuda.is_available():
            self.device = torch.device(kwargs.get('device', 'cuda:0'))
            print(f"使用 GPU: {self.device}")
        else:
            self.device = None
            print("GPU 不可用，使用 CPU")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ML 串联检测器')
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU 设备')
    
    args = parser.parse_args()
    
    detector = MLDetectorGPU(device=args.device)
    result = detector.predict(args.image)
    
    print(f"是否篡改: {result['is_tampered']}")
    print(f"置信度: {result['confidence']:.4f}")
    
    if result['mask'] is not None:
        cv2.imwrite('output_mask.png', result['mask'])
        print("掩码已保存: output_mask.png")