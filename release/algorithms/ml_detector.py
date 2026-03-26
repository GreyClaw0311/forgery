#!/usr/bin/env python3
"""
机器学习串联检测器 - 优化版

优化点:
1. 全局特征预计算 - 避免每个窗口重复 JPEG 编解码
2. 滑动窗口统计 - 从预计算图取值
3. 自动特征维度适配

预期加速: 5-10x
"""

import os
import sys
import pickle
import time
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
    机器学习串联检测器 - 优化版
    
    流程:
    1. GB 分类器判断是否篡改
    2. 全局特征预计算 (ELA/DCT/Noise等)
    3. 滑动窗口统计特征
    4. 像素级 ML 定位篡改区域
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
            device: GPU 设备 (仅用于兼容，当前模型不支持GPU)
        """
        self.device = device
        self.gb_model = None
        self.gb_scaler = None
        self.pixel_model = None
        self.pixel_scaler = None
        self.pixel_threshold = 0.5
        self.pixel_feature_dim = 57
        self.use_lbp = True  # 默认使用LBP
        
        # 默认路径
        if gb_model_path is None:
            gb_model_path = os.path.join(PROJECT_ROOT, 'models', 'gb_classifier', 'model.pkl')
        if gb_scaler_path is None:
            gb_scaler_path = os.path.join(PROJECT_ROOT, 'models', 'gb_classifier', 'scaler.pkl')
        if pixel_model_path is None:
            pixel_model_path = os.path.join(PROJECT_ROOT, 'models', 'pixel_segmentation', 'model.pkl')
        
        # 加载模型
        self._load_gb_model(gb_model_path, gb_scaler_path)
        self._load_pixel_model(pixel_model_path)
    
    def _load_gb_model(self, model_path: str, scaler_path: str):
        """加载 GB 分类器"""
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.gb_model = pickle.load(f)
                print(f"GB 分类器已加载: {model_path}")
            except Exception as e:
                print(f"GB 分类器加载失败: {e}")
        else:
            print(f"警告: GB 模型不存在: {model_path}")
        
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.gb_scaler = pickle.load(f)
            except Exception as e:
                print(f"GB 标准化器加载失败: {e}")
    
    def _load_pixel_model(self, model_path: str):
        """加载像素级模型"""
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                self.pixel_model = data.get('model')
                self.pixel_scaler = data.get('scaler')
                self.pixel_threshold = data.get('threshold', 0.5)
                
                # 从模型配置中读取特征维度
                model_config = data.get('config', {})
                self.pixel_feature_dim = model_config.get('feature_dim', 57)
                self.use_lbp = model_config.get('use_lbp', True)
                
                # 自动检测特征维度（从scaler）
                if self.pixel_scaler is not None:
                    detected_dim = self.pixel_scaler.n_features_in_
                    if detected_dim != self.pixel_feature_dim:
                        print(f"检测到特征维度不一致: 配置={self.pixel_feature_dim}, scaler={detected_dim}")
                        self.pixel_feature_dim = detected_dim
                    
                    # 根据特征维度判断是否使用LBP
                    if self.pixel_feature_dim == 49:
                        self.use_lbp = False
                    elif self.pixel_feature_dim == 57:
                        self.use_lbp = True
                    
                    print(f"像素级模型已加载: {model_path}")
                    print(f"  特征维度: {self.pixel_feature_dim}")
                    print(f"  使用LBP: {self.use_lbp}")
                else:
                    self.pixel_feature_dim = 57
                    self.use_lbp = True
                    print(f"像素级模型已加载: {model_path} (默认配置)")
            except Exception as e:
                print(f"像素级模型加载失败: {e}")
        else:
            print(f"警告: 像素级模型不存在: {model_path}")
    
    def predict(self, image_path: str) -> Dict:
        """
        检测图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            {
                'is_tampered': bool,
                'confidence': float,
                'mask': np.ndarray,
                'preprocess_time': float,
                'inference_time': float
            }
        """
        result = {
            'is_tampered': False,
            'confidence': 0.0,
            'mask': None,
            'preprocess_time': 0.0,
            'inference_time': 0.0
        }
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            return result
        
        # 1. GB 分类器前置检测
        if self.gb_model is not None:
            try:
                from algorithms.features import extract_all_features
                features = extract_all_features(image_path)
                if features is not None and self.gb_scaler is not None:
                    features_scaled = self.gb_scaler.transform([features])
                    proba = self.gb_model.predict_proba(features_scaled)[0, 1]
                    result['is_tampered'] = bool(proba > 0.5)
                    result['confidence'] = float(proba)
                else:
                    result['is_tampered'] = True
                    result['confidence'] = 0.5
            except Exception as e:
                print(f"GB 分类错误: {e}")
                result['is_tampered'] = True
                result['confidence'] = 0.5
        else:
            result['is_tampered'] = True
            result['confidence'] = 1.0
        
        # 如果 GB 判断为正常，直接返回
        if not result['is_tampered']:
            return result
        
        # 2. 像素级检测 (使用全局预计算优化)
        if self.pixel_model is None:
            return result
        
        try:
            preprocess_start = time.time()
            
            # 使用优化版特征提取
            from algorithms.features import GlobalFeatureCache, FastPixelFeatureExtractor
            
            # 全局特征预计算
            cache = GlobalFeatureCache(image, quality=90)
            
            # 创建快速提取器 (根据模型配置决定是否使用LBP)
            extractor = FastPixelFeatureExtractor(
                window_size=32,
                feature_dim=self.pixel_feature_dim,
                use_lbp=self.use_lbp
            )
            extractor.set_cache(cache)
            
            # 滑动窗口
            h, w = image.shape[:2]
            window_size = 32
            stride = 16  # 可以增大到 32 进一步加速
            half = window_size // 2
            
            features_list = []
            positions = []
            
            for y in range(half, h - half, stride):
                for x in range(half, w - half, stride):
                    feat = extractor.extract_from_cache(y, x)
                    features_list.append(feat)
                    positions.append((y, x))
            
            preprocess_time = time.time() - preprocess_start
            result['preprocess_time'] = preprocess_time
            
            if len(features_list) == 0:
                return result
            
            # 模型推理
            inference_start = time.time()
            
            features = np.array(features_list)
            features_scaled = self.pixel_scaler.transform(features)
            proba = self.pixel_model.predict_proba(features_scaled)[:, 1]
            
            inference_time = time.time() - inference_start
            result['inference_time'] = inference_time
            
            # 生成掩码
            mask = self._generate_mask(proba, positions, (h, w))
            result['mask'] = mask
            
        except Exception as e:
            print(f"像素级检测错误: {e}")
            import traceback
            traceback.print_exc()
        
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


# GPU 兼容版 (实际仍使用 CPU)
class MLDetectorGPU(MLDetector):
    """GPU 兼容版 ML 检测器
    
    注意: 当前模型 (sklearn RF/GB) 仅支持 CPU
    此类保留 GPU 接口以便将来扩展
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if HAS_TORCH:
            try:
                if torch.cuda.is_available():
                    self.device = torch.device(kwargs.get('device', 'cuda:0'))
                    print(f"GPU 可用: {self.device}")
                    print("注意: 当前 sklearn 模型仅支持 CPU，GPU 用于其他操作")
                else:
                    print("GPU 不可用，使用 CPU")
            except Exception as e:
                print(f"GPU 初始化失败: {e}")
                print("使用 CPU")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ML 串联检测器 - 优化版')
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU 设备')
    
    args = parser.parse_args()
    
    print("加载模型...")
    detector = MLDetectorGPU(device=args.device)
    
    print(f"\n检测图片: {args.image}")
    start_time = time.time()
    result = detector.predict(args.image)
    total_time = time.time() - start_time
    
    print(f"\n结果:")
    print(f"  是否篡改: {result['is_tampered']}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  预处理时间: {result['preprocess_time']*1000:.1f}ms")
    print(f"  推理时间: {result['inference_time']*1000:.1f}ms")
    print(f"  总耗时: {total_time*1000:.1f}ms")
    
    if result['mask'] is not None:
        cv2.imwrite('output_mask.png', result['mask'])
        print(f"\n掩码已保存: output_mask.png")