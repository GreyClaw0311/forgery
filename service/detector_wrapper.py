"""
Forgery Detector Wrapper - 检测器封装

整合所有检测算法，提供统一接口
"""

import os
import sys
import pickle
import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.ela_detector import ELADetector
from src.detection.dct_detector import DCTBlockDetector
from src.detection.noise_detector import NoiseConsistencyDetector
from src.detection.copy_move_detector import CopyMoveDetector
from src.detection.fusion import AdaptiveFusion
from src.pipeline import ForgeryRegionDetector
from src.utils.visualization import overlay_mask


class ForgeryDetectorWrapper:
    """
    检测器封装类
    
    整合所有检测算法，提供统一的检测接口
    """
    
    def __init__(self, model_dir: str = None):
        """
        初始化检测器
        
        Args:
            model_dir: 模型文件目录
        """
        self.model_dir = Path(model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        ))
        
        # 初始化各检测器
        self.ela_detector = ELADetector()
        self.dct_detector = DCTBlockDetector()
        self.noise_detector = NoiseConsistencyDetector(block_size=32)
        self.copy_move_detector = CopyMoveDetector()
        self.fusion = AdaptiveFusion()
        
        # 完整 Pipeline
        self.pipeline = ForgeryRegionDetector(
            use_methods=['ela', 'dct', 'noise'],
            fusion_threshold=0.2
        )
        
        # 像素级 ML 模型
        self.pixel_ml_model = None
        self.pixel_ml_scaler = None
        self.pixel_ml_config = None
        self._load_pixel_ml_model()
        
        print("检测器初始化完成")
    
    def _load_pixel_ml_model(self):
        """加载像素级 ML 模型"""
        model_path = self.model_dir / "pixel_segmentation" / "model.pkl"
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                self.pixel_ml_model = data['model']
                self.pixel_ml_scaler = data['scaler']
                self.pixel_ml_config = data.get('config', {
                    'window_size': 32,
                    'stride': 16,
                    'feature_dim': 57
                })
                print(f"像素级 ML 模型加载成功: {model_path}")
            except Exception as e:
                print(f"像素级 ML 模型加载失败: {e}")
        else:
            print(f"像素级 ML 模型文件不存在: {model_path}")
    
    def detect(self, image: np.ndarray, algorithm: str) -> Dict[str, Any]:
        """
        执行检测
        
        Args:
            image: 输入图像 (BGR)
            algorithm: 算法名称
            
        Returns:
            {
                "is_tampered": bool,
                "confidence": float,
                "mask_base64": str,
                "tampered_ratio": float
            }
        """
        if algorithm == "ela":
            return self._detect_ela(image)
        elif algorithm == "dct":
            return self._detect_dct(image)
        elif algorithm == "noise":
            return self._detect_noise(image)
        elif algorithm == "copy_move":
            return self._detect_copy_move(image)
        elif algorithm == "fusion":
            return self._detect_fusion(image)
        elif algorithm == "pixel_ml":
            return self._detect_pixel_ml(image)
        elif algorithm == "pipeline":
            return self._detect_pipeline(image)
        else:
            raise ValueError(f"未知算法: {algorithm}")
    
    def _detect_ela(self, image: np.ndarray) -> Dict[str, Any]:
        """ELA 检测"""
        heatmap = self.ela_detector.detect(image)
        mask = self.ela_detector.get_mask(heatmap, threshold=0.2)
        
        # 计算篡改比例和置信度
        tampered_ratio = np.mean(mask > 0)
        confidence = float(np.max(heatmap))
        is_tampered = tampered_ratio > 0.01
        
        # 生成可视化结果
        result_image = self._generate_result_image(image, mask, heatmap)
        
        return {
            "is_tampered": is_tampered,
            "confidence": confidence,
            "mask_base64": self._image_to_base64(result_image),
            "tampered_ratio": float(tampered_ratio)
        }
    
    def _detect_dct(self, image: np.ndarray) -> Dict[str, Any]:
        """DCT 块效应检测"""
        heatmap = self.dct_detector.detect(image)
        mask = self.dct_detector.get_mask(heatmap, threshold=0.3)
        
        tampered_ratio = np.mean(mask > 0)
        confidence = float(np.max(heatmap))
        is_tampered = tampered_ratio > 0.01
        
        result_image = self._generate_result_image(image, mask, heatmap)
        
        return {
            "is_tampered": is_tampered,
            "confidence": confidence,
            "mask_base64": self._image_to_base64(result_image),
            "tampered_ratio": float(tampered_ratio)
        }
    
    def _detect_noise(self, image: np.ndarray) -> Dict[str, Any]:
        """噪声一致性检测"""
        heatmap, mask, _ = self.noise_detector.detect_full(image)
        
        tampered_ratio = np.mean(mask > 0)
        confidence = float(np.max(heatmap)) if heatmap.max() > 0 else 0.0
        is_tampered = tampered_ratio > 0.01
        
        result_image = self._generate_result_image(image, mask, heatmap)
        
        return {
            "is_tampered": is_tampered,
            "confidence": confidence,
            "mask_base64": self._image_to_base64(result_image),
            "tampered_ratio": float(tampered_ratio)
        }
    
    def _detect_copy_move(self, image: np.ndarray) -> Dict[str, Any]:
        """复制移动检测"""
        mask = self.copy_move_detector.detect(image)
        
        tampered_ratio = np.mean(mask > 0)
        confidence = tampered_ratio
        is_tampered = tampered_ratio > 0.01
        
        result_image = self._generate_result_image(image, mask, None)
        
        return {
            "is_tampered": is_tampered,
            "confidence": float(confidence),
            "mask_base64": self._image_to_base64(result_image),
            "tampered_ratio": float(tampered_ratio)
        }
    
    def _detect_fusion(self, image: np.ndarray) -> Dict[str, Any]:
        """多检测器融合检测"""
        # 各检测器结果
        heatmaps = {
            'ela': self.ela_detector.detect(image),
            'dct': self.dct_detector.detect(image),
            'noise': self.noise_detector.detect_full(image)[0]
        }
        
        # 自适应融合
        fused_heatmap = self.fusion.fusion_adaptive(heatmaps)
        
        # 生成 mask
        mask = (fused_heatmap > 0.2).astype(np.uint8) * 255
        
        tampered_ratio = np.mean(mask > 0)
        confidence = float(np.max(fused_heatmap))
        is_tampered = tampered_ratio > 0.01
        
        result_image = self._generate_result_image(image, mask, fused_heatmap)
        
        return {
            "is_tampered": is_tampered,
            "confidence": confidence,
            "mask_base64": self._image_to_base64(result_image),
            "tampered_ratio": float(tampered_ratio)
        }
    
    def _detect_pixel_ml(self, image: np.ndarray) -> Dict[str, Any]:
        """像素级 ML 检测"""
        if self.pixel_ml_model is None:
            raise RuntimeError("像素级 ML 模型未加载")
        
        # 使用 pipeline 的像素级预测
        result = self.pipeline.detect(image)
        mask = result['mask']
        heatmap = result['heatmap']
        
        tampered_ratio = np.mean(mask > 0)
        confidence = float(np.max(heatmap)) if heatmap.max() > 0 else 0.0
        is_tampered = tampered_ratio > 0.01
        
        result_image = self._generate_result_image(image, mask, heatmap)
        
        return {
            "is_tampered": is_tampered,
            "confidence": confidence,
            "mask_base64": self._image_to_base64(result_image),
            "tampered_ratio": float(tampered_ratio)
        }
    
    def _detect_pipeline(self, image: np.ndarray) -> Dict[str, Any]:
        """完整 Pipeline 检测"""
        result = self.pipeline.detect(image)
        mask = result['mask']
        heatmap = result['heatmap']
        
        tampered_ratio = np.mean(mask > 0)
        confidence = float(np.max(heatmap)) if heatmap.max() > 0 else 0.0
        is_tampered = tampered_ratio > 0.01
        
        result_image = self._generate_result_image(image, mask, heatmap)
        
        return {
            "is_tampered": is_tampered,
            "confidence": confidence,
            "mask_base64": self._image_to_base64(result_image),
            "tampered_ratio": float(tampered_ratio)
        }
    
    def _generate_result_image(self, 
                                original: np.ndarray, 
                                mask: np.ndarray, 
                                heatmap: Optional[np.ndarray] = None) -> np.ndarray:
        """
        生成可视化结果图像
        
        包含：原图、Mask 叠加、热力图（可选）
        """
        # 确保原图是 BGR
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # 创建叠加图像
        overlay = original.copy()
        
        # 红色标记篡改区域
        if mask is not None and mask.max() > 0:
            mask_bool = mask > 0
            overlay[mask_bool] = [0, 0, 255]  # 红色
            result = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
        else:
            result = original.copy()
        
        # 添加热力图
        if heatmap is not None and heatmap.max() > 0:
            heatmap_normalized = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            result = cv2.addWeighted(result, 0.5, heatmap_colored, 0.5, 0)
        
        return result
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """将图像编码为 Base64"""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')