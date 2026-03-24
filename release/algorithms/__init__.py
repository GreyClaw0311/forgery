"""
图像篡改检测算法模块

包含:
- ela_detector: ELA (Error Level Analysis) 检测器
- dct_detector: DCT (Discrete Cosine Transform) 检测器
- fusion_detector: 多特征融合检测器
- ml_detector: 机器学习串联检测器
- features: 特征提取模块
"""

from .ela_detector import ELADetector
from .dct_detector import DCTBlockDetector
from .fusion_detector import AdaptiveFusion
from .features import extract_all_features, PixelFeatureExtractor

# 兼容性别名
DCTDetector = DCTBlockDetector
FusionDetector = AdaptiveFusion

__all__ = [
    'ELADetector', 
    'DCTBlockDetector', 
    'DCTDetector',  # 兼容性别名
    'AdaptiveFusion',
    'FusionDetector',  # 兼容性别名
    'extract_all_features',
    'PixelFeatureExtractor'
]