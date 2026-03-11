"""Local Binary Pattern analysis for tamper detection.

LBP can reveal texture inconsistencies caused by manipulation.
"""
import numpy as np
import cv2
from .base import BaseFeatureDetector


class LBPDetector(BaseFeatureDetector):
    """Local Binary Pattern based tamper detection."""
    
    name = "LBP"
    description = "Local Binary Pattern texture analysis"
    
    def __init__(self, radius: int = 1, method: str = 'uniform'):
        self.radius = radius
        self.method = method
        self.n_points = 8 * radius
    
    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """Compute LBP image."""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        # Direction offsets
        angles = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        dx = np.round(self.radius * np.cos(angles)).astype(int)
        dy = np.round(self.radius * np.sin(angles)).astype(int)
        
        for i in range(self.radius, h - self.radius):
            for j in range(self.radius, w - self.radius):
                center = image[i, j]
                binary = 0
                for k, (ox, oy) in enumerate(zip(dx, dy)):
                    if image[i + oy, j + ox] >= center:
                        binary |= (1 << k)
                lbp[i, j] = binary
        
        return lbp
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Compute LBP texture map."""
        gray = self.preprocess(image)
        lbp = self._compute_lbp(gray.astype(np.float32))
        return lbp


class LBPConsistencyDetector(BaseFeatureDetector):
    """Detect LBP consistency across image regions."""
    
    name = "LBP_Consistency"
    description = "LBP texture consistency analysis"
    
    def __init__(self, block_size: int = 32):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Analyze LBP histogram consistency."""
        gray = self.preprocess(image)
        h, w = gray.shape
        
        # Compute LBP
        lbp_detector = LBPDetector(radius=1)
        lbp = lbp_detector._compute_lbp(gray.astype(np.float32))
        
        # Analyze local histogram entropy
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - self.block_size + 1, self.block_size // 2):
            for j in range(0, w - self.block_size + 1, self.block_size // 2):
                block = lbp[i:i+self.block_size, j:j+self.block_size]
                
                # Compute histogram entropy
                hist = np.bincount(block.flatten(), minlength=256)
                hist = hist.astype(np.float32)
                hist = hist[hist > 0]
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log2(hist + 1e-8))
                
                feature_map[i:i+self.block_size, j:j+self.block_size] = entropy
        
        return self.normalize(feature_map)