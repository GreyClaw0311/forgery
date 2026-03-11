"""Blocking artifact detection.

JPEG compression creates 8x8 block artifacts. Inconsistencies can reveal tampering.
"""
import numpy as np
import cv2
from .base import BaseFeatureDetector


class BlockingArtifactDetector(BaseFeatureDetector):
    """Detect JPEG blocking artifacts."""
    
    name = "BLK"
    description = "JPEG blocking artifact detection"
    
    def __init__(self, block_size: int = 8):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect blocking artifacts at JPEG block boundaries.
        
        Tampered images may show inconsistent block patterns.
        """
        gray = self.preprocess(image).astype(np.float32)
        h, w = gray.shape
        
        # Compute horizontal and vertical differences at block boundaries
        block_feature = np.zeros((h, w), dtype=np.float32)
        
        # Horizontal block boundaries
        for j in range(self.block_size, w - 1, self.block_size):
            diff = np.abs(gray[:, j] - gray[:, j - 1])
            block_feature[:, j] = diff
        
        # Vertical block boundaries
        for i in range(self.block_size, h - 1, self.block_size):
            diff = np.abs(gray[i, :] - gray[i - 1, :])
            block_feature[i, :] += diff
        
        return self.normalize(block_feature)


class BlockingGridDetector(BaseFeatureDetector):
    """Detect blocking grid inconsistencies."""
    
    name = "BLK_Grid"
    description = "Detects inconsistent block grid alignment"
    
    def __init__(self, block_size: int = 8):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Analyze block grid alignment consistency."""
        gray = self.preprocess(image).astype(np.float32)
        h, w = gray.shape
        
        # Compute block variance map
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                block = gray[i:i+self.block_size, j:j+self.block_size]
                
                # Compute blockiness score
                # Higher variance at edges indicates blocking
                edges = np.concatenate([
                    block[0, :],   # top
                    block[-1, :],  # bottom
                    block[:, 0],   # left
                    block[:, -1]   # right
                ])
                interior = block[1:-1, 1:-1].flatten()
                
                edge_var = np.var(edges)
                interior_var = np.var(interior) + 1e-8
                
                blockiness = edge_var / interior_var
                
                feature_map[i:i+self.block_size, j:j+self.block_size] = blockiness
        
        return self.normalize(feature_map)