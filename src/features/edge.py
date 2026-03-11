"""Edge-based tamper detection.

Analyzes edge patterns and inconsistencies.
"""
import numpy as np
import cv2
from .base import BaseFeatureDetector


class EdgeDetector(BaseFeatureDetector):
    """Edge-based tamper detection."""
    
    name = "EDGE"
    description = "Edge pattern analysis"
    
    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Compute edge map using Canny detector."""
        gray = self.preprocess(image)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        return edges.astype(np.float32)


class EdgeConsistencyDetector(BaseFeatureDetector):
    """Detect edge consistency anomalies."""
    
    name = "EDGE_Consistency"
    description = "Edge direction consistency analysis"
    
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Analyze edge direction consistency."""
        gray = self.preprocess(image).astype(np.float32)
        h, w = gray.shape
        
        # Compute edges
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute edge direction
        angle = np.arctan2(grad_y, grad_x)
        
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - self.block_size + 1, self.block_size // 2):
            for j in range(0, w - self.block_size + 1, self.block_size // 2):
                edge_block = edges[i:i+self.block_size, j:j+self.block_size]
                angle_block = angle[i:i+self.block_size, j:j+self.block_size]
                
                # Only consider pixels where edges exist
                edge_pixels = edge_block > 0
                
                if np.sum(edge_pixels) > 10:  # Minimum edge pixels
                    angles = angle_block[edge_pixels]
                    
                    # Compute circular variance of angles
                    cos_mean = np.mean(np.cos(angles))
                    sin_mean = np.mean(np.sin(angles))
                    consistency = np.sqrt(cos_mean**2 + sin_mean**2)
                    
                    # Lower consistency = higher suspiciousness
                    feature_map[i:i+self.block_size, j:j+self.block_size] = 1 - consistency
        
        return self.normalize(feature_map)


class EdgeDensityDetector(BaseFeatureDetector):
    """Analyze local edge density for tamper detection."""
    
    name = "EDGE_Density"
    description = "Local edge density analysis"
    
    def __init__(self, block_size: int = 32):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Compute local edge density."""
        gray = self.preprocess(image)
        h, w = gray.shape
        
        # Compute edges
        edges = cv2.Canny(gray, 50, 150).astype(np.float32)
        
        # Compute local density
        kernel_size = self.block_size
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
        
        density = cv2.filter2D(edges, -1, kernel)
        
        return self.normalize(density)