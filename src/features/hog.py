"""Histogram of Oriented Gradients (HOG) based tamper detection.

HOG features can reveal inconsistencies in edge patterns.
"""
import numpy as np
import cv2
from .base import BaseFeatureDetector


class HOGDetector(BaseFeatureDetector):
    """HOG-based tamper detection."""
    
    name = "HOG"
    description = "Histogram of Oriented Gradients analysis"
    
    def __init__(self, cell_size: int = 8, block_size: int = 2, nbins: int = 9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Compute HOG-based feature map.
        
        Returns gradient magnitude map (simplified HOG visualization).
        """
        gray = self.preprocess(image).astype(np.float32)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        
        # Compute magnitude and angle
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        angle[angle < 0] += 360
        
        return self.normalize(magnitude)


class HOGLocalVarianceDetector(BaseFeatureDetector):
    """Analyze local variance in HOG features."""
    
    name = "HOG_Variance"
    description = "HOG local variance analysis"
    
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Compute local HOG variance."""
        gray = self.preprocess(image).astype(np.float32)
        h, w = gray.shape
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute local variance
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - self.block_size + 1, self.block_size // 2):
            for j in range(0, w - self.block_size + 1, self.block_size // 2):
                block = magnitude[i:i+self.block_size, j:j+self.block_size]
                
                # Compute variance
                var = np.var(block)
                feature_map[i:i+self.block_size, j:j+self.block_size] = var
        
        return self.normalize(feature_map)


class GradientInconsistencyDetector(BaseFeatureDetector):
    """Detect gradient direction inconsistencies."""
    
    name = "GRAD_Inconsistency"
    description = "Gradient direction inconsistency analysis"
    
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Analyze gradient direction consistency."""
        gray = self.preprocess(image).astype(np.float32)
        h, w = gray.shape
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute angle
        angle = np.arctan2(grad_y, grad_x)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Check angle consistency with neighbors
                center = angle[i, j]
                neighbors = [
                    angle[i-1, j-1], angle[i-1, j], angle[i-1, j+1],
                    angle[i, j-1], angle[i, j+1],
                    angle[i+1, j-1], angle[i+1, j], angle[i+1, j+1]
                ]
                
                # Compute angular difference
                diffs = [np.abs(np.sin(center - n)) for n in neighbors]
                mean_diff = np.mean(diffs)
                
                # Weight by magnitude
                feature_map[i, j] = mean_diff * magnitude[i, j]
        
        return self.normalize(feature_map)