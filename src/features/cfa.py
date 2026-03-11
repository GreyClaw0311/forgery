"""Color Filter Array (CFA) based tamper detection.

CFA interpolation artifacts can reveal image manipulation.
"""
import numpy as np
import cv2
from .base import BaseFeatureDetector


class CFADetector(BaseFeatureDetector):
    """Color Filter Array artifact detector."""
    
    name = "CFA"
    description = "Detects CFA interpolation inconsistencies"
    
    def __init__(self, block_size: int = 8):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect CFA interpolation artifacts.
        
        Analyzes the periodic pattern caused by Bayer filter interpolation.
        """
        # Convert to grayscale if needed
        gray = self.preprocess(image).astype(np.float32)
        
        # Compute gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze periodicity in blocks
        h, w = gray.shape
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - self.block_size, self.block_size // 2):
            for j in range(0, w - self.block_size, self.block_size // 2):
                block = magnitude[i:i+self.block_size, j:j+self.block_size]
                
                # Compute variance as a measure of CFA artifacts
                var = np.var(block)
                feature_map[i:i+self.block_size, j:j+self.block_size] = var
        
        return self.normalize(feature_map)


class CFAInterpolationDetector(BaseFeatureDetector):
    """Detect inconsistencies in CFA interpolation patterns."""
    
    name = "CFA_Interpolation"
    description = "Detects CFA interpolation pattern inconsistencies"
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Analyze CFA interpolation consistency."""
        if len(image.shape) != 3:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Extract individual color channels
        b, g, r = cv2.split(image)
        
        # Analyze high-frequency patterns in each channel
        feature_maps = []
        
        for channel in [b, g, r]:
            # Apply high-pass filter
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]]) / 8.0
            filtered = cv2.filter2D(channel.astype(np.float32), -1, kernel)
            feature_maps.append(np.abs(filtered))
        
        # Combine channel features
        combined = np.mean(feature_maps, axis=0)
        
        return self.normalize(combined)