"""Noise analysis for tamper detection.

Analyzes noise patterns and inconsistencies across the image.
"""
import numpy as np
import cv2
from .base import BaseFeatureDetector


class NoiseDetector(BaseFeatureDetector):
    """Noise inconsistency detector."""
    
    name = "NOISE"
    description = "Detects noise pattern inconsistencies"
    
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze noise patterns across the image.
        
        Tampered regions often have different noise characteristics.
        """
        gray = self.preprocess(image).astype(np.float32)
        
        # Estimate noise using median filter residual
        median = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
        noise_residual = np.abs(gray - median)
        
        return self.normalize(noise_residual)


class NoiseVarianceDetector(BaseFeatureDetector):
    """Noise variance analysis for localized tamper detection."""
    
    name = "NOISE_Variance"
    description = "Analyzes local noise variance"
    
    def __init__(self, block_size: int = 32):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Compute local noise variance."""
        gray = self.preprocess(image).astype(np.float32)
        h, w = gray.shape
        
        # Denoise
        denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)
        noise = gray - denoised
        
        # Compute local variance
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - self.block_size + 1, self.block_size // 2):
            for j in range(0, w - self.block_size + 1, self.block_size // 2):
                block = noise[i:i+self.block_size, j:j+self.block_size]
                var = np.var(block)
                feature_map[i:i+self.block_size, j:j+self.block_size] = var
        
        return self.normalize(feature_map)


class PRNUNoiseDetector(BaseFeatureDetector):
    """Photo Response Non-Uniformity noise pattern detector."""
    
    name = "PRNU"
    description = "PRNU noise pattern analysis"
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate PRNU noise pattern.
        
        Each camera sensor has unique PRNU, inconsistencies reveal tampering.
        """
        if len(image.shape) == 3:
            # Work with green channel
            gray = image[:, :, 1].astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Denoise to extract noise pattern
        denoised = cv2.GaussianBlur(gray, (7, 7), 2.0)
        noise = gray - denoised
        
        # Normalize by intensity
        mean_intensity = cv2.GaussianBlur(gray, (7, 7), 2.0)
        mean_intensity = np.maximum(mean_intensity, 1.0)  # Avoid division by zero
        prnu = noise / mean_intensity
        
        return self.normalize(np.abs(prnu))