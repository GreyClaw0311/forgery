"""DCT-based tamper detection.

Analyzes Discrete Cosine Transform coefficients for inconsistencies.
"""
import numpy as np
import cv2
from typing import Tuple
from .base import BaseFeatureDetector


class DCTDetector(BaseFeatureDetector):
    """DCT coefficient analysis for tamper detection."""
    
    name = "DCT"
    description = "DCT coefficient analysis"
    
    def __init__(self, block_size: int = 8):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze DCT coefficients for inconsistencies.
        
        Tampered regions often show abnormal DCT coefficient patterns.
        """
        gray = self.preprocess(image).astype(np.float32)
        h, w = gray.shape
        
        # Pad image to be divisible by block_size
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        
        ph, pw = padded.shape
        feature_map = np.zeros((ph, pw), dtype=np.float32)
        
        # Process blocks
        for i in range(0, ph - self.block_size + 1, self.block_size):
            for j in range(0, pw - self.block_size + 1, self.block_size):
                block = padded[i:i+self.block_size, j:j+self.block_size]
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Compute feature (energy of high-frequency coefficients)
                # Exclude DC component and low-frequency
                high_freq = dct_block[2:, 2:]
                energy = np.sum(np.abs(high_freq))
                
                feature_map[i:i+self.block_size, j:j+self.block_size] = energy
        
        # Remove padding and normalize
        feature_map = feature_map[:h, :w]
        return self.normalize(feature_map)


class DCTResidualDetector(BaseFeatureDetector):
    """DCT residual analysis for detecting double compression."""
    
    name = "DCT_Residual"
    description = "Detects DCT residual inconsistencies"
    
    def __init__(self, block_size: int = 8):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Analyze DCT residuals."""
        gray = self.preprocess(image).astype(np.float32)
        h, w = gray.shape
        
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - self.block_size + 1, self.block_size // 2):
            for j in range(0, w - self.block_size + 1, self.block_size // 2):
                block = gray[i:i+self.block_size, j:j+self.block_size]
                
                if block.shape != (self.block_size, self.block_size):
                    continue
                
                # DCT -> IDCT cycle
                dct = cv2.dct(block)
                
                # Zero out low-frequency components
                dct_low = dct.copy()
                dct_low[:2, :2] = 0
                
                # Reconstruct
                reconstructed = cv2.idct(dct_low)
                
                # Compute residual
                residual = np.abs(block - reconstructed)
                mean_residual = np.mean(residual)
                
                feature_map[i:i+self.block_size, j:j+self.block_size] = mean_residual
        
        return self.normalize(feature_map)