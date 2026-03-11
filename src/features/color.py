"""Color consistency analysis for tamper detection.

Analyzes color patterns and inconsistencies across the image.
"""
import numpy as np
import cv2
from .base import BaseFeatureDetector


class ColorConsistencyDetector(BaseFeatureDetector):
    """Color consistency based tamper detection."""
    
    name = "COLOR"
    description = "Color consistency analysis"
    
    def __init__(self, block_size: int = 32):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze color consistency across image regions.
        
        Tampered regions may have inconsistent color statistics.
        """
        if len(image.shape) != 3:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(0, h - self.block_size + 1, self.block_size // 2):
            for j in range(0, w - self.block_size + 1, self.block_size // 2):
                # Extract blocks
                lab_block = lab[i:i+self.block_size, j:j+self.block_size]
                hsv_block = hsv[i:i+self.block_size, j:j+self.block_size]
                
                # Compute color variance
                # A and B channels in LAB are more informative for color consistency
                a_var = np.var(lab_block[:, :, 1].astype(np.float32))
                b_var = np.var(lab_block[:, :, 2].astype(np.float32))
                
                # HSV saturation variance
                s_var = np.var(hsv_block[:, :, 1].astype(np.float32))
                
                # Combined score
                score = np.sqrt(a_var + b_var + s_var)
                feature_map[i:i+self.block_size, j:j+self.block_size] = score
        
        return self.normalize(feature_map)


class IlluminationDetector(BaseFeatureDetector):
    """Detect illumination inconsistencies."""
    
    name = "ILLUMINATION"
    description = "Illumination consistency analysis"
    
    def __init__(self, block_size: int = 64):
        self.block_size = block_size
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Analyze illumination consistency."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        h, w = gray.shape
        
        # Estimate illumination using large-scale blur
        illumination = cv2.GaussianBlur(gray, (self.block_size // 2 * 2 + 1,) * 2, self.block_size // 4)
        
        # Compute reflectance
        reflectance = gray / (illumination + 1e-8)
        
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        # Analyze local illumination variance
        for i in range(0, h - self.block_size + 1, self.block_size // 2):
            for j in range(0, w - self.block_size + 1, self.block_size // 2):
                block = illumination[i:i+self.block_size, j:j+self.block_size]
                var = np.var(block)
                feature_map[i:i+self.block_size, j:j+self.block_size] = var
        
        return self.normalize(feature_map)


class ChromaticAberrationDetector(BaseFeatureDetector):
    """Detect chromatic aberration inconsistencies."""
    
    name = "CHROMATIC"
    description = "Chromatic aberration analysis"
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Analyze chromatic aberration consistency.
        
        Tampered regions may have inconsistent chromatic aberration.
        """
        if len(image.shape) != 3:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        # Split channels
        b, g, r = cv2.split(image)
        
        # Compute channel edges
        b_edges = cv2.Canny(b, 50, 150).astype(np.float32)
        g_edges = cv2.Canny(g, 50, 150).astype(np.float32)
        r_edges = cv2.Canny(r, 50, 150).astype(np.float32)
        
        # Compute channel edge differences
        bg_diff = np.abs(b_edges - g_edges)
        rg_diff = np.abs(r_edges - g_edges)
        
        # Combine differences
        feature_map = bg_diff + rg_diff
        
        return self.normalize(feature_map)