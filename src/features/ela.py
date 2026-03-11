"""Error Level Analysis (ELA) for detecting image tampering.

ELA identifies areas of an image that have been saved at different quality levels,
which can indicate manipulation.
"""
import numpy as np
import cv2
from typing import Tuple
from .base import BaseFeatureDetector


class ELADetector(BaseFeatureDetector):
    """Error Level Analysis detector."""
    
    name = "ELA"
    description = "Error Level Analysis - detects compression inconsistencies"
    
    def __init__(self, quality: int = 90):
        """
        Initialize ELA detector.
        
        Args:
            quality: JPEG quality for re-compression (default 90)
        """
        self.quality = quality
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Perform ELA on the image.
        
        Process:
        1. Save image as JPEG at specified quality
        2. Load the re-saved image
        3. Compute absolute difference from original
        """
        original = image.astype(np.float32)
        
        # Encode and decode at specified quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        reloaded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        reloaded = reloaded.astype(np.float32)
        
        # Compute difference
        diff = np.abs(original - reloaded)
        
        # Combine channels
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)
        
        return self.normalize(diff)


class ELAAdvancedDetector(BaseFeatureDetector):
    """Advanced ELA with multi-scale analysis."""
    
    name = "ELA_Advanced"
    description = "Advanced ELA with multiple quality levels"
    
    def __init__(self, qualities: list = None):
        self.qualities = qualities or [75, 85, 95]
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Multi-quality ELA analysis."""
        original = image.astype(np.float32)
        combined_diff = np.zeros(original.shape[:2], dtype=np.float32)
        
        for quality in self.qualities:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            reloaded = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32)
            
            diff = np.abs(original - reloaded)
            if len(diff.shape) == 3:
                diff = np.mean(diff, axis=2)
            
            combined_diff += diff
        
        combined_diff /= len(self.qualities)
        return self.normalize(combined_diff)