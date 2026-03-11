"""Base class for feature detectors."""
from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Tuple, Optional


class BaseFeatureDetector(ABC):
    """Abstract base class for all feature detectors."""
    
    name: str = "base"
    description: str = "Base feature detector"
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect features in the image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Feature map (grayscale, same size as input)
        """
        pass
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def normalize(self, feature_map: np.ndarray) -> np.ndarray:
        """Normalize feature map to 0-255 range."""
        if feature_map.max() > feature_map.min():
            normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) * 255
            return normalized.astype(np.uint8)
        return np.zeros_like(feature_map, dtype=np.uint8)
    
    def resize_to_match(self, feature_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize feature map to match target shape."""
        if feature_map.shape[:2] != target_shape:
            return cv2.resize(feature_map, (target_shape[1], target_shape[0]))
        return feature_map