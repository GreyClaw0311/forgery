"""SIFT-based feature matching for copy-move and splice detection.

Detects duplicated or inconsistent keypoints.
"""
import numpy as np
import cv2
from typing import List, Tuple
from .base import BaseFeatureDetector


class SIFTDetector(BaseFeatureDetector):
    """SIFT keypoint density detector."""
    
    name = "SIFT"
    description = "SIFT keypoint density analysis"
    
    def __init__(self, nfeatures: int = 500):
        self.nfeatures = nfeatures
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Create feature map based on SIFT keypoint density.
        
        Areas with abnormal keypoint density may indicate tampering.
        """
        gray = self.preprocess(image)
        h, w = gray.shape
        
        # Create SIFT detector
        sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # Create density map
        density_map = np.zeros((h, w), dtype=np.float32)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < w and 0 <= y < h:
                # Add contribution based on keypoint response
                sigma = kp.size / 2
                for dy in range(-int(sigma), int(sigma) + 1):
                    for dx in range(-int(sigma), int(sigma) + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            weight = np.exp(-(dx**2 + dy**2) / (2 * sigma**2 + 1e-8))
                            density_map[ny, nx] += weight * kp.response
        
        return self.normalize(density_map)


class CopyMoveDetector(BaseFeatureDetector):
    """Detect copy-move forgery using SIFT matching."""
    
    name = "CopyMove"
    description = "SIFT-based copy-move detection"
    
    def __init__(self, ratio_thresh: float = 0.75, min_matches: int = 10):
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect potential copy-move regions."""
        gray = self.preprocess(image)
        h, w = gray.shape
        
        # Create feature map
        feature_map = np.zeros((h, w), dtype=np.float32)
        
        # Detect SIFT features
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < self.min_matches:
            return feature_map
        
        # Match features with themselves to find duplicates
        bf = cv2.BFMatcher(cv2.NORM_L2)
        
        try:
            matches = bf.knnMatch(descriptors, descriptors, k=3)
        except cv2.error:
            return feature_map
        
        # Find potential copy-move matches (similar features with different locations)
        for match_list in matches:
            if len(match_list) < 3:
                continue
            
            # Skip if best match is the same point
            m = match_list[0]
            if m.queryIdx == m.trainIdx:
                # Use second match
                if len(match_list) > 1:
                    m = match_list[1]
                else:
                    continue
            
            # Get keypoint locations
            kp1 = keypoints[m.queryIdx]
            kp2 = keypoints[m.trainIdx]
            
            # Check distance between keypoints (avoid self-matches)
            dist = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + (kp1.pt[1] - kp2.pt[1])**2)
            
            # Only consider distant matches as potential copy-move
            if dist > 30:  # Minimum distance threshold
                # Mark both regions
                x1, y1 = int(kp1.pt[0]), int(kp1.pt[1])
                x2, y2 = int(kp2.pt[0]), int(kp2.pt[1])
                
                size = max(3, int(kp1.size / 2))
                
                if 0 <= x1 < w and 0 <= y1 < h:
                    cv2.circle(feature_map, (x1, y1), size, 1.0, -1)
                if 0 <= x2 < w and 0 <= y2 < h:
                    cv2.circle(feature_map, (x2, y2), size, 1.0, -1)
        
        return self.normalize(feature_map)