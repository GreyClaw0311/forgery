"""Utility functions for tamper detection."""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict


def load_image(path: str) -> np.ndarray:
    """Load image from path."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def save_image(image: np.ndarray, path: str) -> bool:
    """Save image to path."""
    return cv2.imwrite(path, image)


def load_mask(path: str) -> np.ndarray:
    """Load ground truth mask."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")
    # Binarize
    return (mask > 127).astype(np.uint8)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Compute precision and recall."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def threshold_feature_map(feature_map: np.ndarray, method: str = 'otsu') -> np.ndarray:
    """Threshold feature map to binary mask."""
    if method == 'otsu':
        _, binary = cv2.threshold(feature_map.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            feature_map.astype(np.uint8), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    else:
        # Fixed threshold at mean + std
        thresh = np.mean(feature_map) + np.std(feature_map)
        binary = (feature_map > thresh).astype(np.uint8) * 255
    
    return binary


def visualize_result(image: np.ndarray, feature_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Create visualization overlay."""
    # Resize feature map to match image
    if feature_map.shape[:2] != image.shape[:2]:
        feature_map = cv2.resize(feature_map, (image.shape[1], image.shape[0]))
    
    # Normalize feature map
    feature_norm = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(feature_norm, cv2.COLORMAP_JET)
    
    # Blend with original
    result = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    # Add ground truth mask overlay if provided
    if mask is not None:
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 2] = mask * 128  # Red channel
        result = cv2.addWeighted(result, 0.8, mask_colored, 0.2, 0)
    
    return result


def get_dataset_files(data_dir: str) -> List[Dict[str, str]]:
    """Get list of image-mask pairs from dataset directory."""
    data_path = Path(data_dir)
    files = []
    
    # Check for images and masks directories
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    
    if images_dir.exists() and masks_dir.exists():
        for img_file in sorted(images_dir.glob("*")):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Try to find corresponding mask
                mask_file = masks_dir / f"{img_file.stem}.png"
                if not mask_file.exists():
                    mask_file = masks_dir / f"{img_file.stem}_mask.png"
                if not mask_file.exists():
                    mask_file = masks_dir / f"{img_file.stem}.jpg"
                
                files.append({
                    'image': str(img_file),
                    'mask': str(mask_file) if mask_file.exists() else None,
                    'name': img_file.stem
                })
    else:
        # Flat structure
        for img_file in sorted(data_path.glob("*")):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                files.append({
                    'image': str(img_file),
                    'mask': None,
                    'name': img_file.stem
                })
    
    return files


__all__ = [
    'load_image',
    'save_image',
    'load_mask',
    'compute_iou',
    'compute_precision_recall',
    'compute_f1',
    'threshold_feature_map',
    'visualize_result',
    'get_dataset_files',
]