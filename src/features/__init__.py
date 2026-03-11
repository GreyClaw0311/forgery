"""Feature detectors for image tamper detection.

This module provides 10 feature detection algorithms for detecting
various types of image manipulation.
"""

from .base import BaseFeatureDetector
from .ela import ELADetector, ELAAdvancedDetector
from .cfa import CFADetector, CFAInterpolationDetector
from .dct import DCTDetector, DCTResidualDetector
from .noise import NoiseDetector, NoiseVarianceDetector, PRNUNoiseDetector
from .blocking import BlockingArtifactDetector, BlockingGridDetector
from .lbp import LBPDetector, LBPConsistencyDetector
from .hog import HOGDetector, HOGLocalVarianceDetector, GradientInconsistencyDetector
from .sift import SIFTDetector, CopyMoveDetector
from .edge import EdgeDetector, EdgeConsistencyDetector, EdgeDensityDetector
from .color import ColorConsistencyDetector, IlluminationDetector, ChromaticAberrationDetector


# All available detectors
ALL_DETECTORS = [
    # Core detectors (10 primary)
    ELADetector,          # 1. Error Level Analysis
    CFADetector,          # 2. Color Filter Array
    DCTDetector,          # 3. DCT coefficient analysis
    NoiseDetector,        # 4. Noise inconsistency
    BlockingArtifactDetector,  # 5. JPEG blocking artifacts
    LBPDetector,          # 6. Local Binary Pattern
    HOGDetector,          # 7. Histogram of Oriented Gradients
    SIFTDetector,         # 8. SIFT keypoint analysis
    EdgeDetector,         # 9. Edge detection
    ColorConsistencyDetector,  # 10. Color consistency
    
    # Advanced variants
    ELAAdvancedDetector,
    CFAInterpolationDetector,
    DCTResidualDetector,
    NoiseVarianceDetector,
    PRNUNoiseDetector,
    BlockingGridDetector,
    LBPConsistencyDetector,
    HOGLocalVarianceDetector,
    GradientInconsistencyDetector,
    CopyMoveDetector,
    EdgeConsistencyDetector,
    EdgeDensityDetector,
    IlluminationDetector,
    ChromaticAberrationDetector,
]


# Default detector set for quick analysis
DEFAULT_DETECTORS = [
    ELADetector,
    CFADetector,
    DCTDetector,
    NoiseDetector,
    BlockingArtifactDetector,
    LBPDetector,
    HOGDetector,
    SIFTDetector,
    EdgeDetector,
    ColorConsistencyDetector,
]


def get_detector_by_name(name: str):
    """Get detector class by name."""
    for detector in ALL_DETECTORS:
        if detector.name == name:
            return detector
    return None


__all__ = [
    'BaseFeatureDetector',
    'ALL_DETECTORS',
    'DEFAULT_DETECTORS',
    'get_detector_by_name',
    # Core detectors
    'ELADetector',
    'CFADetector',
    'DCTDetector',
    'NoiseDetector',
    'BlockingArtifactDetector',
    'LBPDetector',
    'HOGDetector',
    'SIFTDetector',
    'EdgeDetector',
    'ColorConsistencyDetector',
    # Advanced detectors
    'ELAAdvancedDetector',
    'CFAInterpolationDetector',
    'DCTResidualDetector',
    'NoiseVarianceDetector',
    'PRNUNoiseDetector',
    'BlockingGridDetector',
    'LBPConsistencyDetector',
    'HOGLocalVarianceDetector',
    'GradientInconsistencyDetector',
    'CopyMoveDetector',
    'EdgeConsistencyDetector',
    'EdgeDensityDetector',
    'IlluminationDetector',
    'ChromaticAberrationDetector',
]