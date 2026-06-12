import numpy as np
from abc import ABC, abstractmethod

class BaseSegmenter(ABC):
    """
    Abstract base class for image segmentation backends.
    """
    @abstractmethod
    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Ingests a standard 3D NumPy BGR array and returns a single-channel 
        binary mask of identical spatial dimensions.
        Value 255 denotes the semantic zone (e.g., hair), 0 denotes the background.
        """
        pass

class MockSegmenter(BaseSegmenter):
    """
    Simulates a segmentation mask for testing purposes.
    Returns a central rectangular mask.
    """
    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Create a central rectangle (Hair Zone)
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        mask[y1:y2, x1:x2] = 255
        return mask

class DummySegmenter(BaseSegmenter):
    """
    Returns an alternative static mask to verify backend swapping.
    """
    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Create a different zone (e.g., top half)
        mask[0:h // 2, :] = 255
        return mask
