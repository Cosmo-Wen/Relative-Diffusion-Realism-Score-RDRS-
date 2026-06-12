import numpy as np
import cv2
from abc import ABC, abstractmethod

class BaseSegmenter(ABC):
    """
    Abstract base class for image segmentation backends.
    """
    @abstractmethod
    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
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
    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Create a central rectangle (Hair Zone)
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        mask[y1:y2, x1:x2] = 255
        return mask

class DifferenceSegmenter(BaseSegmenter):
    """
    Dynamically generates a mask by comparing the edited image to the original.
    This is highly robust for isolating edited zones (e.g. hair) from preservation zones.
    """
    def __init__(self, threshold=20):
        self.threshold = threshold

    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
        if reference_bgr is None:
            # Fallback to a simple heuristic if no reference is provided
            # (e.g. assume the top-middle is the edit zone)
            h, w = image_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[0:int(h*0.7), int(w*0.1):int(w*0.9)] = 255
            return mask
        
        # Ensure dimensions match for subtraction
        if image_bgr.shape != reference_bgr.shape:
            reference_bgr = cv2.resize(reference_bgr, (image_bgr.shape[1], image_bgr.shape[0]))
            
        # Compute absolute difference
        diff = cv2.absdiff(image_bgr, reference_bgr)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold to create binary mask
        _, mask = cv2.threshold(gray_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup to close small gaps in hair/texture
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        
        return mask

class DummySegmenter(BaseSegmenter):
    """
    Returns an alternative static mask to verify backend swapping.
    """
    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:h // 2, :] = 255
        return mask
