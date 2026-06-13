import numpy as np
import cv2
import os
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
        Value 255 denotes the semantic zone (e.g., hair/subject), 0 denotes the background.
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

class HeuristicHairSegmenter(BaseSegmenter):
    """
    Uses Face Detection to heuristically estimate the hair zone.
    Extremely robust to pixel shifts and requires no heavy AI models.
    """
    def __init__(self):
        # Load the pre-trained Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # Fallback if no face detected: use top 40% of the image
            mask[0:int(h*0.4), :] = 255
            return mask
            
        # Use the largest detected face
        (x, y, fw, fh) = sorted(faces, key=lambda f: f[2]*f[3])[-1]
        
        # Heuristic: Hair is usually from top of image down to eye-level, 
        # and slightly wider than the face.
        h_start = 0
        h_end = y + int(fh * 0.4) # Down to forehead/eyes
        w_start = max(0, x - int(fw * 0.25))
        w_end = min(w, x + fw + int(fw * 0.25))
        
        mask[h_start:h_end, w_start:w_end] = 255
        return mask

class DifferenceSegmenter(BaseSegmenter):
    """
    Dynamically generates a mask by comparing the edited image to the original.
    """
    def __init__(self, threshold=20):
        self.threshold = threshold

    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
        if reference_bgr is None:
            h, w = image_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[0:int(h*0.7), int(w*0.1):int(w*0.9)] = 255
            return mask
        
        if image_bgr.shape != reference_bgr.shape:
            reference_bgr = cv2.resize(reference_bgr, (image_bgr.shape[1], image_bgr.shape[0]))
            
        diff = cv2.absdiff(image_bgr, reference_bgr)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
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
