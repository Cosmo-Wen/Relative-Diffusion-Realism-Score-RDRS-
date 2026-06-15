import numpy as np
import cv2
import torch
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

class TransformersHairSegmenter(BaseSegmenter):
    """
    Uses Segformer (mattmdjaga/segformer_b2_clothes) to isolate the hair zone.
    This is an actual semantic model that provides robust hair-to-hair comparison.
    """
    def __init__(self):
        from transformers import pipeline
        self.device = 0 if torch.cuda.is_available() else -1
        # Label 2 is typically "Hair" for this model
        self.pipe = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes", device=self.device)

    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(image_rgb)
        
        # Segment
        results = self.pipe(pil_img)
        
        # Initialize empty mask
        h, w = image_bgr.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Find the "Hair" segment
        for res in results:
            if res['label'].lower() == 'hair':
                mask = np.array(res['mask']).astype(np.uint8)
                # Segformer mask might be resized, ensure it matches input
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                final_mask[mask > 0] = 255
                break
                
        return final_mask

class MockSegmenter(BaseSegmenter):
    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
        return mask

class HeuristicHairSegmenter(BaseSegmenter):
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            mask[0:int(h*0.4), :] = 255
            return mask
        (x, y, fw, fh) = sorted(faces, key=lambda f: f[2]*f[3])[-1]
        h_start, h_end = 0, min(h, y + int(fh * 0.4))
        w_start, w_end = max(0, x - int(fw * 0.25)), min(w, x + fw + int(fw * 0.25))
        mask[h_start:h_end, w_start:w_end] = 255
        return mask

class DifferenceSegmenter(BaseSegmenter):
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
        return mask

class DummySegmenter(BaseSegmenter):
    def segment(self, image_bgr: np.ndarray, reference_bgr: np.ndarray = None) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:h // 2, :] = 255
        return mask
