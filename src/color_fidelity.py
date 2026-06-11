import cv2
import numpy as np

def get_color_fidelity_score(image1_path, image2_path):
    """
    Computes the 3D Color Histogram Intersection in HSV space.
    Returns a percentage score.
    """
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        raise ValueError("Could not read one or both images.")
    
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # 3D Histogram: H (180), S (256), V (256)
    # Reducing bins to save memory and improve robustness to slight shifts
    bins = [8, 8, 8]
    ranges = [0, 180, 0, 256, 0, 256]
    
    hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, bins, ranges)
    hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, bins, ranges)
    
    # Normalize histograms using L1 (sum of elements = 1)
    cv2.normalize(hist1, hist1, 1, 0, cv2.NORM_L1)
    cv2.normalize(hist2, hist2, 1, 0, cv2.NORM_L1)
    
    # Intersection
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    
    # To get a percentage, we can normalize by the total area of hist1 (self-intersection)
    self_inter = cv2.compareHist(hist1, hist1, cv2.HISTCMP_INTERSECT)
    
    if self_inter == 0:
        return 0.0
        
    score = (intersection / self_inter) * 100.0
    return float(score)
