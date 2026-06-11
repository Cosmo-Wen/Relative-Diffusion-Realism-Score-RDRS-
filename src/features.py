import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def get_glcm_features(image_gray):
    """
    Computes GLCM Contrast and Energy.
    Improved: Averaged over 4 angles for rotational invariance, reduced levels for performance.
    """
    # Reducing levels to 64 for speed and robust texture analysis
    if image_gray.dtype != np.uint8:
        image_gray = (image_gray * 255).astype(np.uint8)
    
    img_reduced = (image_gray // 4).astype(np.uint8)
    
    # 4 angles: 0, 45, 90, 135 degrees
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_reduced, distances=[5], angles=angles, levels=64, symmetric=True, normed=True)
    
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    
    return float(contrast), float(energy)

def get_canny_edge_density(image_gray):
    """
    Computes the proportion of pixels classified as structural edges using Canny.
    Improved: Adaptive thresholding based on median.
    """
    if image_gray.dtype != np.uint8:
        image_gray = (image_gray * 255).astype(np.uint8)
        
    v = np.median(image_gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(image_gray, lower, upper)
    density = np.sum(edges > 0) / edges.size
    return float(density)

def get_variance_blur_measure(image_gray):
    """
    Estimates image sharpness by computing the global variance of a Laplacian-filtered image.
    """
    laplacian = cv2.Laplacian(image_gray.astype(np.float32), cv2.CV_32F)
    variance = np.var(laplacian)
    return float(variance)

def get_mean_spectrum(image_gray):
    """
    Computes the average magnitude of the image's Fourier Transform.
    Improved: Excludes DC component to focus on high-frequency noise.
    """
    # Performance: Use rfft2 for real-valued inputs
    f_transform = np.fft.rfft2(image_gray.astype(np.float32))
    magnitude_spectrum = np.abs(f_transform)
    
    # Exclude DC component (0,0) and very low frequencies
    # For simplicity, we just zero out the top-left corner or calculate mean excluding it
    magnitude_spectrum[0, 0] = 0
    mean_spectrum = np.mean(magnitude_spectrum)
    
    return float(mean_spectrum)

def extract_all_features(image_path):
    """
    Loads an image and extracts all 5 RDRS features.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    contrast, energy = get_glcm_features(gray)
    ced = get_canny_edge_density(gray)
    vbm = get_variance_blur_measure(gray)
    ms = get_mean_spectrum(gray)
    
    return {
        'glcm_c': contrast,
        'ced': ced,
        'glcm_e': energy,
        'vbm': vbm,
        'ms': ms
    }
