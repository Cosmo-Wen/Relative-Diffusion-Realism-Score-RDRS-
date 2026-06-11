import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def get_glcm_features(image_gray):
    """
    Computes GLCM Contrast and Energy.
    """
    # Ensure image is uint8
    if image_gray.dtype != np.uint8:
        image_gray = (image_gray * 255).astype(np.uint8)
    
    glcm = graycomatrix(image_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return float(contrast), float(energy)

def get_canny_edge_density(image_gray):
    """
    Computes the proportion of pixels classified as structural edges using Canny.
    """
    if image_gray.dtype != np.uint8:
        image_gray = (image_gray * 255).astype(np.uint8)
        
    edges = cv2.Canny(image_gray, 100, 200)
    density = np.sum(edges > 0) / edges.size
    return float(density)

def get_variance_blur_measure(image_gray):
    """
    Estimates image sharpness by computing the global variance of a Laplacian-filtered image.
    """
    # Ensure float for Laplacian to avoid overflow/underflow
    laplacian = cv2.Laplacian(image_gray.astype(np.float32), cv2.CV_32F)
    variance = np.var(laplacian)
    return float(variance)

def get_mean_spectrum(image_gray):
    """
    Computes the average magnitude of the image's Fourier Transform.
    """
    f_transform = np.fft.fft2(image_gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
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
