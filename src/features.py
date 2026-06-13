import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def get_glcm_features(image_gray, mask=None, mask_target=255):
    """
    Computes GLCM Contrast and Energy.
    Improved: Averaged over 4 angles for rotational invariance, reduced levels for performance.
    Mask-Aware: Only accumulates co-occurrences where both pixels are within the mask zone.
    """
    if image_gray.dtype != np.uint8:
        image_gray = (image_gray * 255).astype(np.uint8)
    
    # Reducing levels to 64
    img_reduced = (image_gray // 4).astype(np.uint8)
    
    if mask is not None:
        # Ensure mask matches image dimensions
        if mask.shape[:2] != img_reduced.shape[:2]:
            mask = cv2.resize(mask, (img_reduced.shape[1], img_reduced.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Trick: Set masked-out pixels to an intensity outside the [0, 63] range.
        # We use 64 levels for valid data, so level 64 is the 'masked' level.
        valid_mask = (mask == mask_target)
        img_reduced[~valid_mask] = 64
        levels = 65
    else:
        levels = 64
        
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_reduced, distances=[5], angles=angles, levels=levels, symmetric=True, normed=True)
    
    if mask is not None:
        # Slice GLCM to exclude the masked level (index 64)
        glcm = glcm[:64, :64, :, :]
        # Re-normalize if sum > 0
        total = np.sum(glcm)
        if total > 0:
            glcm /= total
            
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    
    return float(contrast), float(energy)

def get_canny_edge_density(image_gray, mask=None, mask_target=255):
    """
    Computes the proportion of pixels classified as structural edges using Canny.
    Improved: Adaptive thresholding based on median.
    Mask-Aware: Density is calculated strictly within the mask zone.
    """
    if image_gray.dtype != np.uint8:
        image_gray = (image_gray * 255).astype(np.uint8)
        
    v = np.median(image_gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(image_gray, lower, upper)
    
    if mask is not None:
        if mask.shape[:2] != image_gray.shape[:2]:
            mask = cv2.resize(mask, (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_mask = (mask == mask_target)
        # Filter edges by mask
        masked_edges = edges[valid_mask]
        area = np.sum(valid_mask)
        if area == 0:
            return 0.0
        density = np.sum(masked_edges > 0) / area
    else:
        density = np.sum(edges > 0) / edges.size
        
    return float(density)

def get_variance_blur_measure(image_gray, mask=None, mask_target=0):
    """
    Estimates image sharpness by computing the global variance of a Laplacian-filtered image.
    Mask-Aware: Image is zeroed outside target zone before Laplacian to ensure boundary 
    consistency, then variance is calculated strictly within the mask zone.
    """
    img_float = image_gray.astype(np.float32)
    
    if mask is not None:
        if mask.shape[:2] != image_gray.shape[:2]:
            mask = cv2.resize(mask, (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Zero out pixels outside the target zone
        img_float[mask != mask_target] = 0
        
    laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
    
    if mask is not None:
        valid_mask = (mask == mask_target)
        valid_laplacian = laplacian[valid_mask]
        if valid_laplacian.size == 0:
            return 0.0
        variance = np.var(valid_laplacian)
    else:
        variance = np.var(laplacian)
        
    return float(variance)

def get_mean_spectrum(image_gray, mask=None, mask_target=0):
    """
    Computes the average magnitude of the image's Fourier Transform.
    Improved: Excludes DC component to focus on high-frequency noise.
    Mask-Aware: Pixels outside target zone are zeroed before FFT.
    """
    img_float = image_gray.astype(np.float32)
    
    if mask is not None:
        if mask.shape[:2] != image_gray.shape[:2]:
            mask = cv2.resize(mask, (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Zero out pixels outside the target zone
        img_float[mask != mask_target] = 0
        
    f_transform = np.fft.rfft2(img_float)
    magnitude_spectrum = np.abs(f_transform)
    
    magnitude_spectrum[0, 0] = 0
    mean_spectrum = np.mean(magnitude_spectrum)
    
    return float(mean_spectrum)

def extract_all_features(image_path, mask=None):
    """
    Loads an image and extracts all 5 RDRS features.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Generic extraction doesn't know target, but for full extraction we assume global
    contrast, energy = get_glcm_features(gray, mask, mask_target=255)
    ced = get_canny_edge_density(gray, mask, mask_target=255)
    vbm = get_variance_blur_measure(gray, mask, mask_target=0)
    ms = get_mean_spectrum(gray, mask, mask_target=0)
    
    return {
        'glcm_c': contrast,
        'ced': ced,
        'glcm_e': energy,
        'vbm': vbm,
        'ms': ms
    }

def extract_real_features(image_path, mask=None):
    """
    Extracts features evaluated against the Original Image (Quality Axes).
    Target Baseline: Original Image (mask_target=0).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, energy = get_glcm_features(gray, mask, mask_target=0)
    vbm = get_variance_blur_measure(gray, mask, mask_target=0)
    ms = get_mean_spectrum(gray, mask, mask_target=0)
    
    return {
        'glcm_e': energy,
        'vbm': vbm,
        'ms': ms
    }

def extract_style_features(image_path, mask=None):
    """
    Extracts features evaluated against the Style Reference (Style Axes).
    Target Baseline: Style Reference (mask_target=255).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    contrast, _ = get_glcm_features(gray, mask, mask_target=255)
    ced = get_canny_edge_density(gray, mask, mask_target=255)
    
    return {
        'glcm_c': contrast,
        'ced': ced,
    }
