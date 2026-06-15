import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def get_glcm_features(image_gray, mask=None):
    if image_gray.dtype != np.uint8:
        image_gray = (image_gray * 255).astype(np.uint8)
    
    # Reducing levels to 64
    img_reduced = (image_gray // 4).astype(np.uint8)
    
    if mask is not None:
        if mask.shape[:2] != img_reduced.shape[:2]:
            mask = cv2.resize(mask, (img_reduced.shape[1], img_reduced.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        valid_mask = (mask == 255)
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

def get_canny_edge_density(masked_img, mask=None):
    if masked_img.dtype != np.uint8:
        masked_img = (masked_img * 255).astype(np.uint8)
        
    if mask is not None:
        if mask.shape[:2] != masked_img.shape[:2]:
            mask = cv2.resize(mask, (masked_img.shape[1], masked_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_mask = (mask == 255)
        v = np.median(masked_img[valid_mask]) if np.any(valid_mask) else 0
    else:
        v = np.median(masked_img)

    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(masked_img, lower, upper)
    
    if mask is not None:
        valid_mask = (mask == 255)
        masked_edges = edges[valid_mask]
        area = np.sum(valid_mask)
        if area == 0:
            return 0.0
        density = np.sum(masked_edges > 0) / area
    else:
        density = np.sum(edges > 0) / edges.size
        
    return float(density)

def get_variance_blur_measure(masked_img, mask=None):
    img_float = masked_img.astype(np.float32)
    laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
    
    if mask is not None:
        if mask.shape[:2] != img_float.shape[:2]:
            mask = cv2.resize(mask, (img_float.shape[1], img_float.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_mask = (mask == 255)
        valid_laplacian = laplacian[valid_mask]
        if valid_laplacian.size == 0:
            return 0.0
        variance = np.var(valid_laplacian)
    else:
        variance = np.var(laplacian)
        
    return float(variance)

def get_mean_spectrum(masked_img, mask=None):
    img_float = masked_img.astype(np.float32)
    f_transform = np.fft.rfft2(img_float)
    magnitude_spectrum = np.abs(f_transform)
    magnitude_spectrum[0, 0] = 0
    mean_spectrum = np.mean(magnitude_spectrum)
    return float(mean_spectrum)

def get_masked_metrics(image_gray, mask=None):
    """
    Extracts all 5 structural metrics for a specific zone defined by the mask.
    The mask should have 255 for the Region of Interest and 0 elsewhere.
    """
    if mask is not None:
        if mask.shape[:2] != image_gray.shape[:2]:
            mask = cv2.resize(mask, (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked_img = cv2.bitwise_and(image_gray, image_gray, mask=mask)
    else:
        masked_img = image_gray

    # GLCM uses the original image + out-of-bounds trick to handle mask correctly
    contrast, energy = get_glcm_features(image_gray, mask)
    ced = get_canny_edge_density(masked_img, mask)
    vbm = get_variance_blur_measure(masked_img, mask)
    ms = get_mean_spectrum(masked_img, mask)
    
    return {
        'glcm_c': contrast,
        'ced': ced,
        'glcm_e': energy,
        'vbm': vbm,
        'ms': ms
    }
