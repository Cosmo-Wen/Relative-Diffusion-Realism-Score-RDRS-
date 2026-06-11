import numpy as np
import pytest
from src.features import (
    get_glcm_features,
    get_canny_edge_density,
    get_variance_blur_measure,
    get_mean_spectrum
)

def test_canny_black_image():
    # Pure black image should have 0 edge density
    black_img = np.zeros((100, 100), dtype=np.uint8)
    density = get_canny_edge_density(black_img)
    assert density == 0.0

def test_glcm_constant_image():
    # Constant image has 0 contrast and maximum energy
    const_img = np.ones((100, 100), dtype=np.uint8) * 128
    contrast, energy = get_glcm_features(const_img)
    assert contrast == 0.0
    assert energy == 1.0

def test_vbm_constant_image():
    # Constant image has 0 variance in Laplacian
    const_img = np.ones((100, 100), dtype=np.uint8) * 128
    vbm = get_variance_blur_measure(const_img)
    assert vbm == 0.0

def test_ms_constant_image():
    # Constant image has a single peak at DC, but mean spectrum is low compared to noise
    const_img = np.ones((100, 100), dtype=np.uint8) * 128
    ms = get_mean_spectrum(const_img)
    # The DC component is large, but other components are 0
    # Mean of magnitude spectrum of constant is roughly the mean value * total pixels / total pixels = mean value
    assert ms > 0

def test_feature_types():
    # Ensure all return floats
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    assert isinstance(get_canny_edge_density(img), float)
    assert isinstance(get_variance_blur_measure(img), float)
    assert isinstance(get_mean_spectrum(img), float)
    c, e = get_glcm_features(img)
    assert isinstance(c, float)
    assert isinstance(e, float)
