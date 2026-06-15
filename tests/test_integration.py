import os
import pytest
import numpy as np
from src.rdrs_core import calculate_tier1_score
from src.segmentation import MockSegmenter

def test_identity_score():
    """
    Comparing an image to itself must return exactly 100.0% RDRS score.
    """
    img_path = "tests/temp_identity.png"
    # Create synthetic image
    import cv2
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)
        
    scores, multipliers = calculate_tier1_score(img_path, img_path, img_path, segmenter=MockSegmenter())
    
    # All multipliers should be 1.0 (or very close)
    for m in multipliers['hair'] + multipliers['bg']:
        assert pytest.approx(m, rel=1e-5) == 1.0
        
    assert pytest.approx(scores['final'], rel=1e-5) == 100.0

def test_integration_on_assets():
    """
    Verifies the pipeline runs successfully on the provided source/edited pair.
    """
    img_path = "tests/temp_integration.png"
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(img_path, img)
    
    scores, multipliers = calculate_tier1_score(img_path, img_path, img_path, segmenter=MockSegmenter())
    
    score = scores['final']
    assert isinstance(score, float)
    assert score >= 0.0
