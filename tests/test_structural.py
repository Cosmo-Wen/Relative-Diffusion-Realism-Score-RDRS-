import numpy as np
import pytest
import cv2
from src.rdrs_core import calculate_tier1_score
from src.segmentation import MockSegmenter, DummySegmenter

def test_inverse_mask_identity():
    """
    Inverse Mask Identity Test:
    If an edited image matches the original perfectly outside the mask zone,
    the preservation-related multipliers (GLCM_E, VBM, MS) must return exactly 1.0,
    leading to a 100% score on those components.
    """
    # Create synthetic images
    h, w = 100, 100
    original = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    edited = original.copy()
    
    # Apply change strictly INSIDE the mock mask (central rectangle)
    segmenter = MockSegmenter()
    mask = segmenter.segment(original)
    edited[mask == 255] = 255 # Corrupt hair zone
    
    # Save to temp files
    orig_path = "tests/temp_orig.png"
    edit_path = "tests/temp_edit.png"
    cv2.imwrite(orig_path, original)
    cv2.imwrite(edit_path, edited)
    
    # Calculate score (style path doesn't matter for this test of preservation)
    score, multipliers = calculate_tier1_score(orig_path, edit_path, orig_path, segmenter=segmenter)
    
    # Multipliers are [GLCM_E, VBM, MS, GLCM_C, CED] - wait, let's check order in normalization.py
    # Sequence in normalization.py: ['glcm_e', 'vbm', 'ms', 'glcm_c', 'ced']
    m_e, m_vbm, m_ms, m_c, m_ced = multipliers
    
    # Preservation multipliers should be exactly 1.0 because image is identical outside mask
    assert pytest.approx(m_e, rel=1e-5) == 1.0
    assert pytest.approx(m_vbm, rel=1e-5) == 1.0
    assert pytest.approx(m_ms, rel=1e-5) == 1.0

def test_backend_interoperability():
    """
    Backend Interoperability Test:
    Swap the segmenter and ensure no signature breaks.
    """
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img_path = "tests/temp_interop.png"
    cv2.imwrite(img_path, img)
    
    # Test with MockSegmenter
    score1, _ = calculate_tier1_score(img_path, img_path, img_path, segmenter=MockSegmenter())
    
    # Test with DummySegmenter
    score2, _ = calculate_tier1_score(img_path, img_path, img_path, segmenter=DummySegmenter())
    
    assert isinstance(score1, float)
    assert isinstance(score2, float)
