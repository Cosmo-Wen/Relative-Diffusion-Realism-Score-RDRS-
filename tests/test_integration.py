import os
import pytest
from src.features import extract_all_features
from src.normalization import get_multipliers
from src.aggregation import get_rdrs_score

def test_identity_score():
    """
    Comparing an image to itself must return exactly 100.0% RDRS score.
    """
    # Use the sample image as both original and edited
    img_path = "data/original/source.png"
    
    if not os.path.exists(img_path):
        pytest.skip("Source image not found for identity test.")
        
    features = extract_all_features(img_path)
    multipliers = get_multipliers(features, features)
    
    # All multipliers should be 1.0
    for m in multipliers:
        assert pytest.approx(m, rel=1e-5) == 1.0
        
    score = get_rdrs_score(multipliers)
    assert pytest.approx(score, rel=1e-5) == 100.0

def test_integration_on_assets():
    """
    Verifies the pipeline runs successfully on the provided source/edited pair.
    """
    orig_path = "data/original/source.png"
    edit_path = "data/edited/edited.png"
    
    if not (os.path.exists(orig_path) and os.path.exists(edit_path)):
        pytest.skip("Assets not found for integration test.")
        
    orig_feat = extract_all_features(orig_path)
    edit_feat = extract_all_features(edit_path)
    multipliers = get_multipliers(orig_feat, edit_feat)
    score = get_rdrs_score(multipliers)
    
    assert isinstance(score, float)
    assert score >= 0.0
