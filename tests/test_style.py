import os
import pytest
from src.real_style import StyleClassifier, calculate_tier4_score

def test_style_classifier_output():
    classifier = StyleClassifier()
    img_path = "data/original/source.png"
    
    if not os.path.exists(img_path):
        pytest.skip("Source image not found.")
        
    score = classifier.get_style_score(img_path)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0

def test_tier4_integration():
    img_path = "data/original/source.png"
    if not os.path.exists(img_path):
        pytest.skip("Source image not found.")
        
    score = calculate_tier4_score(img_path, img_path)
    assert isinstance(score, float)
