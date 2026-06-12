import os
import pytest
import torch
from src.raise_perceptual import PerceptualExtractor, calculate_tier2_score

def test_perceptual_extractor_shape():
    extractor = PerceptualExtractor()
    img_path = "data/original/source.png"
    
    if not os.path.exists(img_path):
        pytest.skip("Source image not found.")
        
    embedding = extractor.get_embedding(img_path)
    # ResNet-18 average pool output is 512
    assert embedding.shape == (1, 512)
    assert isinstance(embedding, torch.Tensor)

def test_perceptual_identity_score():
    img_path = "data/original/source.png"
    
    if not os.path.exists(img_path):
        pytest.skip("Source image not found.")
        
    score = calculate_tier2_score(img_path, img_path)
    # Identity similarity should be exactly 100%
    assert pytest.approx(score, rel=1e-5) == 100.0
