import pytest
from src.real_semantic import calculate_tier3_score, MockVQABackend

def test_tier3_mock_score():
    # With MockVQABackend:
    # S_att = (9/10) * 100 = 90.0
    # S_rel = (4/5) * 100 = 80.0
    # Score = (90 + 80) / 2 = 85.0
    score = calculate_tier3_score("fake_orig.png", "fake_edit.png")
    assert score == 85.0

def test_tier3_custom_mock():
    class CustomMock(MockVQABackend):
        def ask(self, img, q):
            return {"visible_attributes": 1, "correct_attributes": 1, 
                    "visible_relationships": 1, "realistic_relationships": 1}
    
    score = calculate_tier3_score("fake_orig.png", "fake_edit.png", backend=CustomMock())
    assert score == 100.0
