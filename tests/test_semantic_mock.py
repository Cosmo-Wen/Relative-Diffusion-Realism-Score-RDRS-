import pytest
from src.real_semantic import calculate_tier3_score, MockVQABackend

def test_tier3_mock_score():
    # Currently simulated to return 0.0
    score = calculate_tier3_score("fake_orig.png", "fake_edit.png")
    assert score == 0.0

def test_tier3_custom_mock():
    class CustomMock(MockVQABackend):
        def ask(self, img, q):
            return {"visible_attributes": 1, "correct_attributes": 1, 
                    "visible_relationships": 1, "realistic_relationships": 1}
    
    score = calculate_tier3_score("fake_orig.png", "fake_edit.png", backend=CustomMock())
    assert score == 0.0
