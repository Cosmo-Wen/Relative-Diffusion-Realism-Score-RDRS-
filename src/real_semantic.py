from abc import ABC, abstractmethod

class VQABackend(ABC):
    @abstractmethod
    def ask(self, image_path, question):
        pass

class MockVQABackend(VQABackend):
    """
    Simulates a Cloud API response for semantic checks.
    """
    def ask(self, image_path, question):
        # Mocking logic: In a real scenario, this would call GPT-4o or Gemini.
        # For Tier 3, we simulate a schema-based response.
        if "attributes" in question.lower():
            return {
                "visible_attributes": 10,
                "correct_attributes": 9
            }
        if "relationships" in question.lower():
            return {
                "visible_relationships": 5,
                "realistic_relationships": 4
            }
        return {}

def calculate_tier3_score(orig_path, edit_path, backend=None):
    """
    Computes Tier 3: Semantic & Relational Realism.
    Score = (S_att + S_rel) / 2
    """
    if backend is None:
        backend = MockVQABackend()
    
    # 1. Attribute Check
    att_res = backend.ask(edit_path, "Check for correctly depicted visible attributes.")
    s_att = (att_res['correct_attributes'] / att_res['visible_attributes']) * 100.0 if att_res.get('visible_attributes', 0) > 0 else 0.0
    
    # 2. Relationship Check
    rel_res = backend.ask(edit_path, "Check for realism and logical relationships between objects.")
    s_rel = (rel_res['realistic_relationships'] / rel_res['visible_relationships']) * 100.0 if rel_res.get('visible_relationships', 0) > 0 else 0.0
    
    # Combined Score
    score = (s_att + s_rel) / 2.0
    return score
