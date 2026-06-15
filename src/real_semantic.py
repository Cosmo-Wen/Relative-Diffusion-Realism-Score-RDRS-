import json
import base64
import re
from abc import ABC, abstractmethod

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    HAS_VERTEX = True
except ImportError:
    HAS_VERTEX = False

# --- Configuration ---
PROJECT_ID = "modiface-rnd"
LOCATION   = "global"
MODEL_NAME = "gemini-3.5-flash"

if HAS_VERTEX:
    vertexai.init(project=PROJECT_ID, location=LOCATION)

class VQABackend(ABC):
    @abstractmethod
    def ask(self, image_path, question):
        pass

class MockVQABackend(VQABackend):
    """
    Simulates a Cloud API response for semantic checks.
    """
    def ask(self, image_path, question):
        if "attributes" in question.lower():
            return {"visible_attributes": 10, "correct_attributes": 9}
        if "relationships" in question.lower():
            return {"visible_relationships": 5, "realistic_relationships": 4}
        return {}

class GeminiVQABackend(VQABackend):
    """
    Calls Gemini 3.5 Flash via Vertex AI for semantic checks.
    """
    def __init__(self, model_name=MODEL_NAME):
        if not HAS_VERTEX:
            raise ImportError("google-cloud-aiplatform is required for GeminiVQABackend")
        self.model = GenerativeModel(model_name)

    def ask(self, image_path, question):
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_part = Part.from_data(
            data=base64.b64encode(image_bytes).decode("utf-8"),
            mime_type="image/jpeg"
        )

        prompt = (
            f"{question}\n\n"
            "Respond ONLY with a valid JSON object. No explanation, no markdown."
        )

        response = self.model.generate_content([image_part, prompt])
        raw = response.text.strip()
        return self._parse_json_safe(raw)

    def _parse_json_safe(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        fenced = re.sub(r"^(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        fenced = re.sub(r"\s*```$", "", fenced).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"GeminiVQABackend: non-JSON response: {raw}")

def calculate_tier3_score(orig_path, edit_path, backend=None):
    """
    Computes Tier 3: Semantic & Relational Realism.
    Currently simulated/not working: Returns 0.0 to indicate status.
    """
    return 0.0
