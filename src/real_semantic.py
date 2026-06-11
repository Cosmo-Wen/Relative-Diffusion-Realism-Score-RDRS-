import json
import base64
import re
from abc import ABC, abstractmethod

import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
PROJECT_ID = "modiface-rnd"
LOCATION   = "global"
MODEL_NAME = "gemini-3.5-flash"

vertexai.init(project=PROJECT_ID, location=LOCATION)

class VQABackend(ABC):
    @abstractmethod
    def ask(self, image_path, question):
        pass

class GeminiVQABackend(VQABackend):
    """
    Calls Gemini 3.5 Flash via Vertex AI for semantic checks.
    Expects the model to return a JSON object as plain text.
    """

    def __init__(self, model_name=MODEL_NAME):
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
        # 1. Direct parse — happy path
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown code fences (json ...)         
        fenced = re.sub(r"^(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        fenced = re.sub(r"\s*```$", "", fenced).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

        # 3. Extract first {...} block found anywhere in the text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        # 4. Give up — raise with the raw response for debugging
        raise ValueError(
            f"GeminiVQABackend: model returned non-JSON response.\n"
            f"--- Raw response ---\n{raw}\n--------------------"
        )

def calculate_tier3_score(orig_path, edit_path, backend=None):
    """
    Computes Tier 3: Semantic & Relational Realism.
    Score = (S_att + S_rel) / 2
    """
    if backend is None:
        backend = GeminiVQABackend()

    # 1. Attribute Check
    att_res = backend.ask(edit_path, "Check for correctly depicted visible attributes.")
    print(f"Attribute Check Result: {att_res}")
    # s_att = (att_res['correct_attributes'] / att_res['visible_attributes']) * 100.0 if att_res.get('visible_attributes', 0) > 0 else 0.0

    # 2. Relationship Check
    rel_res = backend.ask(edit_path, "Check for realism and logical relationships between objects.")
    print(f"Relationship Check Result: {rel_res}")
    # s_rel = (rel_res['realistic_relationships'] / rel_res['visible_relationships']) * 100.0 if rel_res.get('visible_relationships', 0) > 0 else 0.0

    # Combined Score
    # score = (s_att + s_rel) / 2.0
    return 0