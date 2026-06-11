import torch
import open_clip
from PIL import Image

class StyleClassifier:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        
        # Labels for zero-shot classification
        self.labels = ["a photo", "an illustration", "a painting"]
        self.text_tokens = self.tokenizer(self.labels)

    @torch.no_grad()
    def get_style_score(self, image_path):
        """
        Returns the probability that the image is a "photo" vs "illustration/painting".
        """
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(self.text_tokens)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        
        # Softmax to get probabilities
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        # Probability of the first label "a photo"
        photo_prob = float(probs[0]) * 100.0
        return photo_prob

def calculate_tier4_score(orig_path, edit_path):
    """
    Computes Tier 4: Visual Style Fidelity using OpenCLIP zero-shot classification.
    """
    classifier = StyleClassifier()
    score = classifier.get_style_score(edit_path)
    return score
