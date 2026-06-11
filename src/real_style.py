import torch
import open_clip
from PIL import Image

class StyleClassifier:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        
        # Ensembled labels to catch diffusion artifacts
        self.pos_labels = ["a photorealistic portrait", "a raw photograph"]
        self.neg_labels = ["a digital painting", "a 3D CGI render", "an illustration"]
        
        self.pos_tokens = self.tokenizer(self.pos_labels)
        self.neg_tokens = self.tokenizer(self.neg_labels)

    @torch.no_grad()
    def get_style_logits(self, image_path):
        """
        Returns the raw cosine similarity (logits) for Photo vs Fake.
        """
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Encode text
        pos_feat = self.model.encode_text(self.pos_tokens)
        neg_feat = self.model.encode_text(self.neg_tokens)
        pos_feat /= pos_feat.norm(dim=-1, keepdim=True)
        neg_feat /= neg_feat.norm(dim=-1, keepdim=True)
        
        # Get raw similarities (ignore logit_scale and softmax)
        pos_sim = (image_features @ pos_feat.t()).mean().item()
        neg_sim = (image_features @ neg_feat.t()).mean().item()
        
        # The "Style Score" is how much more it looks like a photo than a painting
        style_margin = pos_sim - neg_sim
        return style_margin

def calculate_tier4_score(orig_path, edit_path):
    classifier = StyleClassifier()
    
    orig_margin = classifier.get_style_logits(orig_path)
    edit_margin = classifier.get_style_logits(edit_path)
    
    # Edge Case: If the original photo somehow scores as an illustration (margin <= 0),
    # we can't penalize the edit for the original's bad baseline.
    if orig_margin <= 0:
        return 100.0 
        
    # 2. Perfect Retention: The edit maintained or improved the photorealism.
    if edit_margin >= orig_margin:
        return 100.0
        
    # Add a smoothing factor to prevent tiny denominators from causing massive drops.
    # 0.10 represents a "base confidence" in the model.
    smoothing = 0.10 
    
    # Smoothed Ratio
    score = ((edit_margin + smoothing) / (orig_margin + smoothing)) * 100.0
    return score