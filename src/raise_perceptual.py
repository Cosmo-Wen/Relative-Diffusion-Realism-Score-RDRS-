import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

class PerceptualExtractor:
    def __init__(self):
        # Load pre-trained ResNet-18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final classification layer (fc)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # ImageNet normalization transforms
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_embedding(self, image_path):
        """
        Loads an image and extracts its 512-dimensional ResNet-18 embedding.
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0)
        
        embedding = self.model(img_tensor)
        # Flatten to (1, 512)
        embedding = embedding.view(1, -1)
        return embedding

def calculate_tier2_score(orig_path, edit_path):
    """
    Computes Tier 2: Perceptual Realism Score using ResNet-18 feature similarity.
    """
    extractor = PerceptualExtractor()
    
    orig_emb = extractor.get_embedding(orig_path)
    edit_emb = extractor.get_embedding(edit_path)
    
    # Compute Cosine Similarity
    # F.cosine_similarity returns a tensor of shape (1,) with value between -1 and 1
    similarity = F.cosine_similarity(orig_emb, edit_emb)
    
    # Convert to percentage (mapping -1,1 to 0,1 is possible, but similarity for 
    # similar images will be positive. We'll clip at 0 and return as percentage.)
    score = max(0.0, float(similarity.item())) * 100.0
    return score
