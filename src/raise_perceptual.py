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
        
        # FIX 1: Handle X * 1024 images without cropping or squashing.
        # We resize the longest edge to 512, keeping the aspect ratio intact.
        # The AdaptiveAvgPool2d inside ResNet will handle the rectangular shape.
        self.preprocess = transforms.Compose([
            transforms.Resize(256), # Scales shortest edge to 256, keeps aspect ratio
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def get_embedding(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0)
        
        embedding = self.model(img_tensor)
        embedding = embedding.view(1, -1) # Flatten to (1, 512)
        return embedding

def calculate_tier2_score(orig_path, edit_path):
    """
    Computes Tier 2: Perceptual Realism Score using ResNet-18 feature similarity.
    """
    extractor = PerceptualExtractor()
    
    orig_emb = extractor.get_embedding(orig_path)
    edit_emb = extractor.get_embedding(edit_path)
    
    # Compute Cosine Similarity
    similarity = F.cosine_similarity(orig_emb, edit_emb).item()
    
    mean = torch.tensor(0.875)
    std = torch.tensor(0.06)
    dist = torch.distributions.Normal(mean, std)
    similarity_tensor = torch.tensor(similarity)
    prob = torch.exp(-0.5 * ((similarity_tensor - mean) / std) ** 2).item()
    return prob * 100.0