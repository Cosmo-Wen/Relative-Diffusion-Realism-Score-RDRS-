import cv2
import os
import numpy as np
from src.features import extract_all_features, extract_real_features, extract_style_features
from src.normalization import get_multipliers
from src.aggregation import get_rdrs_score, get_rdrs_separated_scores

def save_mask_overlay(image_bgr, mask, name, output_dir="debug_masks"):
    """
    Saves a visualization of the mask overlaid on the image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    overlay = image_bgr.copy()
    # Apply green tint to mask area
    if mask is not None:
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay[mask == 255] = [0, 255, 0]
        
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, overlay)
    
    # Label
    cv2.putText(overlay, f"ZONE: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(out_path, overlay)

def calculate_tier1_score(orig_path, edit_path, style_path, segmenter=None, save_masks=False):
    """
    Computes Tier 1: Structural Realism Score using Mask-Aware evaluation.
    """
    # Load images
    edit_img = cv2.imread(edit_path)
    orig_img = cv2.imread(orig_path)
    style_img = cv2.imread(style_path)
    
    if edit_img is None: raise ValueError(f"Could not read image at {edit_path}")
    if orig_img is None: raise ValueError(f"Could not read image at {orig_path}")
    if style_img is None: raise ValueError(f"Could not read style image at {style_path}")
        
    mask = None
    style_mask = None
    if segmenter is not None:
        # Generate independent masks
        mask = segmenter.segment(edit_img)
        style_mask = segmenter.segment(style_img)
        
        if save_masks:
            # For debugging, we use the same filename stem or just generic names
            stem = os.path.basename(edit_path).split('.')[0]
            save_mask_overlay(orig_img, mask, f"{stem}_orig_preservation_zone")
            save_mask_overlay(edit_img, mask, f"{stem}_edit_zones")
            save_mask_overlay(style_img, style_mask, f"{stem}_style_reference_zone")
        
    # Extract features using the mask boundaries
    # Quality Axes: Evaluated outside the mask (mask_target=0) against original baseline
    orig_features = extract_real_features(orig_path, mask=mask)
    edit_real_features = extract_real_features(edit_path, mask=mask)
    
    # Style Axes: Evaluated inside the mask (mask_target=255) against style baseline
    style_features = extract_style_features(style_path, mask=style_mask)
    edit_style_features = extract_style_features(edit_path, mask=mask)
    
    # Merge edited features for normalization
    edit_features = {**edit_real_features, **edit_style_features}
    
    multipliers = get_multipliers(orig_features, edit_features, style_features)
    score = get_rdrs_score(multipliers)
    
    return score, multipliers
