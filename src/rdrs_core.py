import cv2
import os
import numpy as np
from src.features import get_masked_metrics
from src.normalization import get_zone_multipliers
from src.aggregation import get_rdrs_score

def save_mask_overlay(image_bgr, mask, name, output_dir="debug_masks"):
    """
    Saves a visualization of the mask overlaid on the image.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    overlay = image_bgr.copy()
    if mask is not None:
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay[mask == 255] = [0, 255, 0]
        
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, overlay)
    cv2.putText(overlay, f"ZONE: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(out_path, overlay)

def calculate_tier1_score(orig_path, edit_path, style_path, segmenter=None, save_masks=False):
    """
    Computes Tier 1: Structural Realism Score using the 5x2 Symmetric Model.
    """
    # Load images
    edit_img = cv2.imread(edit_path)
    orig_img = cv2.imread(orig_path)
    style_img = cv2.imread(style_path)
    
    if edit_img is None: raise ValueError(f"Could not read image at {edit_path}")
    if orig_img is None: raise ValueError(f"Could not read image at {orig_path}")
    if style_img is None: raise ValueError(f"Could not read style image at {style_path}")
        
    edit_gray = cv2.cvtColor(edit_img, cv2.COLOR_BGR2GRAY)
    orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    style_gray = cv2.cvtColor(style_img, cv2.COLOR_BGR2GRAY)

    mask_edit_hair = None
    mask_edit_bg = None
    mask_orig_bg = None
    mask_style_hair = None
    
    if segmenter is not None:
        # Generate independent hair masks
        mask_edit_hair = segmenter.segment(edit_img)
        mask_orig_hair = segmenter.segment(orig_img)
        mask_style_hair = segmenter.segment(style_img)
        
        # Invert for background
        mask_edit_bg = cv2.bitwise_not(mask_edit_hair)
        mask_orig_bg = cv2.bitwise_not(mask_orig_hair)
        
        if save_masks:
            stem = os.path.basename(edit_path).split('.')[0]
            save_mask_overlay(orig_img, mask_orig_bg, f"{stem}_orig_bg_zone")
            save_mask_overlay(edit_img, mask_edit_bg, f"{stem}_edit_bg_zone")
            save_mask_overlay(edit_img, mask_edit_hair, f"{stem}_edit_hair_zone")
            save_mask_overlay(style_img, mask_style_hair, f"{stem}_style_hair_zone")
        
    # --- ZONE 1: HAIR (Foreground) ---
    # Baseline: Style Image
    style_hair_features = get_masked_metrics(style_gray, mask=mask_style_hair)
    edit_hair_features = get_masked_metrics(edit_gray, mask=mask_edit_hair)
    hair_multipliers = get_zone_multipliers(style_hair_features, edit_hair_features)
    hair_score = get_rdrs_score(hair_multipliers)

    # --- ZONE 2: BACKGROUND (Inverse Mask) ---
    # Baseline: Original Image
    orig_bg_features = get_masked_metrics(orig_gray, mask=mask_orig_bg)
    edit_bg_features = get_masked_metrics(edit_gray, mask=mask_edit_bg)
    bg_multipliers = get_zone_multipliers(orig_bg_features, edit_bg_features)
    bg_score = get_rdrs_score(bg_multipliers)
    
    # --- FINAL SCORE ---
    final_score = (hair_score + bg_score) / 2.0
    
    # Package results
    scores = {'final': final_score, 'hair': hair_score, 'bg': bg_score}
    multipliers = {'hair': hair_multipliers, 'bg': bg_multipliers}
    
    return scores, multipliers
