import cv2
from src.features import extract_all_features, extract_real_features, extract_style_features
from src.normalization import get_multipliers
from src.aggregation import get_rdrs_score, get_rdrs_separated_scores

def calculate_tier1_score(orig_path, edit_path, style_path, segmenter=None):
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
        # The mask is derived from comparing edit to original to isolate spatial changes
        mask = segmenter.segment(edit_img, reference_bgr=orig_img)
        # The style mask is derived from the style image itself to find hair zones in the reference
        style_mask = segmenter.segment(style_img)
        
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
