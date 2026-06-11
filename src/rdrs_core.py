from src.features import extract_all_features, extract_real_features, extract_style_features
from src.normalization import get_multipliers
from src.aggregation import get_rdrs_score, get_rdrs_separated_scores

def calculate_tier1_score(orig_path, edit_path, style_path):
    """
    Computes Tier 1: Structural Realism Score using the RDRS pentagon model.
    """
    orig_features = extract_real_features(orig_path)
    edit_features = extract_all_features(edit_path)
    style_features = extract_style_features(style_path)
    multipliers = get_multipliers(orig_features, edit_features, style_features)
    score = get_rdrs_score(multipliers)
    
    return score, multipliers
