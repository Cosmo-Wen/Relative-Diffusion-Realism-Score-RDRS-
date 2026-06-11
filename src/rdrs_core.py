from src.features import extract_all_features
from src.normalization import get_multipliers
from src.aggregation import get_rdrs_separated_scores

def calculate_tier1_score(orig_path, edit_path):
    """
    Computes Tier 1: Structural Realism Score using the RDRS pentagon model.
    """
    orig_features = extract_all_features(orig_path)
    edit_features = extract_all_features(edit_path)
    multipliers = get_multipliers(orig_features, edit_features)
    score = get_rdrs_separated_scores(multipliers)
    
    return score, multipliers
