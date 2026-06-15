import math

def calibrate_features(features):
    """
    Applies square root to quadratic features to linearize them.
    Linear features remain unchanged.
    """
    calibrated = {}
    
    # Linear features
    if 'ms' in features:
        calibrated['ms'] = features['ms']
    if 'ced' in features:
        calibrated['ced'] = features['ced']
    
    # Quadratic features (apply sqrt)
    if 'glcm_e' in features:
        calibrated['glcm_e'] = math.sqrt(max(features['glcm_e'], 0.0))
    if 'vbm' in features:
        calibrated['vbm'] = math.sqrt(max(features['vbm'], 0.0))
    if 'glcm_c' in features:
        calibrated['glcm_c'] = math.sqrt(max(features['glcm_c'], 0.0))

    return calibrated

def safe_ratio(edit_val, base_val, eps=1e-10):
    """
    Calculates the ratio while safely handling zero-division.
    If both the baseline and edit are essentially zero (e.g. solid background),
    the ratio is 1.0 (no degradation).
    If only the baseline is zero but edit is not, it caps the ratio.
    """
    if base_val < eps and edit_val < eps:
        return 1.0
    return edit_val / (base_val + eps)

def get_zone_multipliers(baseline_features, edit_features):
    """
    Computes normalized multipliers (m1-m5) for a given zone.
    Sequence: GLCM_C -> CED -> GLCM_E -> VBM -> MS
    """
    base_cal = calibrate_features(baseline_features)
    edit_cal = calibrate_features(edit_features)
    
    multipliers = []
    sequence = ['glcm_c', 'ced', 'glcm_e', 'vbm', 'ms']
    
    for key in sequence:
        m = safe_ratio(edit_cal.get(key, 0.0), base_cal.get(key, 0.0))
        multipliers.append(float(m))
        
    return multipliers
