import math

def calibrate_features(features):
    """
    Applies multiplicative inversion to GLCM Energy, VBM, and MS as per spec.
    """
    calibrated = {}
    
    # Linear features
    if 'ms' in features:
        calibrated['ms'] = features['ms']
    if 'ced' in features:
        calibrated['ced'] = features['ced']
    
    # Quadratic features (apply sqrt)
    if 'glcm_e' in features:
        calibrated['glcm_e'] = math.sqrt(features['glcm_e'])
    if 'vbm' in features:
        calibrated['vbm'] = math.sqrt(features['vbm'])
    if 'glcm_c' in features:
        calibrated['glcm_c'] = math.sqrt(features['glcm_c'])

    return calibrated

def get_multipliers(original_features, edited_features, style_features):
    """
    Computes normalized multipliers (m1-m5) by dividing edited features by original features.
    """
    orig_cal = calibrate_features(original_features)
    edit_cal = calibrate_features(edited_features)
    style_cal = calibrate_features(style_features)
    
    multipliers = []
    # Sequence: GLCM_C -> CED -> GLCM_E -> VBM -> MS
    real_sequence = ['glcm_e', 'vbm', 'ms']
    style_sequence = ['glcm_c', 'ced']
    eps = 1e-10
    for key in real_sequence:
        m = edit_cal[key] / (orig_cal[key] + eps)
        multipliers.append(float(m))
    for key in style_sequence:
        m = edit_cal[key] / (style_cal[key] + eps)
        multipliers.append(float(m))
        
    return multipliers
