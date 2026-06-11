import math

def calibrate_features(features):
    """
    Applies multiplicative inversion to GLCM Energy, VBM, and MS as per spec.
    """
    calibrated = {}
    
    # Linear features

    calibrated['ms'] = features['ms']
    calibrated['ced'] = features['ced']
    
    calibrated['glcm_e'] = math.sqrt(features['glcm_e'])
    calibrated['vbm'] = math.sqrt(features['vbm'])
    calibrated['glcm_c'] = math.sqrt(features['glcm_c'])

    
    return calibrated

def get_multipliers(original_features, edited_features):
    """
    Computes normalized multipliers (m1-m5) by dividing edited features by original features.
    """
    orig_cal = calibrate_features(original_features)
    edit_cal = calibrate_features(edited_features)
    
    multipliers = []
    # Sequence: GLCM_C -> CED -> GLCM_E -> VBM -> MS
    sequence = ['glcm_c', 'ced', 'glcm_e', 'vbm', 'ms']
    
    eps = 1e-10
    for key in sequence:
        m = edit_cal[key] / (orig_cal[key] + eps)
        multipliers.append(float(m))
        
    return multipliers
