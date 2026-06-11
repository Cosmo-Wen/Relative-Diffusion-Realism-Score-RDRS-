import math

def get_rdrs_score(multipliers):
    """
    Computes the final RDRS score as a percentage relative to the baseline area (2.377).
    """
    # MAE
    penalties = [abs(1.0 - r) for r in multipliers]
    
    # Average the penalties
    avg_penalty = sum(penalties) / 5
    
    # Convert to a 0-100% score
    score = max(0.0, 100.0 - (avg_penalty * 100))

    return float(score)

def get_rdrs_separated_scores(multipliers):
    """
    Computes the two different RDRS scores: The quality score (MS, GLCM_E, VBM)
    and the style shift index (GLCM_C, CED). 
    """
    ced_r, glcm_c_r, glcm_e_r, vbm_r, ms_r = multipliers
    quality_penalties = [
        abs(1.0 - vbm_r),
        abs(1.0 - ms_r),
        abs(1.0 - glcm_e_r)
    ]
    avg_quality_penalty = sum(quality_penalties) / 3
    
    # Convert penalty to a score out of 100. 
    # (e.g., a 0.2 penalty means 80% retention)
    quality_score = max(0.0, 100.0 - (avg_quality_penalty * 100))

    # 2. STYLE SHIFT INDEX (CED, GLCM_C)
    # We EXPECT these to change. This number just tells us "how much" it changed.
    # A shift of 0.0 means the hair texture didn't change at all.
    style_shifts = [
        abs(1.0 - ced_r),
        abs(1.0 - glcm_c_r)
    ]
    style_shift_index = sum(style_shifts) / 2
    print(f"Quality Penalties: {quality_penalties}, Avg Penalty: {avg_quality_penalty:.4f}, Quality Score: {quality_score:.2f}%, Style Shifts: {style_shifts}, Style Shift Index: {style_shift_index:.4f}")
    return (quality_score, style_shift_index)
