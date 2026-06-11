import math

def calculate_pentagon_area(multipliers):
    """
    Computes the area of a pentagon formed by five radii (multipliers).
    Area = sum(0.5 * m_i * m_{i+1} * sin(72 degrees))
    """
    if len(multipliers) != 5:
        raise ValueError("Exactly 5 multipliers are required for pentagon area calculation.")
    
    angle_rad = math.radians(72)
    sin_72 = math.sin(angle_rad)
    
    area = 0.0
    for i in range(5):
        m_curr = multipliers[i]
        m_next = multipliers[(i + 1) % 5]
        area += 0.5 * m_curr * m_next * sin_72
        
    return area

def get_rdrs_score(multipliers):
    """
    Computes the final RDRS score as a percentage relative to the baseline area (2.377).
    """
    baseline_area = 2.3776412907378837  # 5 * 0.5 * 1 * 1 * sin(72 deg)
    edited_area = calculate_pentagon_area(multipliers)
    
    score = (edited_area / baseline_area) * 100.0
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

    return (quality_score, style_shift_index)
