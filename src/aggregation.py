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
