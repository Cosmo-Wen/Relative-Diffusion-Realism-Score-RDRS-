import math

def get_rdrs_score(multipliers):
    """
    Computes the final RDRS score using a symmetric logarithmic penalty.
    Strict but fair: Doubling a value and halving a value yield the same penalty.
    """
    # log10(1) = 0 (perfect)
    # log10(2) = 0.301, log10(0.5) = -0.301
    # We use log10 and cap it at 1.0 (a 10x difference) for strict but robust scoring.
    penalties = []
    for r in multipliers:
        # Avoid log of zero
        r_safe = max(r, 1e-5)
        p = abs(math.log10(r_safe))
        # Cap penalty at 1.0 (meaning a 10x difference is a maximum penalty)
        penalties.append(min(p, 1.0))
    
    # Average the penalties
    avg_penalty = sum(penalties) / len(multipliers)
    
    # Map penalty [0, 1.0] to score [100, 0]
    # score = 100 * (1 - avg_penalty)
    score = max(0.0, 100.0 * (1.0 - avg_penalty))

    return float(score)

def get_rdrs_separated_scores(multipliers):
    """
    Computes separated scores using the same symmetric log logic.
    """
    if len(multipliers) < 5:
        return 0.0, 0.0
        
    m_e, m_vbm, m_ms, m_c, m_ced = multipliers
    
    # Quality (Preservation)
    q_mults = [m_e, m_vbm, m_ms]
    q_penalties = [min(abs(math.log10(max(r, 1e-5))), 1.0) for r in q_mults]
    avg_q_penalty = sum(q_penalties) / 3
    quality_score = max(0.0, 100.0 * (1.0 - avg_q_penalty))

    # Style Shift (Match)
    s_mults = [m_c, m_ced]
    s_penalties = [min(abs(math.log10(max(r, 1e-5))), 1.0) for r in s_mults]
    avg_s_penalty = sum(s_penalties) / 2
    # Style shift index: 0.0 is perfect match, 1.0 is 10x difference
    style_shift_index = avg_s_penalty
    
    return quality_score, style_shift_index
