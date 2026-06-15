import numpy as np
import matplotlib.pyplot as plt

def plot_rdrs_pentagon(hair_multipliers, bg_multipliers, final_score=None, output_path="rdrs_plot.png"):
    """
    Plots the RDRS pentagon (radar chart) with descriptive labels and interpretation guides for dual zones.
    """
    labels = [
        'GLCM_C\n(Strand Contrast)', 
        'CED\n(Edge Volume)', 
        'GLCM_E\n(Uniformity)', 
        'VBM\n(Sharpness)',
        'MS\n(High-Freq Noise)'
    ]
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is circular, so we need to "complete the loop"
    hair_mults = hair_multipliers + hair_multipliers[:1]
    bg_mults = bg_multipliers + bg_multipliers[:1]
    angles += angles[:1]

    # Baseline (all 1.0)
    baseline = [1.0] * (num_vars + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Squeeze the polar plot to the left to make room for legends/text on the right
    fig.subplots_adjust(left=0.05, right=0.65)
    
    # Draw the baseline
    ax.plot(angles, baseline, color='blue', linewidth=2, linestyle='dashed', label='Baseline (Original/Style)')
    ax.fill(angles, baseline, color='blue', alpha=0.1)

    # Draw the Hair Zone multipliers
    ax.plot(angles, hair_mults, color='red', linewidth=2, label='Hair Zone (vs Style)')
    ax.fill(angles, hair_mults, color='red', alpha=0.25)

    # Draw the Background Zone multipliers
    ax.plot(angles, bg_mults, color='green', linewidth=2, label='Background Zone (vs Original)')
    ax.fill(angles, bg_mults, color='green', alpha=0.25)

    # Fix axis to go in the right order and start at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    # Add the legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15))
    plt.title("RDRS 5x2 Symmetric Feature Comparison", size=18, y=1.1)

    # Add the interpretation text box
    score_text = f"Final Score: {final_score:.1f}%\n\n" if final_score is not None else ""
    
    info_text = (
        f"{score_text}"
        "How to read this chart:\n"
        "Blue Line (1.0) = Baseline\n\n"
        "Zones:\n"
        "• Red (Hair) should match Style\n"
        "• Green (Background) should match Original\n\n"
        "Interpretation:\n"
        "• > 1.0: Added volume/texture/noise\n"
        "• < 1.0: Smoothed/blurred/loss of detail\n"
    )
    
    # Place the text box on the right side of the figure
    fig.text(0.70, 0.4, info_text, fontsize=11, va='center', ha='left', 
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))

    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()
