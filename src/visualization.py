import numpy as np
import matplotlib.pyplot as plt

def plot_rdrs_pentagon(multipliers, final_score=None, output_path="rdrs_plot.png"):
    """
    Plots the RDRS pentagon (radar chart) with descriptive labels and interpretation guides.
    """
    # 1. Added brief descriptions using newlines to keep it clean
    labels = [
        'MS\n(High-Freq Noise)', 
        'GLCM_C\n(Strand Contrast)', 
        'CED\n(Edge Volume)', 
        'GLCM_E\n(Uniformity)', 
        'VBM\n(Sharpness)'
    ]
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is circular, so we need to "complete the loop"
    multipliers = multipliers[-1:] + multipliers[:-1] 
    multipliers += multipliers[:1]
    angles += angles[:1]

    # Baseline (all 1.0)
    baseline = [1.0] * (num_vars + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 2. Squeeze the polar plot to the left to make room for legends/text on the right
    fig.subplots_adjust(left=0.05, right=0.65)
    
    # Draw the baseline
    ax.plot(angles, baseline, color='blue', linewidth=2, linestyle='dashed', label='Original Baseline')
    ax.fill(angles, baseline, color='blue', alpha=0.1)

    # Draw the edited image multipliers
    ax.plot(angles, multipliers, color='red', linewidth=2, label='Edited Realism')
    ax.fill(angles, multipliers, color='red', alpha=0.25)

    # Fix axis to go in the right order and start at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    # 3. Add the legend (now safely inside the figure bounds)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
    plt.title("RDRS Feature Comparison", size=20, y=1.1)

    # 4. Add the interpretation text box
    score_text = f"Final Score: {final_score:.1f}%\n\n" if final_score is not None else ""
    
    info_text = (
        f"{score_text}"
        "How to read this chart:\n"
        "Blue Line (1.0) = Original Image\n\n"
        "Style Indicators (Right Side):\n"
        "• CED & GLCM_C\n"
        "  > 1.0: Added volume/texture\n"
        "  < 1.0: Smoothed/straightened\n\n"
        "Quality Constraints (Left Side):\n"
        "• VBM (Sharpness)\n"
        "  > 1.0: Over-sharpened / Crispy\n"
        "  < 1.0: Unnaturally blurry\n"
        "• MS (Noise)\n"
        "  > 1.0: Added grain / artifacts\n"
        "  < 1.0: Lost natural texture\n"
        "• GLCM_E (Uniformity)\n"
        "  > 1.0: Too perfect / plastic wig\n"
        "  < 1.0: Overly chaotic"
    )
    
    # Place the text box on the right side of the figure
    fig.text(0.70, 0.3, info_text, fontsize=11, va='center', ha='left', 
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.9))

    plt.savefig(output_path, bbox_inches='tight') # bbox_inches='tight' prevents any accidental cropping
    print(f"Visualization saved to {output_path}")
    plt.close()