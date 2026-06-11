import numpy as np
import matplotlib.pyplot as plt

def plot_rdrs_pentagon(multipliers, output_path="rdrs_plot.png"):
    """
    Plots the RDRS pentagon (radar chart).
    """
    labels = ['MS', 'GLCM_C', 'CED', 'GLCM_E', 'VBM']
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is circular, so we need to "complete the loop"
    multipliers = multipliers[-1:] + multipliers[:-1] 
    multipliers += multipliers[:1]
    angles += angles[:1]

    # Baseline (all 1.0)
    baseline = [1.0] * (num_vars + 1)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
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

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("RDRS Feature Comparison", size=20, y=1.1)

    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.close()
