import yaml
import argparse
from src.rdrs_core import calculate_tier1_score
from src.raise_perceptual import calculate_tier2_score
from src.real_semantic import calculate_tier3_score
from src.real_style import calculate_tier4_score
from src.visualization import plot_rdrs_pentagon

# Unified Pipeline
def run_pipeline(config_path, plot=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    orig_path = config['images']['original']
    style_path = config['images']['style']
    edit_path = config['images']['edited']
    weights = config.get('weights', {})
    
    w1 = weights.get('tier1_structural', 0)
    w2 = weights.get('tier2_perceptual', 0)
    w3 = weights.get('tier3_semantic', 0)
    w4 = weights.get('tier4_style', 0)
    
    print(f"--- UDRS Evaluation Pipeline ---")
    print(f"Original: {orig_path}")
    print(f"Style:    {style_path}")
    print(f"Edited:   {edit_path}")
    print(f"Weights:  T1={w1}, T2={w2}, T3={w3}, T4={w4}")
    print(f"--------------------------------")
    
    # Tier 1: Structural
    print("Calculating Tier 1: Structural Realism...")
    if w1 > 0:
        t1_score, multipliers = calculate_tier1_score(orig_path, edit_path, style_path)
    else:
        t1_score = 0
        multipliers = {}

    # Tier 2: Perceptual (Placeholder)
    if w2 > 0:
        t2_score = calculate_tier2_score(orig_path, edit_path)
    else:
        t2_score = 0
    
    # Tier 3: Semantic (Placeholder)
    if w3 > 0:
        t3_score = calculate_tier3_score(orig_path, edit_path)
    else:
        t3_score = 0
    
    # Tier 4: Style (Placeholder)
    if w4 > 0:
        t4_score = calculate_tier4_score(orig_path, edit_path)
    else:
        t4_score = 0
    
    # Weighted Aggregation
    udrs_score = (w1 * t1_score) + (w2 * t2_score) + (w3 * t3_score) + (w4 * t4_score)
    
    # Visualization
    if plot:
        print("Generating visualization...")
        plot_rdrs_pentagon(multipliers.copy(), final_score=t1_score, output_path="udrs_tier1_plot.png")
    
    # Output
    print(f"\nTier Scores:")
    print(f"  - Tier 1 (Structural): {t1_score:.2f}%")
    print(f"  - Tier 2 (Perceptual): {t2_score:.2f}%")
    print(f"  - Tier 3 (Semantic):   {t3_score:.2f}%")
    print(f"  - Tier 4 (Style):      {t4_score:.2f}%")
    
    print(f"\n================================")
    print(f"FINAL UDRS SCORE:     {udrs_score:.2f}%")
    print(f"================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Diffusion Realism Score (UDRS) Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--plot", action="store_true", help="Generate Tier 1 pentagon plot")
    args = parser.parse_args()
    
    run_pipeline(args.config, args.plot)
