import yaml
import argparse
import csv
import os
from src.rdrs_core import calculate_tier1_score
from src.raise_perceptual import calculate_tier2_score
from src.real_semantic import calculate_tier3_score
from src.real_style import calculate_tier4_score
from src.visualization import plot_rdrs_pentagon
from src.segmentation import MockSegmenter

def evaluate_single_triplet(orig_path, edit_path, style_path, weights, segmenter):
    """
    Evaluates a single triplet of images and returns the final UDRS score and individual tier scores.
    """
    w1 = weights.get('tier1_structural', 0)
    w2 = weights.get('tier2_perceptual', 0)
    w3 = weights.get('tier3_semantic', 0)
    w4 = weights.get('tier4_style', 0)
    
    # Tier 1: Structural
    if w1 > 0:
        t1_score, multipliers = calculate_tier1_score(orig_path, edit_path, style_path, segmenter=segmenter)
    else:
        t1_score = 0
        multipliers = {}

    # Tier 2: Perceptual
    if w2 > 0:
        t2_score = calculate_tier2_score(orig_path, edit_path)
    else:
        t2_score = 0
    
    # Tier 3: Semantic
    if w3 > 0:
        t3_score = calculate_tier3_score(orig_path, edit_path)
    else:
        t3_score = 0
    
    # Tier 4: Style
    if w4 > 0:
        t4_score = calculate_tier4_score(orig_path, edit_path)
    else:
        t4_score = 0
    
    # Weighted Aggregation
    udrs_score = (w1 * t1_score) + (w2 * t2_score) + (w3 * t3_score) + (w4 * t4_score)
    
    return udrs_score, (t1_score, t2_score, t3_score, t4_score), multipliers

def run_pipeline(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract Settings
    settings = config.get('settings', {})
    batch_csv = settings.get('batch_csv')
    plot = settings.get('plot', False)
    use_mask = settings.get('use_mask', True)
    weights = config.get('weights', {})
    
    # Initialize Segmenter
    segmenter = MockSegmenter() if use_mask else None
    
    if batch_csv:
        print(f"--- UDRS Batch Evaluation Pipeline ---")
        print(f"CSV Path: {batch_csv}")
        print(f"Masking:  {'Enabled' if use_mask else 'Disabled'}")
        print(f"--------------------------------------")
        
        results = []
        with open(batch_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig_path = row['source_path']
                edit_path = row['edit_path']
                style_path = row['style_path']
                
                # Check if files exist
                if not all(os.path.exists(p) for p in [orig_path, edit_path, style_path]):
                    print(f"Skipping triplet (files missing): {orig_path}, {edit_path}, {style_path}")
                    continue
                
                score, tiers, _ = evaluate_single_triplet(orig_path, edit_path, style_path, weights, segmenter)
                results.append(score)
                print(f"Processed: {edit_path} | UDRS: {score:.2f}%")
        
        if results:
            avg_score = sum(results) / len(results)
            print(f"\n================================")
            print(f"BATCH EVALUATION COMPLETE")
            print(f"Triplets Processed: {len(results)}")
            print(f"AVERAGE UDRS SCORE: {avg_score:.2f}%")
            print(f"================================")
        else:
            print("No triplets were successfully processed.")
            
    else:
        orig_path = config['images']['original']
        style_path = config['images']['style']
        edit_path = config['images']['edited']
        
        print(f"--- UDRS Evaluation Pipeline ---")
        print(f"Original: {orig_path}")
        print(f"Style:    {style_path}")
        print(f"Edited:   {edit_path}")
        print(f"Masking:  {'Enabled' if use_mask else 'Disabled'}")
        print(f"Weights:  T1={weights.get('tier1_structural', 0)}, T2={weights.get('tier2_perceptual', 0)}, T3={weights.get('tier3_semantic', 0)}, T4={weights.get('tier4_style', 0)}")
        print(f"--------------------------------")
        
        score, tiers, multipliers = evaluate_single_triplet(orig_path, edit_path, style_path, weights, segmenter)
        t1, t2, t3, t4 = tiers
        
        # Visualization
        if plot:
            print("Generating visualization...")
            plot_rdrs_pentagon(multipliers.copy(), final_score=t1, output_path="udrs_tier1_plot.png")
        
        # Output
        print(f"\nTier Scores:")
        print(f"  - Tier 1 (Structural): {t1:.2f}%")
        print(f"  - Tier 2 (Perceptual): {t2:.2f}%")
        print(f"  - Tier 3 (Semantic):   {t3:.2f}%")
        print(f"  - Tier 4 (Style):      {t4:.2f}%")
        
        print(f"\n================================")
        print(f"FINAL UDRS SCORE:     {score:.2f}%")
        print(f"================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Diffusion Realism Score (UDRS) Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_pipeline(args.config)
