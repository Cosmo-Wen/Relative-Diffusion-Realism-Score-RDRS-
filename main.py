import yaml
import argparse
import csv
import os
from src.rdrs_core import calculate_tier1_score
from src.raise_perceptual import calculate_tier2_score
from src.real_semantic import calculate_tier3_score
from src.real_style import calculate_tier4_score
from src.visualization import plot_rdrs_pentagon
from src.segmentation import TransformersHairSegmenter

def evaluate_single_triplet(orig_path, edit_path, style_path, weights, segmenter, save_masks=False):
    """
    Evaluates a single triplet of images and returns the final UDRS score and individual tier scores.
    """
    w1 = weights.get('tier1_structural', 0)
    w2 = weights.get('tier2_perceptual', 0)
    w3 = weights.get('tier3_semantic', 0)
    w4 = weights.get('tier4_style', 0)
    
    # Tier 1: Structural
    if w1 > 0:
        t1_scores, multipliers = calculate_tier1_score(orig_path, edit_path, style_path, segmenter=segmenter, save_masks=save_masks)
        t1_score = t1_scores['final']
        t1_hair = t1_scores['hair']
        t1_bg = t1_scores['bg']
    else:
        t1_score, t1_hair, t1_bg = 0, 0, 0
        multipliers = {'hair': [], 'bg': []}

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
    
    return udrs_score, (t1_score, t1_hair, t1_bg, t2_score, t3_score, t4_score), multipliers

def run_pipeline(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract Settings
    settings = config.get('settings', {})
    batch_csv = settings.get('batch_csv')
    plot = settings.get('plot', False)
    use_mask = settings.get('use_mask', True)
    save_masks = settings.get('save_masks', False)
    weights = config.get('weights', {})
    
    # Initialize Segmenter (TransformersHairSegmenter is an actual semantic model)
    segmenter = TransformersHairSegmenter() if use_mask else None
    
    if batch_csv:
        output_csv = settings.get('output_csv', 'batch_results.csv')
        print(f"--- UDRS Batch Evaluation Pipeline ---")
        print(f"CSV Path: {batch_csv}")
        print(f"Output CSV: {output_csv}")
        print(f"Masking:  {'Enabled' if use_mask else 'Disabled'}")
        print(f"--------------------------------------")
        
        results = []
        output_rows = []
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
                
                score, tiers, multipliers = evaluate_single_triplet(orig_path, edit_path, style_path, weights, segmenter, save_masks=save_masks)
                results.append(score)
                t1, t1_hair, t1_bg, t2, t3, t4 = tiers
                
                hair_mults = multipliers.get('hair', [0]*5)
                bg_mults = multipliers.get('bg', [0]*5)
                
                output_rows.append({
                    'source_path': orig_path,
                    'edit_path': edit_path,
                    'style_path': style_path,
                    'UDRS_score': score,
                    'Tier1_Avg': t1,
                    'Tier1_Hair': t1_hair,
                    'Tier1_Bg': t1_bg,
                    'Tier2': t2,
                    'Tier3': t3,
                    'Tier4': t4,
                    'Hair_GLCM_C': hair_mults[0] if len(hair_mults) > 0 else 0,
                    'Hair_CED': hair_mults[1] if len(hair_mults) > 1 else 0,
                    'Hair_GLCM_E': hair_mults[2] if len(hair_mults) > 2 else 0,
                    'Hair_VBM': hair_mults[3] if len(hair_mults) > 3 else 0,
                    'Hair_MS': hair_mults[4] if len(hair_mults) > 4 else 0,
                    'Bg_GLCM_C': bg_mults[0] if len(bg_mults) > 0 else 0,
                    'Bg_CED': bg_mults[1] if len(bg_mults) > 1 else 0,
                    'Bg_GLCM_E': bg_mults[2] if len(bg_mults) > 2 else 0,
                    'Bg_VBM': bg_mults[3] if len(bg_mults) > 3 else 0,
                    'Bg_MS': bg_mults[4] if len(bg_mults) > 4 else 0,
                })
                
                print(f"Processed: {edit_path} | T1_Hair: {t1_hair:.2f}% | T1_Bg: {t1_bg:.2f}% | T1_Avg: {t1:.2f}% | UDRS: {score:.2f}%")
        
        if results:
            with open(output_csv, 'w', newline='') as f:
                fieldnames = [
                    'source_path', 'edit_path', 'style_path', 'UDRS_score', 
                    'Tier1_Avg', 'Tier1_Hair', 'Tier1_Bg', 'Tier2', 'Tier3', 'Tier4', 
                    'Hair_GLCM_C', 'Hair_CED', 'Hair_GLCM_E', 'Hair_VBM', 'Hair_MS',
                    'Bg_GLCM_C', 'Bg_CED', 'Bg_GLCM_E', 'Bg_VBM', 'Bg_MS'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)
            
            avg_score = sum(results) / len(results)
            print(f"\n================================")
            print(f"BATCH EVALUATION COMPLETE")
            print(f"Triplets Processed: {len(results)}")
            print(f"AVERAGE UDRS SCORE: {avg_score:.2f}%")
            print(f"Results saved to {output_csv}")
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
        
        score, tiers, multipliers = evaluate_single_triplet(orig_path, edit_path, style_path, weights, segmenter, save_masks=save_masks)
        t1, t1_hair, t1_bg, t2, t3, t4 = tiers
        
        # Visualization
        if plot:
            print("Generating visualization...")
            plot_rdrs_pentagon(
                multipliers.get('hair', [0]*5), 
                multipliers.get('bg', [0]*5), 
                final_score=t1, 
                output_path="udrs_tier1_plot.png"
            )
        
        # Output
        print(f"\nTier Scores:")
        print(f"  - Tier 1 (Structural Avg): {t1:.2f}%")
        print(f"      - Hair Zone:       {t1_hair:.2f}%")
        print(f"      - Background Zone: {t1_bg:.2f}%")
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
