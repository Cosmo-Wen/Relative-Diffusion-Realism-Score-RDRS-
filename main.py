import yaml
import argparse
from src.features import extract_all_features
from src.normalization import get_multipliers
from src.aggregation import get_rdrs_score
from src.color_fidelity import get_color_fidelity_score

def run_pipeline(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    orig_path = config['images']['original']
    edit_path = config['images']['edited']
    
    print(f"--- RDRS Evaluation Pipeline ---")
    print(f"Original: {orig_path}")
    print(f"Edited:   {edit_path}")
    print(f"--------------------------------")
    
    # 1. Feature Extraction
    print("Extracting features...")
    orig_features = extract_all_features(orig_path)
    edit_features = extract_all_features(edit_path)
    
    # 2. Normalization
    print("Normalizing features...")
    multipliers = get_multipliers(orig_features, edit_features)
    
    # 3. Geometric Aggregation
    print("Calculating RDRS score...")
    rdrs_score = get_rdrs_score(multipliers)
    
    # 4. Color Fidelity
    print("Calculating Color Fidelity score...")
    color_score = get_color_fidelity_score(orig_path, edit_path)
    
    # Breakdown
    labels = ['GLCM_C', 'CED', 'GLCM_E', 'VBM', 'MS']
    print("\nFeature Degradation Breakdown (Multipliers):")
    for label, m in zip(labels, multipliers):
        print(f"  - {label}: {m:.4f}")
    
    print(f"\n================================")
    print(f"FINAL RDRS SCORE:     {rdrs_score:.2f}%")
    print(f"COLOR RETENTION:      {color_score:.2f}%")
    print(f"================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relative Diffusion Realism Score (RDRS) Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    run_pipeline(args.config)
