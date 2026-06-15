# Unified Diffusion Realism Score (UDRS)

UDRS is a multi-tiered evaluation framework designed to assess the physical, perceptual, and semantic realism of images edited via diffusion models. It aggregates four distinct evaluation tiers into a single holistic percentage score.

## Quickstart (Under 5 Minutes)

1. **Activate Environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   The `requirements.txt` includes standard computer vision tools (`opencv`, `scikit-image`) alongside necessary deep learning frameworks (`torch`, `transformers`, `open-clip-torch`) to power the semantic tiers.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure & Run:**
   Edit `config.yaml` to set your image paths, enable batch mode, or toggle masking. Then run:
   ```bash
   python main.py --config config.yaml
   ```

4. **Run Tests:**
   ```bash
   pytest tests/
   ```

## Configuration (config.yaml)
All pipeline behaviors are managed in `config.yaml`:
- **`images`**: Paths for single-triplet evaluation.
- **`settings`**:
    - `batch_csv`: Path to a CSV (e.g., `triplets.csv`) to enable batch processing.
    - `output_csv`: Path to save the batch evaluation results (e.g., `batch_results.csv`).
    - `plot`: Set to `true` to generate a radar chart (single mode only).
    - `use_mask`: Toggle `true`/`false` to enable/disable isolated spatial segmentation (v2.2).
    - `save_masks`: Save mask overlays for debugging independently.

## Tiered Evaluation Architecture
- **Tier 1: Structural Realism**: Low-level physical integrity using mask-aware evaluation to isolate hair style from background preservation.
- **Tier 2: Perceptual Realism**: High-level naturalness using ResNet-18 deep features.
- **Tier 3: Semantic Realism**: Logical and relational integrity (Mocked API).
- **Tier 4: Style Fidelity**: Visual consistency using OpenCLIP zero-shot classification.

## Project Structure
- `src/`: Multi-tiered logic (`rdrs_core.py`, `raise_perceptual.py`, `real_semantic.py`, `real_style.py`, `segmentation.py`).
- `data/`: Image assets.
- `tests/`: Comprehensive unit and integration test suite.
- `main.py`: Unified entry point.
