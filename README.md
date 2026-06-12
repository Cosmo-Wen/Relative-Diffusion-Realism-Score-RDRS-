# Unified Diffusion Realism Score (UDRS)

UDRS is a multi-tiered evaluation framework designed to assess the physical, perceptual, and semantic realism of images edited via diffusion models. It aggregates four distinct evaluation tiers into a single holistic percentage score.

## Quickstart (Under 5 Minutes)

1. **Activate Environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Full Pipeline:**
   ```bash
   python main.py --config config.yaml --plot
   ```

4. **Run Tests:**
   ```bash
   pytest tests/
   ```

## Tiered Evaluation Architecture
- **Tier 1: Structural Realism**: Low-level physical integrity using mask-aware evaluation to isolate hair style from background preservation.
- **Tier 2: Perceptual Realism**: High-level naturalness using ResNet-18 deep features.
- **Tier 3: Semantic Realism**: Logical and relational integrity (Mocked API).
- **Tier 4: Style Fidelity**: Visual consistency using OpenCLIP zero-shot classification.

## Masked Evaluation (v2.2)
The pipeline now supports semantic masking via a modular segmentation layer. By default, it uses a `MockSegmenter` for a lightweight, 5-minute setup that isolates structural changes in the center of the frame.

## Project Structure
- `src/`: Multi-tiered logic (`rdrs_core.py`, `raise_perceptual.py`, `real_semantic.py`, `real_style.py`).
- `data/`: Image assets (Original and Edited).
- `tests/`: Comprehensive unit and integration test suite for all tiers.
- `main.py`: Unified entry point for the UDRS pipeline.
- `config.yaml`: Configurable weights and image paths.
