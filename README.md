# Relative Diffusion Realism Score (RDRS)

RDRS is a metric pipeline designed to evaluate the physical realism of diffusion-edited images by benchmarking them against their original counterparts using geometric image statistics.

## Quickstart (Under 5 Minutes)

Recreate the project results by following these steps:

1. **Activate Environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Evaluation:**
   ```bash
   python main.py --config config.yaml
   ```

4. **Run Tests:**
   ```bash
   pytest tests/
   ```

## Project Structure
- `src/`: Core logic (Features, Normalization, Aggregation, Color Fidelity).
- `data/`: Image assets (Original and Edited).
- `tests/`: Unit and integration test suite.
- `main.py`: Entry point for the pipeline.
- `config.yaml`: Configuration settings for image paths.
- `design_choice.md`: Detailed documentation on architectural decisions.

## Metrics
- **RDRS Score:** A percentage representing global structural and textural realism.
- **Color Retention:** A percentage representing thematic color preservation using HSV Histogram Intersection.
