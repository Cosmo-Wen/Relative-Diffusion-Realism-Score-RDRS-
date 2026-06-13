# Design Choices: RDRS Implementation

This document outlines architectural and technical decisions made during the implementation of the RDRS pipeline that were not explicitly defined in `specs.md`.

## 1. Modular Architecture
Instead of a monolithic `rdrs_core.py`, the logic was separated into specialized modules under `src/`:
- `features.py`: Atomic feature extraction logic using `scikit-image` and `OpenCV`.
- `normalization.py`: Handles mathematical calibration and relative multiplier calculation.
- `aggregation.py`: Pure geometric math for pentagon area calculation.
- `color_fidelity.py`: Decoupled color retention metric.

This ensures that each component can be unit-tested in isolation and easily swapped or extended.

## 2. Library Selection
- **scikit-image**: Chosen for GLCM calculations as it provides high-level, well-tested functions for textural analysis that OpenCV lacks natively.
- **OpenCV (headless)**: Used for Canny edge detection, HSV conversion, and Laplacian variance. Headless version was selected for CLI environment compatibility.
- **NumPy**: The backbone for FFT and array manipulations.
- **PyYAML**: Used for configuration management to avoid bloated CLI flags.

## 3. Configuration Management
A `config.yaml` file was introduced to manage image paths and system settings. This allows users to point the pipeline to different image pairs without modifying the code or passing long lists of CLI arguments.

## 4. Error Handling
Added basic validation to ensure images exist before processing and used epsilon values (`1e-10`) to prevent division-by-zero errors during feature inversion and normalization.

## 6. Enhancements (feature/enhancements branch)

Several mathematical and functional improvements were made to the core pipeline:

- **Rotational Invariance (GLCM)**: Feature extraction now averages over 0, 45, 90, and 135 degrees to ensure the realism score is not biased by image orientation.
- **Adaptive Canny Thresholding**: Thresholds are now computed dynamically based on the image median, making edge density analysis robust across different lighting and contrast levels.
- **High-Frequency Spectral Noise**: The Mean Spectrum calculation now excludes the DC component, focusing more precisely on high-frequency noise artifacts typically introduced by diffusion steps.
- **L1 Histogram Normalization**: Switched from MinMax to L1 normalization in `color_fidelity.py` to ensure histogram intersection represents a valid shared probability distribution.
## 7. Unified Diffusion Realism Score (UDRS) Architecture

The project has been upgraded to a multi-tiered evaluation framework:

- **Tier 1: Structural Realism (RDRS Core)**: Handcrafted image statistics (GLCM, Canny, FFT) aggregated via a pentagon model.
- **Tier 2: Deep Perceptual Realism**: Uses a pre-trained ResNet-18 model to extract 512D embeddings from the global average pooling layer. Similarity is measured via Cosine Similarity.
- **Tier 3: Semantic & Relational Realism**: A modular VQA-based evaluation. Currently implemented with a `MockVQABackend` to simulate cloud API responses for attribute and relationship checks without loading local VLMs.
- **Tier 4: Visual Style Fidelity**: Uses the `OpenCLIP` library (ViT-B-32) for zero-shot style classification. It measures the probability that the edited image maintains a "photo" style versus becoming an "illustration" or "painting".

## 8. Weighted Aggregation

The final UDRS score is calculated as a weighted sum of all implemented tiers. Weights are fully configurable via `config.yaml`, allowing for flexible evaluation depending on compute availability and use-case priority.

## 10. Isolated Spatial Segmentation (v2.2)

To prevent localized edits (like hair) from incorrectly tanking the global realism score, a spatial segmentation layer was introduced:

- **BaseSegmenter Abstraction**: A unified interface for segmentation backends, allowing for hot-swapping between Mock, PoC (MediaPipe), and High-Fidelity (SAM) models.
- **HeuristicHairSegmenter (Robust model-free)**: Due to environment-specific library blockers with MediaPipe, a model-free `HeuristicHairSegmenter` was implemented. It uses standard OpenCV Haar Cascades to detect the face and dynamically defines the "Hair Zone" as the region directly above and surrounding the head. This approach is highly robust, instant, and perfectly handles pixel shifts as it re-localizes the hair in each image independently.
- **Independent Style Masking**: To ensure valid textural baselines, the pipeline now generates a *separate* semantic mask for the style reference image. This prevents the spatial coordinates of the edited image's hair from being incorrectly applied to a different style reference image.
- **Symmetric Logarithmic Scoring (Aggregation)**: The Tier 1 penalty formula was upgraded from linear `abs(1.0 - ratio)` to logarithmic `abs(math.log10(ratio))`. This makes the scoring symmetric (doubling or halving a metric intensity results in the same penalty). A strict cap of 1.0 per metric was added to ensure that a single extreme outlier (like a noisy Canny spike) cannot arbitrarily zero out the entire UDRS score, while still maintaining high strictness for unrealistic artifacts.
- **Mask-Aware Evaluation**: The frame is semantically split into a "Hair Zone" (evaluated against the Style Reference) and a "Preservation Zone" (evaluated against the Original Image).
- **Masked GLCM Implementation**: Standard GLCM functions do not support irregular masks. To implement this mathematically, masked-out pixels are assigned an intensity level of `64` (outside the `0-63` valid range). After co-occurrence accumulation, the matrix is sliced to exclude the `64th` level, ensuring texture properties are only computed for valid neighboring pixel pairs within the semantic zone.
- **Localized Statistics**: Canny density, Laplacian variance, and FFT mean spectrum are computed strictly on masked pixel subsets to ensure local changes do not corrupt global preservation assessments. Boundary artifacts are minimized by zero-masking images *before* kernel application.

## 11. Global Evaluation Toggle (--no-mask)

A `--no-mask` flag was added to the entry point to allow users to bypass the spatial segmentation layer. When enabled:
- The `segmenter` instance is set to `None`.
- Feature extraction reverts to global image statistics while maintaining the dual-baseline logic (Style vs. Style image, Quality vs. Original image).
- This allows for "original method" benchmarking where localized semantic isolation is not desired.
