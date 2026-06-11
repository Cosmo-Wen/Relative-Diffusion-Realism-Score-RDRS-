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
- **Visualization Module**: Added `src/visualization.py` to generate radar charts (pentagons), allowing for intuitive visual comparison of the 5 realism features.
- **Performance Optimizations**: 
    - Reduced GLCM levels to 64 for a 4x speedup in texture analysis.
    - Switched to `np.fft.rfft2` for faster spectral analysis on real-valued images.
