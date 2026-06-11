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

## 5. Version Control
The project was initialized with Git from the start to maintain a clean history and allow for iterative development and easy rollbacks.
