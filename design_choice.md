# Design Choices: RDRS Implementation

This document outlines architectural and technical decisions made during the implementation of the RDRS pipeline that were not explicitly defined in the core `specs.md`.

## 1. 5x2 Symmetric Model Architecture (v2.3)

To eliminate blind spots (e.g., blur in the hair or jagged edges in the background), the Tier 1 Structural Realism evaluation was overhauled into a 5x2 Symmetric Model:
- **Dual-Zone Evaluation**: All 5 structural metrics (CED, VBM, MS, GLCM_C, GLCM_E) are calculated simultaneously on *both* the Hair Zone and the Background Zone.
- **Independent Baselines**: 
  - The **Hair Zone** (Foreground) compares the Edited Image to the Style Reference Image.
  - The **Background Zone** (Inverse Mask) compares the Edited Image to the Original Image.
- **Linearization & Safe Math**: Quadratic metrics (VBM, GLCM_C, GLCM_E) are linearized via square root before ratio calculation. A `safe_ratio` helper handles zero-division (e.g., solid color backgrounds where CED=0) by falling back to `1.0` if both the baseline and edit are mathematically near-zero.
- **Two-Pillar Aggregation**: The final Tier 1 score is the simple average of the Hair Score and the Background Score.
- **Dual Visualization**: The `plot_rdrs_pentagon` function was updated to plot both zones' multipliers on the same radar chart (Red for Hair, Green for Background), allowing for direct visual comparison of structural degradation across zones.
