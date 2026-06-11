# Relative Diffusion Realism Score (RDRS)

System Design Specification

1. System Objective

The RDRS pipeline is designed to evaluate the physical realism of an image edited via a diffusion model by benchmarking it against its unedited, original counterpart. The system computes five primitive image statistics, maps them to a geometric polygon, and outputs a single percentage score representing the retained realism of the global image.

2. Core Mathematical Features

The system processes grayscale representations of both the original and edited images to extract five distinct textural and structural features.

GLCM Contrast ($GLCM_C$): Quantifies local pixel variations. Smoothed or over-denoised generated backgrounds will exhibit a drop in contrast.

Canny Edge Density (CED): The proportion of pixels classified as structural edges. Diffusion models often alter the natural edge density distribution during content generation.

GLCM Energy ($GLCM_E$): Measures texture uniformity.

Variance Blur Measure (VBM): Estimates image sharpness by computing the global variance of a Laplacian-filtered image.

Mean Spectrum (MS): A frequency domain metric computing the average magnitude of the image's Fourier Transform. Unnatural noise injection from diffusion steps heavily impacts this metric.

3. Calibration and Normalization Engine

Unlike standard metrics that rely on large datasets, RDRS uses a pair-wise self-calibration approach:

Feature Inversion: Diffusion generation generally artificially inflates uniformity, frequency noise, and blur variance. To maintain a bounded geometric space, the system takes the multiplicative inverse of $GLCM_E$, VBM, and MS.

Baseline Normalization: The original image's calibrated features are mapped to a perfect theoretical baseline of 1.0.

Relative Scaling: The edited image's calibrated features are divided by the original image's corresponding features to create five normalized multipliers ($m_1$ through $m_5$).

4. Geometric Aggregation Model

The final score is computed by mapping the five normalized multipliers to the radii of a pentagon.

Arrangement Sequence: To maximize relational differences, the radii are fixed in the following sequence: $GLCM_C \rightarrow CED \rightarrow GLCM_E \rightarrow VBM \rightarrow MS$.

Area Computation: The total area of the pentagon is the sum of the five triangles formed by adjacent radii. The central angle is fixed at 72° ($\frac{2\pi}{5}$ radians).

$Area = \sum \frac{1}{2} (m_i \cdot m_{i+1}) \cdot \sin(72^\circ)$

Final Output: The area of the perfect baseline (where all radii = 1.0) is $\approx 2.377$. The final RDRS is (Edited_Area / 2.377) * 100, bounded between 0% and a theoretical cap.

5. Project Directory Organization

The repository should be structured to isolate the core mathematics from fallback metrics and testing environments.

```text
diffusion_realism_eval/
│
├── data/
│   ├── original/              # Unedited baseline images
│   └── edited/                # Diffusion-edited variants
│
├── src/
│   ├── __init__.py
│   ├── rdrs_core.py           # Core RDRS pentagon implementation
│   └── color_fidelity.py      # Next Step: Color preservation metric
│
├── tests/
│   ├── __init__.py
│   ├── test_features.py       # Unit tests for GLCM, CED, MS math
│   └── test_integration.py    # End-to-end pair-wise tests
│
├── requirements.txt
└── README.md
```

6. Testing Strategy

To ensure the mathematical validity of the pipeline, testing is broken into two phases:

Unit Testing (test_features.py): Pass synthetic images (e.g., pure black, pure white, random noise) into the 5 feature extractors to verify boundaries (e.g., CED of a pure black image must be 0).

Integration Testing (test_integration.py):

Identity Test: Pass the exact same original image as both inputs. The system must return exactly 100.0%.

User Validation Test: Utilize the specifically provided Original/Edited pair. Output the individual feature degradation breakdown alongside the final pentagon score for manual review.

7. Next Steps: Color Discard & Fidelity Assessment

Because the RDRS relies exclusively on grayscale structural data, a diffusion model could accidentally turn a vibrant green forest into a neon purple forest, and the RDRS might score it highly if the texture is preserved.

Solution: Run a parallel color_fidelity.py script that converts both images to the HSV (Hue, Saturation, Value) color space. By computing the 3D Color Histogram Intersection between the original and the edited image, we can generate a secondary percentage score representing "Color Retention". This ensures both physical realism and thematic color intent are preserved globally across the background and foreground.