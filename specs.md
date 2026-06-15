# Unified Diffusion Realism Score (UDRS)

## 1. System Objective
The Unified Diffusion Realism Score (UDRS) is a multi-tiered evaluation framework designed to holistically assess the physical, perceptual, semantic, and artistic realism of images edited via diffusion models. It evaluates generated imagery against original backgrounds and intended style references, mitigating global score degradation caused by localized compositional edits. The pipeline aggregates four distinct tiers into a single percentage score.

---

## 2. Tier 1: Structural Realism (Core RDRS)
Evaluates low-level physical integrity and textural consistency using handcrafted image statistics.

### A. Mathematical Features
1. **GLCM Contrast ($GLCM_C$)**: Quantifies local pixel variations. Averaged over 4 angles (0, 45, 90, 135 degrees) for rotational invariance. Intensity levels are reduced to 64 for performance.
2. **Canny Edge Density ($CED$)**: The proportion of pixels classified as structural edges. Uses adaptive thresholding based on the image median ($\sigma = 0.33$).
3. **GLCM Energy ($GLCM_E$)**: Measures texture uniformity.
4. **Variance Blur Measure ($VBM$)**: Estimates image sharpness by computing the global variance of a Laplacian-filtered image.
5. **Mean Spectrum ($MS$)**: A frequency domain metric computing the average magnitude of the image's Fourier Transform (`np.fft.rfft2`). The DC component (0,0) is excluded to focus precisely on high-frequency diffusion noise.

### B. Isolated Spatial Segmentation & Triple Masking
To prevent localized structural edits (like hair) from incorrectly tanking the global background realism score, the framework utilizes an independent masking system.
- **Segmenter**: The `TransformersHairSegmenter` utilizes HuggingFace's `pipeline` with the `mattmdjaga/segformer_b2_clothes` Segformer model to deeply and semantically isolate the "Hair" class.
- **Triple Masking**: Three independent semantic masks are generated for the Original Image, Edited Image, and Style Reference Image. This guarantees that "hair is compared to hair" regardless of spatial misalignment or compositional differences between the three inputs.
- **Zone Evaluation**:
  - **Style Axes ($GLCM_C$, $CED$)**: Evaluated *inside* the mask (Hair Zone, `mask_target=255`) and benchmarked against the Style Reference Image.
  - **Quality Axes ($GLCM_E$, $VBM$, $MS$)**: Evaluated *outside* the mask (Preservation Zone, `mask_target=0`) and benchmarked against the Original Image.
- **Mask-Aware Mathematics**:
  - *GLCM Trick*: Masked-out pixels are assigned an out-of-bounds intensity of `64`. After co-occurrence accumulation, the matrix is sliced to exclude this level, ensuring texture properties are computed only for valid neighboring pairs.
  - *Boundary Consistency*: For Laplacian and FFT, pixels outside the target zone are hard-zeroed *before* kernel application to prevent edge artifact corruption.

### C. Symmetric Logarithmic Aggregation
The edited features are divided by their baseline counterparts to create ratios. Instead of linear absolute error, Tier 1 uses a symmetric logarithmic penalty:
$$Penalty = \min(|\log_{10}(\max(Ratio, 10^{-5}))|, 1.0)$$
This ensures doubling ($Ratio = 2.0$) and halving ($Ratio = 0.5$) a metric intensity result in the exact same penalty ($\approx 0.301$). The $1.0$ cap ensures that a single extreme outlier (a $10\times$ difference) cannot completely zero out the entire UDRS score.
* **Quality Score** is mapped from the average penalty of the 3 Quality Axes.
* **Style Shift Index** is mapped from the average penalty of the 2 Style Axes.

---

## 3. Tier 2: Deep Perceptual Realism
Evaluates high-level, human-aligned "naturalness."
- **Model**: A pre-trained `ResNet-18` (via PyTorch/Torchvision). The final classification layer is removed.
- **Extraction**: Images are resized to 256 and CenterCropped to 224 (to preserve aspect ratio). A 512-dimensional embedding is extracted from the global average pooling layer.
- **Scoring**: Cosine Similarity is computed between the Original and Edited embeddings. The similarity is mapped to a Gaussian probability curve, calibrated to target an optimal peak similarity of $\mu = 0.88$ with $\sigma = 0.06$.
$$Score = 100.0 \times \exp\left(-\frac{1}{2}\left(\frac{\text{Similarity} - 0.88}{0.06}\right)^2\right)$$

---

## 4. Tier 3: Semantic & Relational Realism
Evaluates the logical consistency and relational plausibility of objects.
- **Architecture**: Defines a `VQABackend` interface for querying Vision-Language Models.
- **Metrics**: 
  - $S_{att}$ (Attribute Score): $\frac{\text{Correct Attributes}}{\text{Visible Attributes}}$
  - $S_{rel}$ (Relationship Score): $\frac{\text{Realistic Relationships}}{\text{Visible Relationships}}$
- **Implementation Status**: Currently simulated to avoid spending API credits or loading heavy local VLMs. The `MockVQABackend` simulates responses. In the main pipeline execution, this tier explicitly returns `0.0` to clearly signal its simulated/non-working status.

---

## 5. Tier 4: Visual Style Fidelity
Ensures the diffusion process does not inadvertently shift the artistic medium of the image.
- **Model**: `OpenCLIP` using a lightweight `ViT-B-32` architecture.
- **Scoring**: Performs zero-shot classification comparing the edited image against three text tokens: `["a photo", "an illustration", "a painting"]`.
- The final Tier 4 score is the raw Softmax probability that the image remains "a photo", mapped to a 0-100% scale.

---

## 6. Pipeline Configuration & Architecture
The system features a heavily debloated CLI, relying entirely on a centralized YAML configuration file.

### CLI Interface
```bash
python main.py --config config.yaml
```

### Configuration (`config.yaml`)
- **`images`**: Defines `original`, `style`, and `edited` paths for single-triplet evaluation.
- **`settings`**:
  - `batch_csv`: If provided (e.g., `triplets.csv`), runs bulk evaluation and averages scores.
  - `plot`: Generates a geometric radar chart for Tier 1 features (single mode only).
  - `use_mask`: Toggles the spatial segmentation layer. If `false`, evaluation reverts to a global comparison.
  - `save_masks`: If `true`, independent transparent green mask overlays for the Original, Edited, and Style images are saved to `debug_masks/` to visually prove segmentation alignment.
- **`weights`**: Adjusts the aggregation weight of the 4 tiers (e.g., $T_1=0.3, T_2=0.3, T_3=0.2, T_4=0.2$).