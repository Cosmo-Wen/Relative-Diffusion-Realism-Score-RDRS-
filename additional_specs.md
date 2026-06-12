# Modification Specification: Isolated Spatial Segmentation & Modular Masking Tiers (v2.2)

## 1. Problem Statement & Architectural Driver
Global image evaluations degrade when an edit introduces massive structural changes (e.g., transforming a bald or short-haired profile into voluminous hair). Because the structural metrics ($GLCM$, $CED$, $VBM$, $MS$) evaluate the frame as a single matrix, replacing a smooth background with complex hair strand textures tanks the preservation score to 0%, even if the background remains untouched. 

To correct this, the pipeline must break the image into isolated semantic zones via a binary mask:
1. **The Hair Zone (Within Mask):** Evaluated strictly against the **Style Reference Image** to score how well the generated textures match the intended hair profile.
2. **The Preservation Zone (Outside Mask):** Evaluated strictly against the **Original Image** to guarantee that the face, clothing, and background canvas suffer no degradation, blurring, or structural hallucinations.

---

## 2. Git Workflow & Branch Strategy
All modifications outlined in this document must be developed outside the mainline production environment.
* **Target Branch:** `feature/isolated-segmentation-masking`
* **Workflow:** Branch off `main` or the current stable release. No code is to be merged back into `main` until it passes the integration testing suite with the mock backend verification.

---

## 3. Modular Segmentation Abstract Layer (Backend Swapping)
To prevent tight coupling to an individual computer vision engine, the codebase must enforce a strict interface contract. This allows a lightweight framework like SegFormer or MediaPipe to be used for rapid Proof of Concept (PoC) development, while leaving the architecture fully prepared to hot-swap to high-fidelity backends like SAM (Segment Anything Model) or custom internal models later without rewriting downstream math.

### Abstract Interface Definition
An abstract base class `BaseSegmenter` establishes a unified interface signature. Every backend must ingest a standard 3D NumPy BGR array and return a single-channel binary mask of identical spatial dimensions, where pixel value `255` denotes hair and `0` denotes background/face canvas.

---

## 4. Tier 1 Code Re-Architecture (Mask-Aware Realism)
When calculating structural statistics, the image matrix must be filtered through the generated binary mask.

### A. Inside-Mask Calculations (Hair Style Match)
* **Target Baseline:** Style Reference Image.
* **Metrics Active:** $GLCM_C$ and $CED$.
* **Matrix Filtering:** Features are calculated strictly using pixel locations where $\text{Mask} == 255$. For GLCM matrices, gray-level co-occurrences are only accumulated if both neighboring pixels fall inside the mask bounds to avoid edge boundaries corrupting texture properties.

### B. Outside-Mask Calculations (Background & Face Quality Preservation)
* **Target Baseline:** Original Image.
* **Metrics Active:** $VBM$, $MS$, and $GLCM_E$.
* **Matrix Filtering:** Features are calculated strictly using pixel locations where $\text{Mask} == 0$. For Fourier Transforms ($MS$) and Laplacians ($VBM$), the pixels inside the hair mask are zeroed out or masked out of the global averaging loop to ensure new hair configurations do not influence background blur or high-frequency static noise assessments.

---

## 5. Verification & Testing Protocol
The testing suite inside `tests/test_structural.py` must expand to cover spatial boundaries:
1. **The Inverse Mask Identity Test:** If an edited image matches the original image perfectly outside the mask zone, the Preservation Zone Score ($VBM, MS, GLCM_E$) must return exactly `100.0%`, regardless of what structural changes happened inside the hair mask.
2. **Backend Interoperability Test:** A unit test must initialize the evaluator with a dummy mock segmenter, verify execution, swap the property to a second segmenter class instance, and re-execute to confirm zero signature breaks in the main orchestration layer.