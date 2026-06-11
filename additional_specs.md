# Unified Diffusion Realism Score (UDRS)

System Design Specification (v2.0)

1. System Objective

The UDRS pipeline is a multi-tiered evaluation framework designed to assess the physical, perceptual, and semantic realism of an image edited via a diffusion model. By integrating structural statistics, deep feature extraction, and cloud-based VLM semantic checks, it provides a holistic percentage score benchmarking the edited image against its unedited baseline.

2. Tiered Evaluation Architecture

The system is divided into three independent but combinable evaluation tiers to allow for modular execution based on available compute.

Tier 1: Structural Realism (RDRS Core)

Validating the low-level physical integrity of the image using the pentagon geometric model. As proven by the RAISE dataset analysis, handcrafted features strongly correlate with subjective human realness ratings.

Features: GLCM Contrast, Canny Edge Density, GLCM Energy, Variance Blur Measure, Mean Spectrum.

Method: Relative calibration against the original image to compute the RDRS area percentage.

Tier 2: Deep Perceptual Realism (RAISE ResNet Module)

Evaluating the global "naturalness" of the image using transfer learning from foundation vision models.

Methodology: The RAISE paper demonstrates that a ResNet-18 backbone, pre-trained on ImageNet, effectively acts as a feature extractor for perceptual realness.

Implementation: Pass both the original and edited images through a frozen, local PyTorch ResNet-18 model. Extract the 512-dimensional feature maps from global average pooling. Compute the Cosine Similarity between the original and edited embeddings to generate a "Deep Perceptual Retention" percentage.

Tier 3: Semantic & Relational Realism (REAL API Module)

Assessing the logical integrity of fine-grained visual attributes and unusual visual relationships.

Methodology: Generates structured prompts to check for visibility, description match, and relationship plausibility.

Implementation:Uses a schema generator to define expected attributes.Pings a cloud API (GPT-4o or Gemini 1.5 Pro) to act as the Visual Question Answering (VQA) engine.

Calculates an Attribute Score ($S_{att}$) based on the ratio of correctly depicted visible attributes to total visible attributes.

Calculates a Relationship Score ($S_{rel}$) based on visibility, realism, and relationship checks between objects.

Tier 4: Visual Style Fidelity (REAL CLIP Module)

Ensuring the diffusion model did not shift a photorealistic image into an illustrative style .

Implementation: A local, lightweight CLIP model fine-tuned on "photo" vs "illustration" classes. It outputs a probability score indicating the likelihood the edited image remains in the "photo" class.

3. Aggregation Engine

The final UDRS score is a weighted aggregation of the tiers.

$Score_{Total} = (w_1 \cdot Structural) + (w_2 \cdot Perceptual) + (w_3 \cdot Semantic) + (w_4 \cdot Style)$

Weights are configurable depending on the specific use-case constraints (e.g., setting $w_3 = 0$ if API access is offline).

4. Updated Project Directory Organization

```text
diffusion_realism_eval/
│
├── data/
│   ├── original/              # Unedited baseline images
│   └── edited/                # Diffusion-edited variants
│
├── src/
│   ├── __init__.py
│   ├── rdrs_core.py           # Tier 1: Structural pentagon math (numpy/scikit-image)
│   ├── raise_perceptual.py    # Tier 2: ResNet-18 deep feature extraction (PyTorch)
│   ├── real_semantic.py       # Tier 3: VQA schemas and Cloud API wrappers (requests/SDK)
│   └── real_style.py          # Tier 4: CLIP zero-shot/fine-tuned style classification
│
├── tests/
│   ├── __init__.py
│   ├── test_structural.py
│   ├── test_perceptual.py
│   └── test_semantic_mock.py  # Mocks API responses to test VQA math without spending credits
│
├── config/
│   └── api_keys.env           # Stores OpenAI/Google API keys
│
├── requirements.txt
└── README.md
```

5. API Fallback & Future-Proofing

The real_semantic.py module is designed with an abstract VQABackend class.

Current State: Instantiates CloudVQABackend targeting standard REST endpoints.

Future State: Can instantiate LocalVQABackend to load weights for models like mPLUG-Owl3 into VRAM once hardware permits.

To get this off the ground, how would you like to handle the weighting between these four modules for the final composite score?