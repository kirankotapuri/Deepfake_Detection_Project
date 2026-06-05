# Cross-Dataset Generalization in Deepfake Detection

> **Research Paper:** *Cross-Dataset Generalization in Deepfake Detection: A Systematic Evaluation of ResNet50, CLIP, and DINOv2 with Weighted Fusion and Diffusion Robustness Analysis*
> N.Y.S. Surya Prabha, K. Kiran Kumar, H.T. Sumanth Raj — SRM University AP
> Guided by: Dr. Ajay Dilip Kumar Marapatla

---

## Overview

This project implements and evaluates a **multi-backbone deepfake detection system** under a **linear probing** paradigm. Three frozen pre-trained backbones (ResNet50, CLIP ViT-B/32, DINOv2) are compared across twelve cross-dataset transfer scenarios. A **weighted two-backbone fusion model** (ResNet50 + CLIP) is developed and evaluated, followed by a systematic **diffusion robustness analysis** testing GAN-trained detectors against Stable Diffusion-generated faces. GradCAM and t-SNE **explainability** analyses complete the study.

### Research Questions
| # | Question |
|---|----------|
| RQ1 | How well do detectors trained on one benchmark transfer to an unseen dataset? |
| RQ2 | Which backbone (ResNet50, CLIP, DINOv2) generalizes most reliably, and which is most stable? |
| RQ3 | Can a learned weighted fusion of ResNet50 + CLIP outperform either backbone individually in terms of balanced detection (F1)? |
| RQ4 | Do GAN-trained detectors hold up against Stable Diffusion-generated faces? |

---

## System Architecture

```
Raw Video / Image Input
        ↓
  Frame Extraction  (1 fps, max 100/video)
        ↓
  MTCNN Face Detection  (20px margin)
        ↓
  Resize 224×224  +  ImageNet Normalisation
        ↓
  ┌──────────────┬──────────────┬───────────────┐
  │  ResNet50    │  CLIP        │  DINOv2       │
  │  (2048-d)    │  ViT-B/32    │  ViT-B/14     │
  │              │  (512-d)     │  (768-d)      │
  └──────┬───────┴──────┬───────┴───────┬───────┘
         │  Linear Probing (frozen backbone)     │
         │  MLP: D→512→128→2  (BN+ReLU+Drop)    │
         ↓                                       ↓
  Individual Predictions        Weighted Fusion Branch
                                  CLIP(64.5%) + ResNet50(35.5%)
                                  → fused → MLP → Prediction
        ↓
  Explainability
     ├── GradCAM  (spatial attention heatmaps)
     └── t-SNE    (feature-space cluster visualisation)
```

---

## Key Results

### Experiment 1 — Cross-Dataset Baseline (12 Scenarios)

| Train | Test | Backbone | Accuracy | Precision | Recall | F1 | AUC |
|-------|------|----------|----------|-----------|--------|----|-----|
| CelebDF | DFDC | ResNet50 | 0.5176 | 0.5464 | 0.1616 | 0.2494 | 0.5361 |
| CelebDF | DFDC | CLIP | 0.5101 | 0.6364 | 0.0285 | 0.0545 | 0.5335 |
| CelebDF | DFDC | DINOv2 | 0.5232 | 0.6188 | 0.1006 | 0.1731 | 0.5123 |
| CelebDF | UADFV | ResNet50 | 0.5729 | 0.4677 | 0.7838 | 0.5859 | 0.7055 |
| **CelebDF** | **UADFV** | **CLIP ★** | **0.7708** | **0.8000** | 0.5405 | **0.6452** | **0.8310** |
| CelebDF | UADFV | DINOv2 | 0.5000 | 0.4068 | 0.6486 | 0.5000 | 0.5703 |
| UADFV | CelebDF | ResNet50 | 0.5312 | 0.7411 | 0.4392 | 0.5515 | 0.6314 |
| UADFV | CelebDF | CLIP | 0.3819 | 0.8235 | 0.0741 | 0.1359 | 0.6141 |
| UADFV | CelebDF | DINOv2 | 0.3889 | 0.6585 | 0.1429 | 0.2348 | 0.5178 |
| UADFV | DFDC | ResNet50 | 0.5297 | 0.5612 | 0.2378 | 0.3340 | 0.5643 |
| UADFV | DFDC | CLIP | 0.5091 | 0.6923 | 0.0183 | 0.0356 | 0.5510 |
| UADFV | DFDC | DINOv2 | 0.5040 | 0.5000 | 0.1728 | 0.2568 | 0.5344 |

### Backbone Stability Summary

| Backbone | Mean Acc | Std Dev | Best | Worst | Recommendation |
|----------|----------|---------|------|-------|----------------|
| ResNet50 | 0.524 | **±0.028** | 0.573 | 0.482 | Unknown target distribution |
| CLIP | 0.524 | ±0.142 | **0.771** | 0.382 | Known target distribution |
| DINOv2 | 0.476 | ±0.057 | 0.523 | 0.389 | Not recommended |

### Experiment 2 — Weighted Two-Backbone Fusion

| Train | Test | Best Baseline | BL F1 | 2-BB F1 | 3-BB F1 | BL AUC | 2-BB AUC |
|-------|------|---------------|-------|---------|---------|--------|----------|
| CelebDF | UADFV | CLIP (0.7708) | 0.6452 | **0.7033 (+8.7%)** | 0.5570 | 0.8310 | 0.8254 |
| CelebDF | DFDC | DINOv2 | 0.1731 | 0.1559 | 0.1820 | 0.5123 | 0.5183 |
| UADFV | CelebDF | ResNet50 | 0.5515 | 0.4678 | 0.4565 | 0.6314 | 0.6179 |
| UADFV | DFDC | ResNet50 | 0.3340 | 0.3190 | 0.3090 | 0.5643 | 0.5622 |

**Learned fusion weights** are consistent across training datasets: CLIP ≈ 64–67%, ResNet50 ≈ 33–36%.

### Experiment 3 — Diffusion Robustness

| Train | Test | Backbone | Accuracy | F1 | AUC | vs GAN Accuracy |
|-------|------|----------|----------|----|-----|-----------------|
| **CelebDF** | Stable Diffusion | **ResNet50 ★** | **0.5885** | 0.5683 | 0.6260 | 0.5729 → 0.5885 |
| CelebDF | Stable Diffusion | CLIP | 0.4271 | 0.0179 | 0.1579 | **0.7708 → 0.4271 (−34%)** |
| CelebDF | Stable Diffusion | DINOv2 | 0.3750 | 0.0625 | 0.3063 | 0.5000 → 0.3750 |
| UADFV | Stable Diffusion | ResNet50 | 0.4479 | 0.2319 | 0.3482 | 0.5312 → 0.4479 |
| UADFV | Stable Diffusion | CLIP | 0.5000 | 0.0000 | 0.3069 | **Complete failure (all real)** |
| UADFV | Stable Diffusion | DINOv2 | 0.4531 | 0.0367 | 0.2983 | — |

> CLIP's AUC of 0.1579 is **below random chance** — it actively mispredicts diffusion fakes as real.

---

## Datasets

| Dataset | Type | Videos | Frames Used | Role | Ref |
|---------|------|--------|-------------|------|-----|
| [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) | GAN face-swap | 5,639 | ~31,250 | Primary training | Li et al., CVPR 2020 |
| [UADFV](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi) | GAN face-swap | 49 | ~1,225 | Train + Test | Li et al., WIFS 2018 |
| [DFDC](https://ai.meta.com/datasets/dfdc/) | Multi-method GAN | 3,426+ | ~18,500 | Test only | Dolhansky et al., 2020 |
| [Stable Diffusion](https://huggingface.co/datasets/FDHFLWR/stable-diffusion-face-dataset) | Diffusion (text-to-img) | N/A | 200–500 | Robustness test | Rombach et al., CVPR 2022 |

> Datasets are **not included** (size: ~46 GB). Download from the links above.
> Place raw data under `dataset/` following the structure expected by `dataset/standardize_datasets.py`.

---

## Project Structure

```
Deepfake_Detection_Project/
│
├── main.py                              # Primary entry point — all pipeline modes
├── run_diffusion_experiment.py          # Standalone diffusion robustness runner
├── run_attention_fusion_experiment.py   # Standalone weighted fusion experiment runner
├── run_explainability.py                # Standalone GradCAM + t-SNE runner
├── resume_cross_eval.py                 # Resume/patch incomplete cross-eval runs
├── master_analysis.py                   # Master suite: graphs + tables + explainability
├── generate_table.py                    # Generate formatted result tables
├── monitor_progress.py                  # Live progress monitoring utility
│
├── config/
│   ├── config.yaml                      # Dataset paths, training hyperparameters
│   └── config.py                        # Config dataclass definitions
│
├── dataset/
│   ├── standardize_datasets.py          # Phase 1A: flatten raw datasets → real/fake
│   ├── prepare_all_datasets.py          # Phase 1B: create 1/4 subsets + splits
│   ├── prepare_dataset.py               # Core 70/15/15 split logic
│   ├── prepare_diffusion_dataset.py     # Prepare Stable Diffusion face dataset
│   ├── create_small_dataset.py          # Utility: create tiny debug subsets
│   ├── clean_dataset.py                 # Remove corrupt/duplicate images
│   └── download_datasets.py             # Dataset download helpers
│
├── models/
│   ├── resnet_feature_extractor.py      # ResNet50 frozen backbone (2048-d)
│   ├── clip_feature_extractor.py        # CLIP ViT-B/32 frozen backbone (512-d)
│   ├── dinov2_feature_extractor.py      # DINOv2 ViT-B/14 frozen backbone (768-d)
│   ├── attention_fusion_classifier.py   # Fusion models:
│   │                                    #   ImprovedFusionClassifier (ResNet+CLIP)
│   │                                    #   ImprovedFusionWithDINO (3-backbone)
│   │                                    #   MultiBackboneAttentionFusionClassifier
│   ├── cnn_model.py                     # Standalone CNN model
│   └── resnet_model.py                  # ResNet end-to-end model
│
├── pipeline/
│   ├── train_pipeline.py                # Single-backbone feature extraction + training
│   ├── cross_dataset_pipeline.py        # 12-scenario cross-dataset experiment
│   ├── fusion_cross_dataset_pipeline.py # Weighted fusion experiment (2-BB and 3-BB)
│   ├── diffusion_experiment.py          # Diffusion robustness pipeline
│   ├── inference_pipeline.py            # Single-image inference
│   └── pipeline.py                      # Base pipeline utilities
│
├── training/
│   ├── train_classifier.py              # LinearClassifier + MLP training loop
│   ├── train_fusion_classifier.py       # Fusion model training with OneCycleLR
│   ├── train_model.py                   # End-to-end model training utilities
│   └── loss_functions.py               # Custom loss functions
│
├── evaluation/
│   ├── evaluate_model.py                # Accuracy, AUC, F1, confusion matrix, ROC
│   ├── cross_dataset_test.py            # Cross-dataset evaluation utilities
│   └── metrics.py                       # Metric computation helpers
│
├── explainability/
│   ├── gradcam.py                       # GradCAM heatmaps (ResNet50 + attention layers)
│   └── tsne_visualization.py            # t-SNE feature-space visualisation
│
├── preprocessing/
│   ├── dataset_loader.py                # PyTorch Dataset + ImageNet transforms
│   ├── frame_extraction.py              # Video → frames at 1 fps
│   ├── face_extraction.py               # MTCNN face detection + crop
│   └── preprocess.py                    # General preprocessing utilities
│
├── scripts/
│   ├── generate_graphs.py               # Publication-ready graphs (heatmaps, bar charts, ROC)
│   └── explainability_suite.py          # Batch GradCAM + t-SNE generation
│
├── utils/
│   ├── config_loader.py                 # YAML config loader
│   ├── helpers.py                       # Shared utilities
│   └── logger.py                        # Logging setup
│
├── checkpoints/
│   ├── classifier_resnet.pt             # Single-dataset ResNet50 checkpoint
│   ├── cross/                           # 6 cross-dataset backbone checkpoints
│   │   ├── celebdf_{resnet,clip,dinov2}.pt
│   │   └── uadfv_{resnet,clip,dinov2}.pt
│   ├── diffusion/                       # 3 diffusion-experiment checkpoints
│   │   └── diffusion_{resnet,clip,dinov2}.pt
│   └── fusion/cross/                    # 7 fusion model checkpoints
│       ├── celebdf_{two,three,attention,weighted}_backbone.pt
│       └── uadfv_{two,three,attention}_backbone.pt
│
├── results/
│   ├── cross_dataset_results.csv        # Raw 12-scenario results
│   ├── cross_dataset_table.txt          # Formatted cross-dataset table
│   ├── metrics.csv                      # General metrics log
│   ├── cm_*.png                         # Confusion matrices (12 scenarios)
│   ├── tsne_{ResNet50,CLIP,DINOv2}.png  # t-SNE plots
│   ├── diffusion_results/               # All diffusion experiment outputs
│   │   ├── diffusion_results.csv
│   │   ├── cm_exp*.png
│   │   ├── roc_exp*.png
│   │   └── report_exp*.txt
│   ├── fusion/                          # All fusion experiment outputs
│   │   ├── fusion_cross_dataset_results.csv
│   │   ├── fusion_weights_{two,three}_backbone.csv
│   │   ├── baseline_vs_fusion_comparison*.csv / .txt
│   │   └── cm_fusion_*.png
│   └── gradcam_images/                  # GradCAM heatmaps
│
├── notebooks/                           # Jupyter notebooks
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake-detection-project.git
cd deepfake-detection-project

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

> PyTorch with CUDA is recommended. The project runs on CPU but is significantly slower.

---

## Usage

### Step 1 — Prepare Datasets

```bash
# Standardize raw datasets + create 1/4 subsets (70/15/15 splits)
python main.py --mode prepare
```

For the diffusion experiment, also prepare the Stable Diffusion dataset:
```bash
python dataset/prepare_diffusion_dataset.py
```

### Step 2 — Cross-Dataset Generalization Experiment (12 scenarios)

```bash
python main.py --mode cross_eval
```

Trains 6 models (2 training datasets × 3 backbones), evaluates on all held-out test
datasets, and writes results to `results/cross_dataset_table.txt` and
`results/cross_dataset_results.csv`.

If a run was interrupted, resume without retraining completed scenarios:
```bash
python resume_cross_eval.py
```

### Step 3 — Weighted Fusion Experiment

```bash
# Two-backbone fusion (ResNet50 + CLIP) — recommended
python run_attention_fusion_experiment.py --model-type two_backbone

# Three-backbone fusion (ResNet50 + CLIP + DINOv2) — for comparison
python run_attention_fusion_experiment.py --model-type three_backbone
```

Outputs go to `results/fusion/` and `checkpoints/fusion/cross/`.

### Step 4 — Diffusion Robustness Analysis

```bash
python run_diffusion_experiment.py

# Skip Experiment 2 (reverse: SD-trained → CelebDF)
python run_diffusion_experiment.py --skip-exp2
```

Outputs (confusion matrices, ROC curves, classification reports, GradCAM on diffusion
faces) go to `results/diffusion_results/`.

### Step 5 — Explainability (GradCAM + t-SNE)

```bash
python run_explainability.py
```

Produces heatmaps in `results/gradcam_images/` and t-SNE plots in `results/`.

### Step 6 — Publication Graphs & Master Analysis

```bash
# Generate all publication-ready graphs
python master_analysis.py --mode graphs

# Generate explainability visualisations
python master_analysis.py --mode explainability

# Run everything
python master_analysis.py --mode all
```

### Training / Evaluating a Single Backbone

```bash
# Train one backbone
python main.py --mode train --backbone clip \
    --train-dir dataset/CelebDF_quarter/train \
    --val-dir   dataset/CelebDF_quarter/val

# Evaluate a saved checkpoint
python main.py --mode evaluate --backbone clip \
    --test-dir dataset/UADFV_quarter/test \
    --checkpoint checkpoints/cross/celebdf_clip.pt

# Full pipeline (prepare → train → evaluate → explain)
python main.py --mode full_pipeline
```

---

## Model Architecture Details

### Linear Probing MLP Classifier

All three backbones share the same classifier head (only the input dimension varies):

```
D → Linear(512) → BN → ReLU → Dropout(0.4)
  → Linear(128) → BN → ReLU → Dropout(0.4)
  → Linear(2)   [real / fake]
```

where `D` = 2048 (ResNet50), 512 (CLIP), 768 (DINOv2).

### Weighted Two-Backbone Fusion

```
f_fused = w_C · proj_C(norm(f_CLIP)) + w_R · proj_R(norm(f_ResNet))
```

- `w_C = σ(α)`, `w_R = 1 − w_C`, where `α` is a single learnable scalar
- Initialized: `α = σ⁻¹(0.65) ≈ 0.619` (nudged toward CLIP from baseline knowledge)
- `proj_C: ℝ⁵¹² → ℝ²⁵⁶` and `proj_R: ℝ²⁰⁴⁸ → ℝ²⁵⁶` (each: Linear → BN → ReLU)
- **Learned weights (CelebDF training):** CLIP = 64.48%, ResNet50 = 35.52%
- **Learned weights (UADFV training):** CLIP = 66.59%, ResNet50 = 33.41%

---

## Training Configuration

| Hyperparameter | Individual Backbones | Fusion Model |
|----------------|---------------------|--------------|
| Optimizer | Adam (β₁=0.9, β₂=0.999) | Adam |
| Learning rate | 1×10⁻⁴ | proj: 5×10⁻⁵, α: 1×10⁻³, head: 1×10⁻⁴ |
| Batch size | 16 | 16 |
| Max epochs | 10 | 25 |
| Early stopping patience | 5 | 8 |
| Dropout | 0.4 | 0.3 |
| LR scheduler | ReduceLROnPlateau | OneCycleLR |
| Gradient clipping | — | max_norm = 1.0 |
| Loss | CrossEntropyLoss | CrossEntropyLoss |

### Preprocessing & Augmentation (training only)

```
RandomHorizontalFlip → RandomRotation(±10°) → ColorJitter
→ RandomGrayscale(p=0.05) → RandomResizedCrop(scale=0.85–1.0)
→ Resize(224×224) → ToTensor → Normalize(ImageNet stats)
```

---

## Evaluation Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Confidence in fake predictions |
| Recall | TP/(TP+FN) | Proportion of fakes detected |
| F1-Score | 2·(P·R)/(P+R) | **Primary metric for fusion** |
| AUC-ROC | Area under ROC curve | Threshold-independent discrimination |
| Std Dev | σ(Acc across scenarios) | Deployment stability |

---

## Explainability Findings

**GradCAM:** ResNet50 concentrates on localized artifact-rich regions (skin-texture
boundaries, jawline, periocular areas) — explaining its deployment stability. CLIP
attends to the full face holistically — enabling peak performance when distributions
match, but failing to localize artifacts under distribution shift. DINOv2 diffuses
attention into background regions with low signal-to-noise ratio.

**t-SNE:** Feature-space cluster separability directly predicts generalization:
- ResNet50 → tight, well-separated clusters ↔ ±2.8% variance
- CLIP → moderate separation with spread ↔ ±14.2% variance  
- DINOv2 → heavy cluster overlap ↔ consistently poor performance

> **Practical insight:** t-SNE on a small labelled sample can serve as a cheap
> pre-deployment diagnostic before full cross-dataset evaluation.

---

## Practical Deployment Guidance

| Scenario | Recommended Backbone |
|----------|---------------------|
| Target distribution is **known** | CLIP (77.08% acc, AUC 0.8310) |
| Target distribution is **unknown** | ResNet50 (±2.8% std dev) |
| Both false positives & missed detections matter | Two-backbone weighted fusion (F1 0.7033) |
| Against **diffusion-generated** content | Do **not** use GAN-only trained models |
| General use | Avoid DINOv2 (mean 47.6% across 12 scenarios) |

---

## Requirements

Key dependencies (see [requirements.txt](requirements.txt) for pinned versions):

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `transformers` | DINOv2 feature extractor |
| `open_clip_torch` | CLIP ViT-B/32 feature extractor |
| `facenet-pytorch` | MTCNN face detection |
| `scikit-learn` | Metrics, t-SNE |
| `matplotlib`, `seaborn` | Visualisation |
| `pandas`, `numpy` | Data handling |
| `opencv-python` | Frame extraction |
| `Pillow` | Image loading |
| `jupyterlab` | Notebook interface |


---

## Notes

- All backbone weights are **frozen**; only classifiers / fusion heads are trained.
- The project uses **1/4 of each dataset** due to computational constraints.
- Datasets, checkpoints, and large result files are excluded via `.gitignore`.
- The `HF_HUB_OFFLINE=1` environment variable is set in fusion pipelines to use
  cached HuggingFace weights on networks with SSL restrictions.
- Attention-based fusion was prototyped but replaced with scalar-weighted fusion
  because UADFV (49 videos) lacks sufficient data for attention weights to converge.
