# Deepfake Detection Project

## Overview

This project implements a multi-backbone deepfake detection system that investigates
**cross-dataset generalization** — measuring how well models trained on one dataset
detect deepfakes in unseen datasets.

### Research Questions
- Which backbone generalizes better across datasets: **ResNet50**, **CLIP**, or **DINOv2**?
- Do GAN-trained detectors transfer to **diffusion-generated** deepfakes?
- Which facial regions drive predictions? (**Explainability via GradCAM**)

---

## System Architecture

```
Image Input
      ↓
Dataset Loader
      ↓
Feature Extraction  (ResNet50 | CLIP | DINOv2)
      ↓
Linear Classifier
      ↓
Prediction
      ↓
Explainability
   ├── GradCAM
   └── t-SNE
```

---

## Datasets Used

| Dataset | Type | Role |
|---------|------|------|
| [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) | GAN deepfakes (images) | Training |
| [DFDC](https://ai.meta.com/datasets/dfdc/) | GAN deepfakes (video→frames) | Test only |
| [UADFV](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi) | GAN deepfakes (video→frames) | Test only |

> **Note:** Datasets are not included due to size (~46 GB). Download from the links above.

---

## Project Structure

```
Deepfake_Detection_Project/
├── main.py                          # Main entry point (all modes)
├── config/
│   └── config.yaml                  # Dataset paths, training hyper-parameters
├── dataset/
│   ├── standardize_datasets.py      # Phase 1A: flatten raw datasets to real/fake
│   ├── prepare_all_datasets.py      # Phase 1B: create 1/4 subsets + train/val/test splits
│   └── prepare_dataset.py           # Core split logic
├── models/
│   ├── resnet_feature_extractor.py  # ResNet50 backbone
│   ├── clip_feature_extractor.py    # CLIP ViT-B/32 backbone
│   └── dinov2_feature_extractor.py  # DINOv2 backbone
├── pipeline/
│   ├── train_pipeline.py            # Single-backbone training
│   ├── cross_dataset_pipeline.py    # Phase 4: full generalization experiment
│   ├── diffusion_experiment.py      # Phase 5: diffusion robustness
│   └── inference_pipeline.py        # Single-image inference
├── training/
│   └── train_classifier.py          # MLP classifier + training loop
├── evaluation/
│   └── evaluate_model.py            # Accuracy, AUC, F1, confusion matrix
├── explainability/
│   ├── gradcam.py                   # GradCAM heatmaps
│   └── tsne_visualization.py        # t-SNE feature visualization
├── preprocessing/
│   ├── dataset_loader.py            # PyTorch Dataset + transforms
│   ├── frame_extraction.py          # Video → frames (UADFV)
│   └── face_extraction.py           # MTCNN face detection
├── utils/
│   ├── config_loader.py
│   └── logger.py
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/yourusername/deepfake-detection-project.git
cd deepfake-detection-project

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

---

## Usage

### Step 1 — Prepare Datasets
```bash
python main.py --mode prepare
```
Standardizes dataset structure and creates 1/4 subsets (70/15/15 splits).

### Step 2 — Cross-Dataset Generalization Experiment
```bash
python main.py --mode cross_eval
```
Trains 6 models (2 training scenarios × 3 backbones), evaluates on all test datasets,
and saves a comparison table to `results/cross_dataset_table.txt`.

### Step 3 — Diffusion Robustness Test
```bash
python main.py --mode diffusion --diffusion-dir dataset/diffusion_faces
```

### Step 4 — Explainability
```bash
python main.py --mode explain --backbone resnet --gradcam --tsne
```

### All-in-one
```bash
python main.py --mode full_pipeline --diffusion-dir dataset/diffusion_faces
```

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score
- AUC (Area Under ROC Curve)
- Confusion Matrix

---

## Cross-Dataset Results (Example)

| Train Dataset | Test Dataset | Backbone | Accuracy | AUC |
|---|---|---|---|---|
| CelebDF | DFDC | ResNet50 | 0.52 | 0.54 |
| CelebDF | DFDC | CLIP | 0.51 | 0.53 |
| CelebDF | DFDC | DINOv2 | — | — |
| CelebDF | UADFV | ResNet50 | 0.57 | 0.71 |
| CelebDF | UADFV | CLIP | 0.77 | 0.83 |

> Full results saved to `results/cross_dataset_table.txt` after running `--mode cross_eval`.

---

## Requirements

See [requirements.txt](requirements.txt) for the full list. Key dependencies:

- `torch`, `torchvision`
- `transformers` (DINOv2)
- `open_clip_torch` (CLIP)
- `scikit-learn`, `matplotlib`, `pandas`
- `opencv-python` (frame extraction)
- `facenet-pytorch` (MTCNN face detection)

---

## Notes

- Backbone weights are frozen; only the linear classifier is trained.
- All large files (datasets, checkpoints, results) are excluded via `.gitignore`.
- The project uses ~1/4 of each dataset due to GPU memory constraints.
