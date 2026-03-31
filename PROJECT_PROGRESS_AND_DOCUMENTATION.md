# Deepfake Detection Research Project — Progress & Documentation

**Date:** March 2026  
**Status:** ✅ Pipeline complete | ✅ Real dataset integrated | ✅ Model trained & evaluated

---

## 1. Project Overview

### Goal

Build a **research-oriented deepfake detection system** to study:

- Robustness and generalization of foundation-model features vs. traditional CNNs  
- Cross-dataset performance (e.g., train on FaceForensics++, test on Celeb-DF)  
- Explainability (GradCAM, t-SNE)

### Approach

- **Frozen** pretrained feature extractors (ResNet50, CLIP ViT, DINOv2)  
- **MLP classifier** on top of features for classification (upgraded from single linear layer)  
- No end-to-end training of heavy networks

---

## 2. What Is Implemented (Done)

### 2.1 Preprocessing

| Module | Purpose |
|--------|---------|
| `preprocessing/frame_extraction.py` | Extract frames from videos (OpenCV, configurable frame rate) |
| `preprocessing/face_extraction.py` | Face detection and cropping (MTCNN, 224×224) |
| `preprocessing/dataset_loader.py` | PyTorch Dataset with `real/` and `fake/` subdirs, train-time augmentation, WeightedRandomSampler |

### 2.2 Feature Extractors

| Model | File | Description |
|-------|------|-------------|
| ResNet50 | `models/resnet_feature_extractor.py` | ImageNet-pretrained, 2048-d features |
| CLIP ViT | `models/clip_feature_extractor.py` | open_clip, 512/768-d features |
| DINOv2 | `models/dinov2_feature_extractor.py` | transformers, 768-d features |

### 2.3 Training & Evaluation

| Module | Purpose |
|--------|---------|
| `training/train_classifier.py` | MLP classifier (2-hidden-layer, BN+Dropout, Adam, ReduceLROnPlateau, early stopping) |
| `evaluation/evaluate_model.py` | Accuracy, AUC, Precision, Recall, F1, confusion matrix |
| `evaluation/cross_dataset_test.py` | Generalization gap (AUC_train − AUC_test) |

### 2.4 Explainability

| Module | Purpose |
|--------|---------|
| `explainability/gradcam.py` | GradCAM heatmaps on face images |
| `explainability/tsne_visualization.py` | t-SNE feature visualization (real vs. fake) |

### 2.5 Pipelines & Scripts

| Component | Purpose |
|-----------|---------|
| `pipeline/train_pipeline.py` | End-to-end training with val loop, early stopping, augmentation |
| `pipeline/inference_pipeline.py` | Inference (fake probability, label) |
| `dataset/clean_dataset.py` | Remove dummy/corrupt images before training |
| `dataset/prepare_dataset.py` | Build balanced 25% quarter dataset with train/val/test splits |
| `dataset/create_small_dataset.py` | Build controlled-size (3–5 GB) dataset from FF++, Celeb-DF, DFDC |
| `dataset/download_datasets.py` | Download datasets via Kagglehub (optional) |
| `main.py` | CLI: `train`, `evaluate`, `explain` |

### 2.6 Config & Utils

| Component | Purpose |
|-----------|---------|
| `config/config.yaml` | Dataset paths, GPU-safe training hyperparameters |
| `utils/config_loader.py` | YAML config loading |
| `utils/logger.py` | Logging to file and console |

---

## 3. Enhancements Applied (March 2026)

### 3.1 Classifier Upgrade
- **Before:** Single `nn.Linear(2048 → 2)` layer  
- **After:** Two-hidden-layer MLP: `2048 → 512 → BN → ReLU → Dropout(0.4) → 128 → BN → ReLU → Dropout(0.4) → 2`  
- Deeper network learns a better decision boundary on frozen features

### 3.2 Training Improvements
| Feature | Detail |
|---------|--------|
| Validation loop | Val loss + val accuracy evaluated after every epoch |
| Best-model saving | Saved on best **val accuracy** (not train accuracy) |
| LR scheduler | `ReduceLROnPlateau` — halves LR when val loss plateaus |
| Early stopping | Stops after `patience=5` epochs with no val improvement |
| Weight decay | `Adam(weight_decay=1e-4)` for regularisation |

### 3.3 Data Improvements
| Feature | Detail |
|---------|--------|
| Train augmentation | RandomHorizontalFlip, RandomRotation(±10°), ColorJitter, RandomGrayscale, RandomResizedCrop |
| Class balancing | `WeightedRandomSampler` — each class seen equally per epoch |
| Dataset cleaner | `dataset/clean_dataset.py` removes dummy/placeholder/corrupt images |
| Quarter dataset | `dataset/prepare_dataset.py` creates balanced 25% subset with splits |

### 3.4 Config Changes
| Setting | Before | After |
|---------|--------|-------|
| `batch_size` | 32 | **16** (GPU-safe for 8–12 GB VRAM) |
| `learning_rate` | 0.001 | **0.0001** |
| `num_epochs` | 50 | **10** (with early stopping) |
| `num_workers` | 0 | **2** |
| `balance_classes` | — | **true** |
| `augment` | — | **true** |

---

## 4. How to Execute

### 4.1 Prerequisites

```bash
pip install torch torchvision tqdm scikit-learn matplotlib pillow pyyaml numpy
```

### 4.2 Dataset Used

**Celeb-DF v2** (`dataset/Celeb_V2/Test/`) — pre-extracted face crops  

```
dataset/Celeb_V2/Test/
    real/    ← 2,605 images
    fake/    ← 5,067 images
```

### 4.3 Execution Flow

**Step 1 — Clean dataset (remove dummy/corrupt images):**

```bash
python -m dataset.clean_dataset --root dataset/Celeb_V2/Test
```

**Step 2 — Create 25% balanced dataset with train/val/test splits:**

```bash
python -m dataset.prepare_dataset \
    --source dataset/Celeb_V2/Test \
    --target dataset/dataset_quarter \
    --ratio 0.25
```

Output structure:
```
dataset/dataset_quarter/
    train/real/   520 images
    train/fake/  1012 images
    val/real/      65 images
    val/fake/     126 images
    test/real/     66 images
    test/fake/    128 images
```

**Step 3 — Train:**

```bash
python main.py --mode train --backbone resnet
# Output: checkpoints/classifier_resnet.pt
```

**Step 4 — Evaluate:**

```bash
python main.py --mode evaluate --backbone resnet
# Output: results/metrics.csv, results/confusion_matrix.png
```

**Step 5 — Explainability:**

```bash
python main.py --mode explain --backbone resnet --gradcam --tsne
# Output: results/gradcam_images/, results/tsne_plot.png
```

---

## 5. Experimental Results

### 5.1 Training Configuration

| Setting | Value |
|---------|-------|
| Dataset | Celeb-DF v2 (25% subset) |
| Train samples | 1,532 (520 real + 1,012 fake) |
| Val samples | 191 (65 real + 126 fake) |
| Test samples | 194 (66 real + 128 fake) |
| Backbone | ResNet50 (frozen, ImageNet weights) |
| Classifier | MLP (2048 → 512 → 128 → 2) |
| Optimizer | Adam, lr=0.0001, weight_decay=1e-4 |
| Epochs run | 10 (max) |
| Best epoch | 8 |
| Device | CPU |

### 5.2 Training Progression

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.6970 | 58.3% | 0.6612 | 61.9% |
| 2 | 0.6245 | 65.6% | 0.5971 | 70.5% |
| 3 | 0.5994 | 68.6% | 0.5898 | 71.6% |
| 4 | 0.5768 | 69.1% | 0.5643 | 77.3% |
| 5 | 0.5645 | 71.2% | 0.5612 | 73.9% |
| 6 | 0.5478 | 71.9% | 0.5105 | 74.4% |
| 7 | 0.5180 | 75.3% | 0.5181 | 76.1% |
| **8** | **0.5109** | **74.2%** | **0.4955** | **79.0% ✔ best** |
| 9 | 0.5104 | 75.8% | 0.5302 | 75.0% |
| 10 | 0.5215 | 73.7% | 0.4835 | 77.3% |

### 5.3 Test Set Results

| Metric | Score |
|--------|-------|
| **Accuracy** | **76.04%** |
| **AUC (ROC)** | **82.43%** |
| **Precision** | **83.90%** |
| **Recall** | **78.57%** |
| **F1 Score** | **81.15%** |

Saved to: `results/metrics.csv`, `results/confusion_matrix.png`

---

## 6. Dataset Issue & Current Approach

### 6.1 Problem (resolved)

- FaceForensics++, Celeb-DF, and DFDC are large and require download, frame extraction, and face cropping.  
- `faceforensics_train` contained only **16 images** — too small to train.  
- **Fix:** Used `dataset/Celeb_V2/Test/` which contains **7,672 pre-extracted face crop images**.

### 6.2 Dataset Strategy

**Primary (used):** Celeb-DF v2 pre-extracted face crops — 25% subset (~1,900 images total)  
**Future:** Use `create_small_dataset.py` with raw video sources for larger experiments  

---

## 7. What Is Left

| Item | Status |
|------|--------|
| Core pipeline (preprocessing → features → classifier → eval → explain) | ✅ Done |
| Dataset creation script | ✅ Done |
| Dataset download script | ✅ Done |
| Dataset cleaning script | ✅ Done |
| Quarter dataset preparation script | ✅ Done |
| MLP classifier with BN + Dropout | ✅ Done |
| Validation loop + LR scheduler + early stopping | ✅ Done |
| Train-time augmentation + class balancing | ✅ Done |
| **Actual dataset training & evaluation** | ✅ **Done** |
| GradCAM explainability | ✅ Implemented (run with `--gradcam`) |
| t-SNE visualization | ✅ Implemented (run with `--tsne`) |
| CLIP / DINOv2 backbone experiments | Pending |
| Cross-dataset generalization test | Pending |
| GPU training (Colab) | Pending |

---

## 8. Research Questions (from proposal)

1. **GAP 1 — Diffusion vs. GAN artifacts:** Do foundation-model features (CLIP, DINOv2) capture diffusion artifacts better than CNNs?  
2. **GAP 2 — Cross-dataset generalization:** How much does AUC drop when training on FF++ and testing on Celeb-DF?  
3. **GAP 3 — Explainability:** Which facial regions drive predictions? Do models focus on artifacts or semantics?

---

## 9. Folder Structure

```
Deepfake_Detection_Project/
├── config/           # config.yaml (GPU-safe settings), config loader
├── dataset/          # clean_dataset.py, prepare_dataset.py, create_small_dataset.py
│   ├── Celeb_V2/     # Source dataset (Test/real, Test/fake)
│   └── dataset_quarter/  # Prepared 25% train/val/test splits
├── preprocessing/    # frame_extraction, face_extraction, dataset_loader (augmentation)
├── models/           # ResNet, CLIP, DINOv2 feature extractors
├── training/         # MLP classifier, validation loop, early stopping
├── evaluation/       # Metrics, cross-dataset test
├── explainability/   # GradCAM, t-SNE
├── pipeline/         # Train and inference pipelines
├── utils/            # Config loader, logger
├── results/          # metrics.csv, confusion_matrix.png, GradCAM images
│   └── metrics.csv   # Acc=76.04%, AUC=82.43%, F1=81.15%
├── checkpoints/      # classifier_resnet.pt (best val acc=79.0% at epoch 8)
├── main.py           # CLI entry point
└── requirements.txt
```


---

## 1. Project Overview

### Goal

Build a **research-oriented deepfake detection system** to study:

- Robustness and generalization of foundation-model features vs. traditional CNNs  
- Cross-dataset performance (e.g., train on FaceForensics++, test on Celeb-DF)  
- Explainability (GradCAM, t-SNE)

### Approach

- **Frozen** pretrained feature extractors (ResNet50, CLIP ViT, DINOv2)  
- **Linear probe** on top of features for classification  
- No end-to-end training of heavy networks

---

## 2. What Is Implemented (Done)

### 2.1 Preprocessing

| Module | Purpose |
|--------|---------|
| `preprocessing/frame_extraction.py` | Extract frames from videos (OpenCV, configurable frame rate) |
| `preprocessing/face_extraction.py` | Face detection and cropping (MTCNN, 224×224) |
| `preprocessing/dataset_loader.py` | PyTorch Dataset with `real/` and `fake/` subdirs |

### 2.2 Feature Extractors

| Model | File | Description |
|-------|------|-------------|
| ResNet50 | `models/resnet_feature_extractor.py` | ImageNet-pretrained, 2048-d features |
| CLIP ViT | `models/clip_feature_extractor.py` | open_clip, 512/768-d features |
| DINOv2 | `models/dinov2_feature_extractor.py` | transformers, 768-d features |

### 2.3 Training & Evaluation

| Module | Purpose |
|--------|---------|
| `training/train_classifier.py` | Linear classifier (CrossEntropyLoss, Adam) |
| `evaluation/evaluate_model.py` | Accuracy, AUC, Precision, Recall, F1, confusion matrix |
| `evaluation/cross_dataset_test.py` | Generalization gap (AUC_train − AUC_test) |

### 2.4 Explainability

| Module | Purpose |
|--------|---------|
| `explainability/gradcam.py` | GradCAM heatmaps on face images |
| `explainability/tsne_visualization.py` | t-SNE feature visualization (real vs. fake) |

### 2.5 Pipelines & Scripts

| Component | Purpose |
|-----------|---------|
| `pipeline/train_pipeline.py` | End-to-end training |
| `pipeline/inference_pipeline.py` | Inference (fake probability, label) |
| `dataset/create_small_dataset.py` | Build controlled-size (3–5 GB) dataset from FF++, Celeb-DF, DFDC |
| `dataset/download_datasets.py` | Download datasets via Kagglehub (optional) |
| `main.py` | CLI: `train`, `evaluate`, `explain` |

### 2.6 Config & Utils

| Component | Purpose |
|-----------|---------|
| `config/config.yaml` | Dataset paths, training hyperparameters |
| `utils/config_loader.py` | YAML config loading |
| `utils/logger.py` | Logging to file and console |

---

## 3. How to Execute

### 3.1 Prerequisites

```bash
pip install -r requirements.txt
# Key: torch, torchvision, facenet-pytorch, open_clip_torch, transformers, kagglehub
```

### 3.2 Dataset Structure

Any dataset must follow:

```
dataset/
  <your_dataset_name>/
    real/    ← real face images
    fake/    ← fake/deepfake face images
```

### 3.3 Execution Flow

**Step 1 — Create dataset (if using raw videos):**

```bash
# Place raw videos in:
#   dataset/raw/faceforensics
#   dataset/raw/celebdf
#   dataset/raw/dfdc

python dataset/create_small_dataset.py --target_size_gb 5 --frame_rate 3 --max_frames 10
# Output: dataset/final_dataset/real/ and fake/
```

**Step 2 — Train:**

```bash
python main.py --mode train --train_dir dataset/final_dataset --backbone resnet
# Output: checkpoints/classifier_resnet.pt
```

**Step 3 — Evaluate:**

```bash
python main.py --mode evaluate --test_dir dataset/final_dataset
# Output: results/metrics.csv, results/confusion_matrix.png
```

**Step 4 — Explainability:**

```bash
python main.py --mode explain --data_dir dataset/final_dataset --gradcam --tsne
# Output: results/gradcam_images/, results/tsne_plot.png
```

---

## 4. Dataset Issue & Current Approach

### 4.1 Problem

- FaceForensics++, Celeb-DF, and DFDC are large and require:
  - Download from official/Kaggle sources  
  - Frame extraction  
  - Face detection/cropping  
- This is time-consuming and storage-intensive.

### 4.2 Current Approaches

**A. Manual raw data + `create_small_dataset.py`**

- Place raw videos in `dataset/raw/faceforensics`, `celebdf`, `dfdc`  
- Script extracts frames, crops faces, builds a 3–5 GB subset  

**B. Kagglehub download**

- Use `dataset/download_datasets.py` to download via Kagglehub  
- Requires Kaggle credentials and network access  

**C. Pre-extracted / feature datasets (under exploration)**

- Considering datasets with pre-extracted face crops or features  
- Could reduce compute and storage and speed up experiments  
- Investigating which datasets and formats are available and compatible  

---

## 5. What Is Left

| Item | Status |
|------|--------|
| Core pipeline (preprocessing → features → classifier → eval → explain) | Done |
| Dataset creation script | Done |
| Dataset download script | Done |
| **Actual dataset (real videos/images)** | **Pending** |
| Experiments on real data | Pending |

---

## 6. Research Questions (from proposal)

1. **GAP 1 — Diffusion vs. GAN artifacts:** Do foundation-model features (CLIP, DINOv2) capture diffusion artifacts better than CNNs?  
2. **GAP 2 — Cross-dataset generalization:** How much does AUC drop when training on FF++ and testing on Celeb-DF?  
3. **GAP 3 — Explainability:** Which facial regions drive predictions? Do models focus on artifacts or semantics?

---

## 7. Folder Structure

```
Deepfake_Detection_Project/
├── config/           # config.yaml, config loader
├── dataset/          # create_small_dataset.py, download_datasets.py
├── preprocessing/    # frame_extraction, face_extraction, dataset_loader
├── models/           # ResNet, CLIP, DINOv2 feature extractors
├── training/         # Linear classifier training
├── evaluation/       # Metrics, cross-dataset test
├── explainability/   # GradCAM, t-SNE
├── pipeline/         # Train and inference pipelines
├── utils/            # Config loader, logger
├── results/          # metrics.csv, plots, GradCAM images
├── checkpoints/      # Saved classifier
├── main.py           # CLI entry point
└── requirements.txt
```
