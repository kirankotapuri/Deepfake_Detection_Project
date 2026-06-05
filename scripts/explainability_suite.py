#!/usr/bin/env python3
"""
Comprehensive Explainability Script for Deepfake Detection

Generates:
1. GradCAM heatmaps for all backbones (normal + diffusion)
2. t-SNE visualizations for all backbones (normal + diffusion)
3. Combined comparison plots

Run:
    python -c "from scripts.explainability_suite import *; generate_all_explainability()"
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import seaborn as sns

try:
    from sklearn.manifold import TSNE
except ImportError:
    print("Warning: scikit-learn not found. Install with: pip install scikit-learn")

# ============================================================================
# GRADCAM IMPLEMENTATION
# ============================================================================

def compute_gradcam(model, target_layer, input_tensor, target_class=None):
    """
    Compute GradCAM for visualization.
    
    Args:
        model: Full model (extractor + classifier)
        target_layer: Layer to extract activations from (e.g., conv layer)
        input_tensor: Input image [1, 3, H, W]
        target_class: Class to maximize (None = predicted class)
    
    Returns:
        heatmap: 2D array [H', W'] where each value indicates influence
    """
    model.eval()
    input_tensor.requires_grad_(True)
    
    activations = []
    gradients = []
    
    def save_activation(module, input, output):
        activations.append(output.detach())
    
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)
    
    try:
        # Forward pass
        with torch.enable_grad():
            output = model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        output[0, target_class].backward()
        
        # Compute GradCAM
        acts = activations[0][0]  # [C, H, W]
        grads = gradients[0][0]   # [C, H, W]
        
        # Compute weights
        weights = grads.mean(dim=(1, 2))  # [C]
        
        # Weighted activation
        cam = (weights.view(-1, 1, 1) * acts).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize
        cam_np = cam.cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        
        return cam_np
    
    finally:
        forward_handle.remove()
        backward_handle.remove()


def overlay_gradcam(heatmap, original_image, alpha=0.4):
    """Overlay heatmap on original image."""
    
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
        if original_image.shape[0] == 3:
            original_image = original_image.transpose(1, 2, 0)
    
    # Denormalize if needed
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    else:
        original_image = original_image.astype(np.uint8)
    
    # Resize heatmap to image size
    heatmap_resized = np.array(Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    ).resize((original_image.shape[1], original_image.shape[0])))
    
    # Apply colormap
    heatmap_colored = plt.cm.jet(heatmap_resized / 255.0)[:, :, :3]
    
    # Blend
    overlaid = (1 - alpha) * (original_image / 255.0) + alpha * heatmap_colored
    
    return (np.clip(overlaid, 0, 1) * 255).astype(np.uint8)
  

def generate_gradcam_for_dataset(
    dataset_dir,
    feature_extractor,
    classifier,
    backbone_name,
    output_dir,
    num_samples=10,
    target_layer=None,
):
    """Generate GradCAM visualizations for a dataset."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine extractor + classifier for GradCAM
    class CombinedModel(torch.nn.Module):
        def __init__(self, extractor, classifier):
            super().__init__()
            self.extractor = extractor
            self.classifier = classifier
        
        def forward(self, x):
            # Get features then classify
            feats = self.extractor(x)
            logits = self.classifier(feats)
            return logits
    
    combined = CombinedModel(feature_extractor, classifier)
    combined.eval()
    
    # Determine target layer
    if target_layer is None:
        if hasattr(feature_extractor, 'backbone'):
            target_layer = feature_extractor.backbone[-1]  # Last layer
        else:
            target_layer = feature_extractor  # Use feature extractor
    
    # Load sample images
    dataset_path = Path(dataset_dir)
    image_files = []
    for fake_real in ['fake', 'real']:
        subdir = dataset_path / fake_real
        if subdir.exists():
            image_files.extend(list(subdir.glob('*.jpg')))
            image_files.extend(list(subdir.glob('*.png')))
    
    image_files = image_files[:num_samples]
    
    print(f"  [{backbone_name}] Generating GradCAM ({len(image_files)} images)...")
    
    for idx, img_path in enumerate(tqdm(image_files, desc=f"GradCAM {backbone_name}")):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_arr = np.array(img)
            
            # Preprocess
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            img_tensor = transform(img).unsqueeze(0).to(next(combined.parameters()).device)
            
            # Compute GradCAM
            heatmap = compute_gradcam(combined, target_layer, img_tensor)
            
            # Overlay on original
            overlaid = overlay_gradcam(heatmap, img_arr)
            
            # Save
            output_file = output_dir / f"{idx:04d}_{img_path.stem}_gradcam.jpg"
            Image.fromarray(overlaid).save(output_file)
        
        except Exception as e:
            print(f"    Error processing {img_path}: {e}")
    
    print(f"  ✓ Saved {len(image_files)} GradCAM images to {output_dir}")


# ============================================================================
# t-SNE VISUALIZATION
# ============================================================================

def compute_tsne_embeddings(
    feature_extractor,
    dataset_dir,
    n_components=2,
    perplexity=30,
):
    """Extract features and compute t-SNE embeddings."""
    
    all_features = []
    all_labels = []
    
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    # Create simple dataset
    class SimpleImageDataset:
        def __init__(self, root_dir):
            self.images = []
            self.labels = []
            
            for label, cls_name in enumerate(['real', 'fake']):
                cls_dir = Path(root_dir) / cls_name
                if cls_dir.exists():
                    for img_path in cls_dir.glob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                            self.images.append(img_path)
                            self.labels.append(label)
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            return transform(img), self.labels[idx]
    
    dataset = SimpleImageDataset(dataset_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    feature_extractor.eval()
    device = next(feature_extractor.parameters()).device
    
    print(f"  Extracting features from {len(dataset)} images...")
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Feature extraction"):
            images = images.to(device)
            features = feature_extractor(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    print(f"  Computing t-SNE with {len(all_features)} samples...")
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(all_features) - 1),
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    
    embedding = tsne.fit_transform(all_features)
    
    return embedding, all_labels


def plot_tsne(embedding, labels, title, output_path):
    """Plot and save t-SNE visualization."""
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # Plot real samples
    real_mask = labels == 0
    fake_mask = labels == 1
    
    ax.scatter(embedding[real_mask, 0], embedding[real_mask, 1],
              c='blue', label='Real', alpha=0.6, s=30, edgecolors='navy')
    ax.scatter(embedding[fake_mask, 0], embedding[fake_mask, 1],
              c='red', label='Fake', alpha=0.6, s=30, edgecolors='darkred')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def generate_tsne_for_all(
    dataset_dir,
    feature_extractors,
    output_dir='results/tsne',
):
    """Generate t-SNE for all backbones."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for backbone, extractor in feature_extractors.items():
        print(f"\n[{backbone}]")
        
        try:
            embedding, labels = compute_tsne_embeddings(extractor, dataset_dir)
            
            output_file = output_dir / f"tsne_{backbone}.png"
            plot_tsne(embedding, labels, 
                     f't-SNE Feature Space: {backbone}',
                     output_file)
        
        except Exception as e:
            print(f"  Error: {e}")


# ============================================================================
# COMPARISON PLOTS
# ============================================================================

def plot_all_tsne_comparison(output_dir='results/tsne'):
    """Create 3-panel comparison of all backbones."""
    
    output_dir = Path(output_dir)
    
    # Load all t-SNE images
    tsne_files = list(output_dir.glob('tsne_*.png'))
    
    if len(tsne_files) < 3:
        print("Warning: Not enough t-SNE images found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, backbone in enumerate(['ResNet50', 'CLIP', 'DINOv2']):
        tsne_file = output_dir / f'tsne_{backbone}.png'
        
        if tsne_file.exists():
            img = Image.open(tsne_file)
            axes[idx].imshow(img)
            axes[idx].set_title(backbone, fontsize=14, fontweight='bold')
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'tsne_comparison.png'}")
    plt.close()


# ============================================================================
# GRADCAM COMPARISON
# ============================================================================

def plot_gradcam_comparison(gradcam_dir='results/gradcam_images'):
    """Create grid of GradCAM results."""
    
    gradcam_dir = Path(gradcam_dir)
    
    if not gradcam_dir.exists():
        print(f"Warning: GradCAM directory not found: {gradcam_dir}")
        return
    
    # Organize by backbone
    backbones = ['ResNet50', 'CLIP', 'DINOv2']
    images_per_backbone = {}
    
    for backbone in backbones:
        images = sorted(gradcam_dir.glob(f'*{backbone}*'))
        images_per_backbone[backbone] = images[:5]  # First 5 images
    
    # Create grid
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    
    for row, backbone in enumerate(backbones):
        images = images_per_backbone.get(backbone, [])
        
        for col in range(5):
            ax = axes[row, col]
            
            if col < len(images):
                img = Image.open(images[col])
                ax.imshow(img)
            
            ax.axis('off')
            
            if col == 0:
                ax.set_ylabel(backbone, fontsize=12, fontweight='bold')
    
    plt.suptitle('GradCAM Visualizations Across Backbones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(gradcam_dir).parent / 'gradcam_grid.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {Path(gradcam_dir).parent / 'gradcam_grid.png'}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_explainability():
    """Generate all explainability visualizations."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EXPLAINABILITY ANALYSIS")
    print("="*70 + "\n")
    
    # Import necessary models
    try:
        from models.resnet_feature_extractor import ResNetFeatureExtractor
        from models.clip_feature_extractor import CLIPFeatureExtractor
        from models.dinov2_feature_extractor import DINOv2FeatureExtractor
        from training.train_classifier import LinearClassifier
    except ImportError as e:
        print(f"Error importing models: {e}")
        print("Make sure you are in the project root directory")
        return
    
    # Paths
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dir = 'dataset/CelebDF_quarter/test'
    
    if not Path(test_dir).exists():
        print(f"Warning: Test directory not found: {test_dir}")
        return
    
    print("1. LOADING MODELS...")
    print("-" * 70)
    
    # Load feature extractors
    feature_extractors = {
        'ResNet50': ResNetFeatureExtractor(pretrained=True, freeze=True).to(device),
        'CLIP': CLIPFeatureExtractor(freeze=True).to(device),
        'DINOv2': DINOv2FeatureExtractor(freeze=True).to(device),
    }
    
    feature_dims = {
        'ResNet50': 2048,
        'CLIP': 512,
        'DINOv2': 768,
    }
    
    print("  ✓ Feature extractors loaded")
    
    # Load classifiers
    classifiers = {}
    checkpoint_dir = Path('checkpoints/cross')
    
    for backbone in ['resnet', 'clip', 'dinov2']:
        checkpoint = checkpoint_dir / f'celebdf_{backbone}.pt'
        
        if checkpoint.exists():
            classifier = LinearClassifier(
                input_dim=feature_dims[backbone.replace('resnet', 'ResNet50')
                                            .replace('clip', 'CLIP')
                                            .replace('dinov2', 'DINOv2')],
                num_classes=2
            ).to(device)
            
            state_dict = torch.load(checkpoint, map_location=device)
            if 'classifier_state' in state_dict:
                classifier.load_state_dict(state_dict['classifier_state'])
            else:
                classifier.load_state_dict(state_dict)
            
            classifiers[backbone] = classifier
    
    if classifiers:
        print(f"  ✓ Loaded {len(classifiers)} classifiers")
    else:
        print("  Warning: No classifiers found. Generate by running: python main.py --mode cross_eval")
    
    print("\n2. GENERATING GradCAM VISUALIZATIONS...")
    print("-" * 70)
    
    for backbone, extractor in feature_extractors.items():
        extractor.eval()
        
        if backbone in [f.replace('_', '').lower() for f in classifiers.keys()]:
            output_dir = f'results/gradcam_images/celebdf_{backbone.lower()}'
            
            try:
                if backbone in classifiers:
                    generate_gradcam_for_dataset(
                        test_dir,
                        extractor,
                        classifiers[backbone.lower()],
                        backbone,
                        output_dir,
                        num_samples=10
                    )
            except Exception as e:
                print(f"  Error generating GradCAM for {backbone}: {e}")
    
    print("\n3. GENERATING t-SNE VISUALIZATIONS...")
    print("-" * 70)
    
    try:
        generate_tsne_for_all(test_dir, feature_extractors)
        plot_all_tsne_comparison()
    except Exception as e:
        print(f"Error generating t-SNE: {e}")
    
    print("\n4. CREATING COMPARISON GRIDS...")
    print("-" * 70)
    
    try:
        plot_gradcam_comparison()
    except Exception as e:
        print(f"Error creating comparison grid: {e}")
    
    print("\n" + "="*70)
    print("EXPLAINABILITY ANALYSIS COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    generate_all_explainability()
