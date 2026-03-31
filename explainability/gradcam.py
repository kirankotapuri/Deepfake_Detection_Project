"""
GradCAM Explainability for Deepfake Detection

Generates attention heatmaps overlaid on face images.
Visualizes which regions influence the model's prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def gradcam_resnet(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """
    Compute GradCAM for ResNet-style model (conv before FC).

    Args:
        model: Full model (backbone + classifier).
        target_layer: Last conv layer to use for gradients.
        input_tensor: Input image [1, 3, H, W].
        target_class: Class to maximize (default: predicted class).

    Returns:
        Heatmap [H', W'] (spatial size of target_layer output).
    """
    model.eval()
    input_tensor = input_tensor.requires_grad_(True)
    activations = []
    gradients = []

    def save_activation(module, input, output):
        activations.append(output.detach())

    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = target_layer.register_forward_hook(save_activation)
    handle_bwd = target_layer.register_full_backward_hook(save_gradient)

    out = model(input_tensor)
    if target_class is None:
        target_class = out.argmax(dim=1).item()
    model.zero_grad()
    out[0, target_class].backward()

    handle_fwd.remove()
    handle_bwd.remove()

    acts = activations[0][0]  # [C, H, W]
    grads = gradients[0][0]   # [C, H, W]
    weights = grads.mean(dim=(1, 2))
    cam = (weights.unsqueeze(1).unsqueeze(2) * acts).sum(dim=0)
    cam = F.relu(cam)
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay_heatmap(
    heatmap: np.ndarray,
    image: np.ndarray | Image.Image,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay heatmap on image.

    Args:
        heatmap: 2D heatmap (will be resized to image size).
        image: RGB image [H, W, 3] or PIL Image.
        alpha: Blending factor.

    Returns:
        Overlaid image [H, W, 3].
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    from PIL import Image as PILImage
    heatmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_pil.resize((image.shape[1], image.shape[0])))
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    overlaid = (1 - alpha) * image / 255.0 + alpha * heatmap_colored
    return (np.clip(overlaid, 0, 1) * 255).astype(np.uint8)


def save_gradcam(
    image_path: str,
    heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.5,
) -> None:
    """
    Load image, overlay heatmap, save to file.

    Args:
        image_path: Path to original image.
        heatmap: GradCAM heatmap.
        output_path: Path to save result.
    """
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)
    overlaid = overlay_heatmap(heatmap, img_arr, alpha=alpha)
    Image.fromarray(overlaid).save(output_path)
