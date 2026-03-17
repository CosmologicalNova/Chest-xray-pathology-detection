"""
visualize.py — All diagnostic charts for model debugging and evaluation
=======================================================================
Charts included:
  1. Train/Val Loss curve        → tells you if model is overfitting/underfitting
  2. Train/Val AUC curve         → tells you when to stop training
  3. Per-class ROC curves        → detailed performance per pathology
  4. Multi-label Confusion Matrix → per-class TP/FP/TN/FN breakdown
  5. Uncertainty vs Error plot   → validates MC Dropout uncertainty
  6. Grad-CAM heatmaps           → where the model looks on the X-ray
  7. Ablation comparison table   → all models side by side
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


def plot_training_curves(log_csv_path: str, model_name: str, save_dir: str):
    """
    Plots train loss, val loss, and val AUC over epochs.

    How to interpret:
      - If train_loss >> val_loss: model is underfitting
      - If train_loss << val_loss: model is overfitting
        → Try: increase dropout, add weight decay, reduce model size, more data
      - If val_auc plateaus early: learning rate may be too high or model capacity too low
      - If val_auc keeps rising at last epoch: train more epochs
    """
    df = pd.read_csv(log_csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Train vs Val Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # AUC curve
    ax2.plot(df["epoch"], df["val_auc"], label="Val AUC", color="green")
    ax2.axhline(y=df["val_auc"].max(), linestyle="--",
                color="red", alpha=0.5, label=f"Best: {df['val_auc'].max():.4f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title(f"{model_name} — Validation AUC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved: {path}")


def plot_roc_curves(all_labels: np.ndarray, all_probs: np.ndarray,
                    model_name: str, save_dir: str):
    """
    Plots per-class ROC curves in a 4x4 grid.

    How to interpret:
      - Curve hugging top-left corner = high AUC = good
      - Diagonal line = random guessing (AUC = 0.5)
      - Classes with low AUC may need class-specific threshold tuning
        or more augmentation for that class
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        if all_labels[:, i].sum() == 0:
            axes[i].set_title(f"{cls}\n(no samples)")
            continue

        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        auc_score = auc(fpr, tpr)

        axes[i].plot(fpr, tpr, color="steelblue", lw=2,
                     label=f"AUC = {auc_score:.3f}")
        axes[i].plot([0, 1], [0, 1], "k--", lw=1)  # Diagonal = random
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1.02])
        axes[i].set_xlabel("FPR")
        axes[i].set_ylabel("TPR")
        axes[i].set_title(f"{cls}")
        axes[i].legend(loc="lower right", fontsize=8)

    # Hide extra subplots (we have 14 classes, 16 slots)
    for j in range(len(CLASSES), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"{model_name} — Per-Class ROC Curves", fontsize=16, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved: {path}")


def plot_confusion_matrices(all_labels: np.ndarray, all_preds: np.ndarray,
                             model_name: str, save_dir: str):
    """
    Plots a 2x2 confusion matrix per class.

    How to interpret:
      - High FP (False Positives): model is over-diagnosing, lower precision
        → Raise threshold for that class
      - High FN (False Negatives): model is missing cases, lower recall
        → Lower threshold for that class (critical in medical setting)
      - For medical AI: FN is usually worse than FP
        (missing a disease is more dangerous than a false alarm)
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        if all_labels[:, i].sum() == 0:
            axes[i].set_title(f"{cls}\n(no samples)")
            continue

        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        im = axes[i].imshow(cm, interpolation="nearest", cmap="Blues")
        axes[i].set_title(f"{cls}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(["Neg", "Pos"])
        axes[i].set_yticklabels(["Neg", "Pos"])

        # Annotate cells with counts
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                axes[i].text(col, row, str(cm[row, col]),
                             ha="center", va="center",
                             color="white" if cm[row, col] > cm.max() / 2 else "black")

    for j in range(len(CLASSES), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"{model_name} — Confusion Matrices (per class)", fontsize=16)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrices saved: {path}")


def plot_uncertainty_vs_error(mean_probs: np.ndarray, uncertainty: np.ndarray,
                               all_labels: np.ndarray, model_name: str,
                               save_dir: str):
    """
    Validates MC Dropout: checks if high uncertainty correlates with wrong predictions.

    How to interpret:
      - Points on the RIGHT (high uncertainty) should mostly be WRONG predictions
      - Points on the LEFT (low uncertainty) should mostly be CORRECT predictions
      - If there's NO separation: uncertainty estimates are not meaningful
      - If there IS separation: your MC Dropout is working correctly — you can
        trust the model's confidence scores as a clinical flag
    """
    preds = (mean_probs >= 0.5).astype(int)
    # Use mean uncertainty across all 14 classes per sample
    sample_uncertainty = uncertainty.mean(axis=1)
    # Is the prediction correct on at least one class?
    correct = (preds == all_labels).all(axis=1).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if c else "red" for c in correct]
    ax.scatter(sample_uncertainty, mean_probs.max(axis=1),
               c=colors, alpha=0.4, s=10)

    ax.set_xlabel("Prediction Uncertainty (MC Dropout Variance)")
    ax.set_ylabel("Max Predicted Probability")
    ax.set_title(f"{model_name} — Uncertainty vs Confidence\n"
                 f"Green = Correct | Red = Incorrect")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="green", label="Correct"),
                        Patch(color="red", label="Incorrect")])
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"{model_name}_uncertainty_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Uncertainty plot saved: {path}")


def plot_ablation_table(results_dict: dict, save_dir: str):
    """
    Side-by-side comparison table of all models.
    results_dict format: {"custom_cnn": metrics, "densenet": metrics, "vit": metrics, "ensemble": metrics}

    How to interpret:
      - Ensemble should be >= any individual model on mean AUC
      - If ensemble < individual model: soft voting weights may need tuning
      - Large gap between DenseNet and Custom CNN validates transfer learning
    """
    model_names = list(results_dict.keys())
    metrics = ["auc", "f1", "precision", "recall"]

    data = {m: [results_dict[name]["MEAN"][m] for name in model_names]
            for m in metrics}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, metric in enumerate(metrics):
        bars = axes[i].bar(model_names, data[metric], color=colors)
        axes[i].set_title(metric.upper(), fontsize=13)
        axes[i].set_ylim(0, 1.0)
        axes[i].set_ylabel("Score")
        axes[i].tick_params(axis="x", rotation=15)
        # Annotate bar heights
        for bar in bars:
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.01,
                         f"{bar.get_height():.3f}",
                         ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Ablation Study — Mean Metrics Across All Classes",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation chart saved: {path}")


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Maps.

    What it does:
      Hooks into the last convolutional layer of a CNN to capture:
        1. The feature maps (what the layer detected)
        2. The gradients of the target class w.r.t. those maps
           (how much each feature map region matters for that class)
      Then weights the feature maps by their gradients and overlays on the image.

    Why last conv layer?
      It has the highest-level spatial features — still spatially resolved
      (unlike fully connected layers which lose spatial information).

    Note: Grad-CAM works naturally on CNNs. For ViT, use attention rollout instead.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model        : Trained CNN model
            target_layer : Last conv layer, e.g. model.features[-1]
                           Change this if your architecture is different
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture forward activations and backward gradients
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor, class_idx: int):
        """
        Generate Grad-CAM heatmap for a specific class.

        Args:
            image     : Single image tensor (1, C, H, W)
            class_idx : Which of the 14 classes to visualize
        """
        self.model.eval()
        output = self.model(image)

        self.model.zero_grad()
        # Backprop only through the target class score
        output[0, class_idx].backward()

        # Global average pool gradients to get channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weight the activations by gradient importance
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()


def plot_gradcam(model, image_tensor: torch.Tensor, target_layer,
                 class_idx: int, original_image: np.ndarray,
                 model_name: str, save_dir: str):
    """
    Saves a Grad-CAM overlay showing where the model looks for a specific class.

    Args:
        class_idx      : Index of pathology class (0=Atelectasis, 6=Pneumonia, etc.)
        original_image : Unnormalized numpy image (H, W, 3) for display
    """
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(image_tensor, class_idx)

    # Resize CAM to match image size
    cam_resized = np.array(
        plt.cm.jet(cam)[:, :, :3]  # Apply colormap
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title(f"Grad-CAM: {CLASSES[class_idx]}")
    axes[1].axis("off")

    # Overlay: blend original image with heatmap
    from PIL import Image as PILImage
    cam_pil = PILImage.fromarray((cam * 255).astype(np.uint8))
    cam_resized_full = np.array(cam_pil.resize((224, 224), PILImage.BILINEAR)) / 255.0
    cam_color = plt.cm.jet(cam_resized_full)[:, :, :3]
    overlay = 0.5 * original_image / 255.0 + 0.5 * cam_color
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(f"{model_name} — Grad-CAM for {CLASSES[class_idx]}")
    plt.tight_layout()
    path = os.path.join(save_dir,
                        f"{model_name}_gradcam_{CLASSES[class_idx]}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Grad-CAM saved: {path}")
