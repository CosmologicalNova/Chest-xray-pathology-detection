"""
train.py — Main entry point for training all models
=====================================================
Run from project root:
    python train.py

What happens:
  1. Loads config from configs/config.yaml
  2. Builds data loaders with stratified train/val/test split
  3. Trains CustomCNN → DenseNet121 → ViT sequentially
  4. Saves best checkpoint per model to checkpoints/
  5. Logs train/val loss and AUC to logs/ for plotting
"""

import os
import yaml
import torch
import numpy as np

from src.data.dataset import get_dataloaders
from src.models.custom_cnn import CustomCNN
from src.models.densenet import DenseNet121
from src.models.vit import ViTClassifier
from src.training.trainer import Trainer


def compute_pos_weights(train_loader, num_classes: int, device: str):
    """
    Compute positive class weights for BCEWithLogitsLoss.

    Why: The NIH dataset is heavily imbalanced.
    Hernia appears in <0.2% of images. Without weighting,
    the model learns to always predict "No Finding" and still
    gets high accuracy — but 0 recall on rare diseases.

    pos_weight[i] = (# negative samples) / (# positive samples) for class i
    This penalizes missing a positive case proportionally to its rarity.
    """
    print("Computing class weights for imbalanced dataset...")
    all_labels = []

    for _, labels in train_loader:
        all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    pos_counts = all_labels.sum(axis=0)
    neg_counts = len(all_labels) - pos_counts

    # Avoid division by zero for classes with no samples
    pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    print(f"Class weights range: {pos_weights.min():.1f} — {pos_weights.max():.1f}")

    return torch.tensor(pos_weights, dtype=torch.float32).to(device)


def main():
    # ── Load config ───────────────────────────────────────────────────────────
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ── Device setup ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using device: mps (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("Using device: cpu")

    # ── Create output directories ─────────────────────────────────────────────
    for path in config["paths"].values():
        os.makedirs(path, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=config["data"]["csv_path"],
        images_dir=config["data"]["images_dir"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        data_fraction=config["data"]["data_fraction"]
    )

    num_classes = config["labels"]["num_classes"]
    pos_weights = compute_pos_weights(train_loader, num_classes, device)

    # ── Save test loader for evaluate.py ──────────────────────────────────────
    # We save the test split indices so evaluate.py uses the exact same test set
    torch.save(test_loader, os.path.join(config["paths"]["checkpoints"],
                                          "test_loader.pt"))

    # ══════════════════════════════════════════════════════════════════════════
    # Model 1: Custom CNN (Baseline)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("MODEL 1: Custom CNN (Baseline)")
    print("="*60)

    custom_cnn = CustomCNN(
        num_classes=num_classes,
        dropout_rate=config["custom_cnn"]["dropout_rate"]
    )

    cnn_trainer = Trainer(
        model=custom_cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_name="custom_cnn",
        device=device,
        freeze_epochs=0      # No pretrained weights to freeze
    )
    # Apply class weights to loss function
    cnn_trainer.criterion.pos_weight = pos_weights
    cnn_trainer.train()

    # ══════════════════════════════════════════════════════════════════════════
    # Model 2: DenseNet-121 with Transfer Learning
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("MODEL 2: DenseNet-121 (Transfer Learning)")
    print("="*60)

    densenet = DenseNet121(
        num_classes=num_classes,
        dropout_rate=config["densenet"]["dropout_rate"]
    )
    # Freeze backbone for first N epochs
    densenet.freeze_backbone()

    densenet_trainer = Trainer(
        model=densenet,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_name="densenet",
        device=device,
        freeze_epochs=config["densenet"]["freeze_epochs"]
    )
    densenet_trainer.criterion.pos_weight = pos_weights
    densenet_trainer.train()

    # ══════════════════════════════════════════════════════════════════════════
    # Model 3: Vision Transformer
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("MODEL 3: Vision Transformer (ViT)")
    print("="*60)

    vit = ViTClassifier(
        num_classes=num_classes,
        model_name=config["vit"]["model_name"],
        dropout_rate=config["vit"]["dropout_rate"]
    )
    vit.freeze_backbone()

    vit_trainer = Trainer(
        model=vit,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_name="vit",
        device=device,
        freeze_epochs=config["densenet"]["freeze_epochs"]
    )
    vit_trainer.criterion.pos_weight = pos_weights
    vit_trainer.train()

    print("\n" + "="*60)
    print("All models trained. Run evaluate.py for full evaluation.")
    print("="*60)


if __name__ == "__main__":
    main()
