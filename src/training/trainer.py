"""
trainer.py — Training loop with full diagnostics
================================================
Handles:
  - Train and validation loop per epoch
  - Loss and AUC logging to CSV (for plotting later)
  - Early stopping (stops when val AUC stops improving)
  - Model checkpointing (saves best weights)
  - Learning rate scheduling

How to read the training diagnostics:
  Train loss >> Val loss       → Underfitting (model too simple or undertrained)
  Train loss << Val loss       → Overfitting (add regularization or more data)
  Both losses high and equal   → Underfitting (increase model capacity or epochs)
  Both losses low and equal    → Good fit
"""

import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    """
    General-purpose trainer — works with CustomCNN, DenseNet121, or ViT.

    Args:
        model          : PyTorch model
        train_loader   : Training DataLoader
        val_loader     : Validation DataLoader
        config         : Config dict (from yaml)
        model_name     : String label for saving ("custom_cnn", "densenet", "vit")
        device         : "cuda" or "cpu"
        freeze_epochs  : Epochs to freeze backbone before full fine-tuning
                         (only relevant for DenseNet and ViT)
    """

    def __init__(self, model, train_loader, val_loader, config,
                 model_name: str, device: str, freeze_epochs: int = 0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name
        self.device = device
        self.freeze_epochs = freeze_epochs

        # ── Loss function ─────────────────────────────────────────────────────
        # BCEWithLogitsLoss = Sigmoid + Binary Cross Entropy in one stable operation
        # pos_weight handles class imbalance:
        #   If Nodule appears in 5% of images, its weight = (95/5) = 19
        #   This penalizes missing a Nodule 19x more than a false positive
        # pos_weight is set in train.py after computing class frequencies
        self.criterion = nn.BCEWithLogitsLoss()   # pos_weight added in train.py

        # ── Optimizer ─────────────────────────────────────────────────────────
        # AdamW = Adam + proper weight decay decoupling
        # Why AdamW over Adam? Adam's weight decay is entangled with gradient scaling.
        # AdamW fixes this — standard choice for transformers and modern CNNs.
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get(model_name, {}).get("learning_rate", config["training"]["learning_rate"]),
            weight_decay=config["training"]["weight_decay"]
        )

        # ── LR Scheduler ──────────────────────────────────────────────────────
        # Cosine annealing: smoothly reduces LR from initial to near-zero
        # Why: constant LR causes loss to bounce. Decaying LR lets the model
        # settle into a better minimum at the end of training.
        # Change T_max to control how fast LR decays
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"],
            eta_min=1e-6
        )

        # ── Tracking ──────────────────────────────────────────────────────────
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.patience = config["training"]["early_stopping_patience"]
        self.epochs = config["training"]["epochs"]

        # Log file for plotting later
        os.makedirs(config["paths"]["logs"], exist_ok=True)
        os.makedirs(config["paths"]["checkpoints"], exist_ok=True)
        self.log_path = os.path.join(config["paths"]["logs"], f"{model_name}_log.csv")

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_auc", "lr"])

    # ── Training epoch ────────────────────────────────────────────────────────
    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping — prevents exploding gradients, especially in ViT
            # If you see loss suddenly jump to NaN, this is why
            # Increase max_norm if gradients are legitimately large
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)} "
                      f"| Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)

    # ── Validation epoch ──────────────────────────────────────────────────────
    def _val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                # Convert logits to probabilities for AUC calculation
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Mean AUC-ROC across all 14 classes
        # Skip classes with no positive samples (avoid sklearn error)
        auc_scores = []
        for i in range(all_labels.shape[1]):
            if all_labels[:, i].sum() > 0:
                auc_scores.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        mean_auc = np.mean(auc_scores)

        return total_loss / len(self.val_loader), mean_auc

    # ── Main training loop ────────────────────────────────────────────────────
    def train(self):
        print(f"\n{'='*60}")
        print(f"Training: {self.model_name}")
        print(f"{'='*60}")

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            # Unfreeze backbone after freeze_epochs
            # (only affects DenseNet and ViT which have freeze_backbone method)
            if epoch == self.freeze_epochs + 1 and hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()
                # Reset optimizer to include newly unfrozen parameters
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config["training"]["learning_rate"] * 0.1,  # Lower LR for fine-tuning
                    weight_decay=self.config["training"]["weight_decay"]
                )

            train_loss = self._train_epoch()
            val_loss, val_auc = self._val_epoch()
            self.scheduler.step()

            elapsed = time.time() - start
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch:03d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.1f}s")

            # Log to CSV
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, train_loss, val_loss, val_auc, current_lr]
                )

            # ── Checkpointing ──────────────────────────────────────────────────
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                ckpt_path = os.path.join(
                    self.config["paths"]["checkpoints"],
                    f"{self.model_name}_best.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  ✓ New best val AUC: {val_auc:.4f} — checkpoint saved")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.patience})")

            # ── Early stopping ─────────────────────────────────────────────────
            # Stops training when val AUC hasn't improved for patience epochs
            # This prevents wasting time training after the model has peaked
            # and prevents overfitting to the training set
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best val AUC: {self.best_val_auc:.4f})")
                break

        print(f"\nTraining complete. Best Val AUC: {self.best_val_auc:.4f}")
        print(f"Log saved to: {self.log_path}")
        return self.best_val_auc
