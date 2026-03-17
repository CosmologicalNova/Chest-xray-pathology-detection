"""
metrics.py — Evaluation metrics and uncertainty quantification
==============================================================
Handles:
  - AUC-ROC per class and mean
  - F1, Precision, Recall per class
  - Monte Carlo Dropout uncertainty estimation
  - Full evaluation on test set with results saved to CSV
"""

import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score)

# Must match dataset.py CLASSES list
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


def evaluate_model(model, dataloader, device: str, threshold: float = 0.5):
    """
    Run model on a dataloader and compute all metrics.

    Args:
        threshold : Probability cutoff for binary prediction.
                    Lower threshold → higher recall, lower precision (catches more)
                    Higher threshold → higher precision, lower recall (misses more)
                    0.5 is default — tune per class if needed

    Returns:
        dict with per-class and mean AUC, F1, Precision, Recall
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)   # (N, 14)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, 14)
    all_preds = (all_probs >= threshold).astype(int)  # Binary predictions

    results = {}
    auc_scores, f1_scores, prec_scores, rec_scores = [], [], [], []

    for i, cls in enumerate(CLASSES):
        # Skip classes with no positive samples (causes sklearn error)
        if all_labels[:, i].sum() == 0:
            print(f"  Warning: No positive samples for {cls} in this split")
            continue

        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        f1  = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        prec = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        rec  = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)

        results[cls] = {"auc": auc, "f1": f1, "precision": prec, "recall": rec}
        auc_scores.append(auc)
        f1_scores.append(f1)
        prec_scores.append(prec)
        rec_scores.append(rec)

    # Mean across all classes
    results["MEAN"] = {
        "auc":       np.mean(auc_scores),
        "f1":        np.mean(f1_scores),
        "precision": np.mean(prec_scores),
        "recall":    np.mean(rec_scores)
    }

    return results, all_probs, all_labels, all_preds


def mc_dropout_uncertainty(model, images: torch.Tensor,
                            n_passes: int = 50, device: str = "cuda"):
    """
    Monte Carlo Dropout uncertainty estimation.

    How it works:
      1. Enable dropout during inference (model.enable_mc_dropout())
      2. Run the same image through the model n_passes times
      3. Each pass randomly drops different neurons → slightly different predictions
      4. Variance across passes = uncertainty

    Args:
        model    : Model with enable_mc_dropout() method
        images   : Batch of images (B, C, H, W)
        n_passes : Number of stochastic forward passes
                   More passes = more accurate uncertainty estimate
                   50 is standard, 100 for publication-quality results

    Returns:
        mean_probs    : (B, 14) — average prediction across all passes
        uncertainty   : (B, 14) — variance across passes (higher = less confident)
    """
    model.eval()
    model.enable_mc_dropout()  # Keep dropout ON during inference

    images = images.to(device)
    all_pass_probs = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_pass_probs.append(probs.cpu().numpy())

    # Stack: (n_passes, B, 14)
    all_pass_probs = np.stack(all_pass_probs, axis=0)

    mean_probs  = all_pass_probs.mean(axis=0)   # Average prediction
    uncertainty = all_pass_probs.var(axis=0)     # Variance = uncertainty

    return mean_probs, uncertainty


def print_results_table(results: dict):
    """Print a nicely formatted per-class metrics table."""
    print(f"\n{'Class':<22} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)
    for cls, metrics in results.items():
        print(f"{cls:<22} {metrics['auc']:>8.4f} {metrics['f1']:>8.4f} "
              f"{metrics['precision']:>10.4f} {metrics['recall']:>8.4f}")
