"""
ensemble/soft_voting.py — Soft Voting Ensemble
===============================================
Combines probability outputs from CustomCNN, DenseNet121, and ViT.

Why soft voting over hard voting?
  Hard voting: each model votes yes/no per class → majority wins
  Soft voting: average the raw probabilities → preserves confidence information

  Example:
    CustomCNN  → Nodule prob: 0.90
    DenseNet   → Nodule prob: 0.85
    ViT        → Nodule prob: 0.30
    Hard vote  → 2 yes, 1 no → Nodule = Yes
    Soft vote  → (0.90+0.85+0.30)/3 = 0.68 → Nodule = Yes (with lower confidence)

  Soft voting is better because a near-miss (0.48) is treated differently
  from a definitive no (0.05) — hard voting treats them identically.

Weighted soft voting:
  Models that perform better on val set get higher weight.
  If DenseNet has AUC 0.82 and Custom CNN has AUC 0.70,
  DenseNet's predictions should count more.
"""

import numpy as np
import torch


def get_model_probs(model, dataloader, device: str) -> np.ndarray:
    """
    Run a model on a dataloader and collect all predicted probabilities.

    Returns:
        np.ndarray of shape (N, 14)
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def soft_voting_ensemble(prob_list: list, weights: list = None) -> np.ndarray:
    """
    Combine a list of probability arrays via weighted averaging.

    Args:
        prob_list : List of (N, 14) probability arrays, one per model
        weights   : Optional list of weights per model
                    e.g. [0.2, 0.4, 0.4] for [custom_cnn, densenet, vit]
                    Default is equal weighting [0.33, 0.33, 0.33]

    Returns:
        ensemble_probs : (N, 14) averaged probabilities

    What to change:
        - Tune weights based on individual model val AUC scores
        - Use val AUC directly as weights: weights = [auc_cnn, auc_densenet, auc_vit]
          then normalize: weights = [w/sum(weights) for w in weights]
        - Try max voting: np.max(stacked, axis=0) instead of average
          (takes highest confidence prediction per class)
    """
    stacked = np.stack(prob_list, axis=0)   # (n_models, N, 14)

    if weights is None:
        weights = [1.0 / len(prob_list)] * len(prob_list)

    weights = np.array(weights)
    weights = weights / weights.sum()        # Normalize to sum to 1

    # Weighted average across models
    ensemble_probs = np.average(stacked, axis=0, weights=weights)  # (N, 14)
    return ensemble_probs


def run_ensemble(models: dict, dataloader, device: str,
                 weights: list = None) -> np.ndarray:
    """
    Convenience function: runs all models and returns ensemble probabilities.

    Args:
        models  : dict of {"model_name": model_instance}
        weights : list of weights in same order as models.values()

    Returns:
        ensemble_probs : (N, 14)
    """
    print("Running ensemble inference...")
    prob_list = []
    for name, model in models.items():
        print(f"  Getting probabilities from {name}...")
        probs = get_model_probs(model, dataloader, device)
        prob_list.append(probs)
        print(f"  {name}: {probs.shape}")

    ensemble_probs = soft_voting_ensemble(prob_list, weights)
    print(f"Ensemble output shape: {ensemble_probs.shape}")
    return ensemble_probs
