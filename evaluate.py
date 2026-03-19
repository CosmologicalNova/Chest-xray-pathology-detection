"""
evaluate.py — Full evaluation on test set + all diagnostic charts
Run AFTER train.py: python evaluate.py
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd

from src.models.custom_cnn import CustomCNN
from src.models.densenet import DenseNet121
from src.models.vit import ViTClassifier
from src.evaluation.metrics import (evaluate_model, mc_dropout_uncertainty,
                                     print_results_table, CLASSES)
from src.evaluation.visualize import (plot_training_curves, plot_roc_curves,
                                       plot_confusion_matrices,
                                       plot_uncertainty_vs_error,
                                       plot_ablation_table,
                                       plot_gradcam, GradCAM)
from src.ensemble.soft_voting import run_ensemble
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def load_model(model_class, checkpoint_path, config, device, **kwargs):
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    return model


def run_gradcam(model, model_name, test_loader, device, results_dir):
    """
    Generates Grad-CAM heatmaps for a sample batch.
    Only runs on CNN-based models (Custom CNN and DenseNet).
    ViT uses attention maps not conv feature maps so standard Grad-CAM does not apply.
    Targets the last conv/dense block where spatial features are highest level.
    """
    if model_name == "vit":
        print(f"  Skipping Grad-CAM for ViT (not CNN-based)")
        return

    # Get target layer — last convolutional block for each model
    # Change this if you modify the model architecture
    if model_name == "custom_cnn":
        target_layer = model.features[-1].block[0]   # Last Conv2d in last ConvBlock
    elif model_name == "densenet":
        target_layer = model.features.denseblock4     # Last dense block

    model.train()  # Grad-CAM needs gradients — temporarily set to train mode
    images, labels = next(iter(test_loader))
    images = images.to(device)

    # Generate Grad-CAM for up to 3 sample images, for the most confident class
    n_samples = min(3, images.shape[0])
    for i in range(n_samples):
        image = images[i:i+1]

        # Get model prediction to find most confident class for this image
        with torch.no_grad():
            logits = model(image)
            probs = torch.sigmoid(logits)
        class_idx = probs.argmax(dim=1).item()

        # Get original image for display (denormalize)
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
                  np.array([0.485, 0.456, 0.406]))
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        plot_gradcam(
            model=model,
            image_tensor=image,
            target_layer=target_layer,
            class_idx=class_idx,
            original_image=img_np,
            model_name=f"{model_name}_sample{i}",
            save_dir=results_dir
        )

    model.eval()


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    num_classes = config["labels"]["num_classes"]
    results_dir = config["paths"]["results"]
    logs_dir    = config["paths"]["logs"]
    ckpt_dir    = config["paths"]["checkpoints"]
    os.makedirs(results_dir, exist_ok=True)

    # ── Load test dataloader ───────────────────────────────────────────────────
    test_loader = torch.load(
        os.path.join(ckpt_dir, "test_loader.pt"), weights_only=False
    )

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading trained models...")
    custom_cnn = load_model(
        CustomCNN, os.path.join(ckpt_dir, "custom_cnn_best.pth"),
        config, device,
        num_classes=num_classes,
        dropout_rate=config["custom_cnn"]["dropout_rate"]
    )
    densenet = load_model(
        DenseNet121, os.path.join(ckpt_dir, "densenet_best.pth"),
        config, device,
        num_classes=num_classes,
        dropout_rate=config["densenet"]["dropout_rate"]
    )
    vit = load_model(
        ViTClassifier, os.path.join(ckpt_dir, "vit_best.pth"),
        config, device,
        num_classes=num_classes,
        model_name=config["vit"]["model_name"],
        dropout_rate=config["vit"]["dropout_rate"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1: Training curves
    # ══════════════════════════════════════════════════════════════════════════
    print("\nPlotting training curves...")
    for model_name in ["custom_cnn", "densenet", "vit"]:
        log_path = os.path.join(logs_dir, f"{model_name}_log.csv")
        if os.path.exists(log_path):
            plot_training_curves(log_path, model_name, results_dir)
        else:
            print(f"  No log found for {model_name} — skipping curve")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2: Per-model evaluation + MC Dropout + Grad-CAM
    # ══════════════════════════════════════════════════════════════════════════
    all_results = {}

    for model_name, model in [("custom_cnn", custom_cnn),
                               ("densenet",   densenet),
                               ("vit",        vit)]:
        print(f"\nEvaluating {model_name}...")
        results, all_probs, all_labels, all_preds = evaluate_model(
            model, test_loader, device
        )
        all_results[model_name] = results
        print_results_table(results)

        plot_roc_curves(all_labels, all_probs, model_name, results_dir)
        plot_confusion_matrices(all_labels, all_preds, model_name, results_dir)

        # MC Dropout uncertainty
        print(f"  Running MC Dropout for {model_name}...")
        sample_images, sample_labels = next(iter(test_loader))
        mc_passes = config[model_name]["mc_dropout_passes"]
        mean_probs, uncertainty = mc_dropout_uncertainty(
            model, sample_images, n_passes=mc_passes, device=device
        )
        plot_uncertainty_vs_error(
            mean_probs, uncertainty,
            sample_labels.numpy(), model_name, results_dir
        )

        # Grad-CAM (CNN models only)
        print(f"  Running Grad-CAM for {model_name}...")
        run_gradcam(model, model_name, test_loader, device, results_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3: Ensemble
    # ══════════════════════════════════════════════════════════════════════════
    print("\nRunning ensemble...")
    models_dict = {"custom_cnn": custom_cnn, "densenet": densenet, "vit": vit}
    ensemble_probs = run_ensemble(
        models_dict, test_loader, device, config["ensemble"]["weights"]
    )

    all_labels_list = []
    for _, labels in test_loader:
        all_labels_list.append(labels.numpy())
    all_labels = np.concatenate(all_labels_list, axis=0)

    ensemble_results = {}
    auc_scores, f1_scores, prec_scores, rec_scores = [], [], [], []
    all_preds = (ensemble_probs >= 0.5).astype(int)

    for i, cls in enumerate(CLASSES):
        if all_labels[:, i].sum() == 0:
            continue
        auc  = roc_auc_score(all_labels[:, i], ensemble_probs[:, i])
        f1   = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        prec = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        rec  = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        ensemble_results[cls] = {"auc": auc, "f1": f1, "precision": prec, "recall": rec}
        auc_scores.append(auc)
        f1_scores.append(f1)
        prec_scores.append(prec)
        rec_scores.append(rec)

    ensemble_results["MEAN"] = {
        "auc":       np.mean(auc_scores),
        "f1":        np.mean(f1_scores),
        "precision": np.mean(prec_scores),
        "recall":    np.mean(rec_scores)
    }
    all_results["ensemble"] = ensemble_results

    print("\nEnsemble Results:")
    print_results_table(ensemble_results)
    plot_roc_curves(all_labels, ensemble_probs, "ensemble", results_dir)
    plot_confusion_matrices(all_labels, all_preds, "ensemble", results_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: Ablation chart
    # ══════════════════════════════════════════════════════════════════════════
    plot_ablation_table(all_results, results_dir)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: Save metrics CSV
    # ══════════════════════════════════════════════════════════════════════════
    rows = []
    for model_name, results in all_results.items():
        for cls, metrics in results.items():
            rows.append({"model": model_name, "class": cls, **metrics})

    pd.DataFrame(rows).to_csv(
        os.path.join(results_dir, "metrics_summary.csv"), index=False
    )
    print(f"\nAll results saved to {results_dir}/")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()