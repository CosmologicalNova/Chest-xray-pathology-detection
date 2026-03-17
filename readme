# Chest X-Ray Pathology Detection

Multi-label pulmonary disease classification using the NIH Chest X-Ray14 dataset.
Trains and compares three architectures — Custom CNN, DenseNet-121, and Vision Transformer (ViT) —
combined into a soft voting ensemble with Monte Carlo Dropout uncertainty quantification
and Grad-CAM heatmap visualizations.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Get your Kaggle API token**
- Go to kaggle.com → Profile → Settings → API → Create New Token
- Copy the token from the popup

**3. Create a `.env` file in the project root**
```
KAGGLE_API_TOKEN=your_token_here
```

**4. Download and organize the dataset**
```bash
python scripts/setup_data.py
```
This downloads the full 45GB NIH dataset and organizes it into `data/images/` automatically.
If already downloaded before, it returns the cached path instantly.

---

## Running

**Quick test run — edit `configs/config.yaml` first:**
```yaml
data_fraction: 0.10
epochs: 2
```

**Train all models:**
```bash
python train.py
```

**Evaluate and generate all charts:**
```bash
python evaluate.py
```

**Learning curve across multiple data fractions:**

After each training run at a different fraction, rename the logs before the next run:
```bash
Rename-Item logs/custom_cnn_log.csv logs/custom_cnn_0.10.csv
Rename-Item logs/densenet_log.csv logs/densenet_0.10.csv
Rename-Item logs/vit_log.csv logs/vit_0.10.csv
```
Then after all fractions are done:
```bash
python scripts/learning_curve.py
```

---

## Project Structure

```
Chest-xray-pathology-detection/
├── configs/
│   └── config.yaml                 — all hyperparameters in one place
├── data/
│   ├── images/                     — 112,120 PNG chest X-rays
│   └── Data_Entry_2017.csv         — labels and metadata
├── scripts/
│   ├── setup_data.py               — downloads and organizes dataset
│   └── learning_curve.py           — plots AUC vs data fraction across runs
├── src/
│   ├── data/
│   │   └── dataset.py              — Dataset class, DataLoader, stratified splits
│   ├── models/
│   │   ├── custom_cnn.py           — baseline CNN built from scratch
│   │   ├── densenet.py             — DenseNet-121 with transfer learning
│   │   └── vit.py                  — Vision Transformer via timm
│   ├── training/
│   │   └── trainer.py              — training loop, early stopping, checkpointing
│   ├── evaluation/
│   │   ├── metrics.py              — AUC-ROC, F1, MC Dropout uncertainty
│   │   └── visualize.py            — all charts including Grad-CAM
│   └── ensemble/
│       └── soft_voting.py          — weighted probability averaging
├── checkpoints/                    — saved model weights (.pth files)
├── logs/                           — per-epoch training CSVs
├── results/                        — all generated charts and metrics
├── .env                            — your Kaggle token (never pushed to GitHub)
├── train.py                        — main training entry point
├── evaluate.py                     — main evaluation entry point
└── requirements.txt
```

---

## Models

| Model | Type | Description |
|---|---|---|
| Custom CNN | From scratch | 4 conv blocks, Global Average Pooling, baseline |
| DenseNet-121 | Transfer learning | ImageNet pretrained, CheXNet approach |
| ViT | Transfer learning | ImageNet-21k pretrained, patch-based self-attention |
| Ensemble | Soft voting | Weighted average of all three model outputs |

---

## Output Charts

| File | Description |
|---|---|
| `*_training_curves.png` | Train vs val loss and AUC over epochs |
| `*_roc_curves.png` | Per-class ROC curves (4x4 grid) |
| `*_confusion_matrices.png` | TP/FP/TN/FN per class |
| `*_uncertainty_plot.png` | MC Dropout variance vs prediction correctness |
| `*_gradcam_*.png` | Grad-CAM heatmaps showing model attention on X-rays |
| `ablation_comparison.png` | All 4 models compared side by side |
| `metrics_summary.csv` | Full per-class metrics for all models |
| `learning_curve.png` | AUC vs data fraction (run scripts/learning_curve.py) |

---

## Config Reference

| Parameter | What it controls |
|---|---|
| `data_fraction` | Fraction of 112k images to use |
| `batch_size` | Lower to 16 if CUDA out of memory |
| `learning_rate` | 0.0001 for transfer learning |
| `epochs` | Max training epochs |
| `early_stopping_patience` | Stop if val AUC stalls for N epochs |
| `dropout_rate` | Increase if overfitting |
| `freeze_epochs` | Epochs before unfreezing DenseNet/ViT backbone |
| `mc_dropout_passes` | Forward passes for uncertainty estimation |
| `ensemble/weights` | Per-model voting weight |

---

## Evaluation Metrics

- AUC-ROC per pathology class — primary metric, compared against CheXNet benchmark
- F1-Score, Precision, Recall per class
- MC Dropout variance vs misclassification rate — validates uncertainty estimates

---

## References

- CheXNet (Rajpurkar et al., 2017): https://arxiv.org/pdf/1711.05225
- Vision Transformer (Dosovitskiy et al., 2020): https://arxiv.org/pdf/2010.11929
- MC Dropout (Gal & Ghahramani, 2016): https://proceedings.mlr.press/v48/gal16.html
- NIH Dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data