"""
scripts/learning_curve.py — Plot AUC vs data fraction across training runs
==========================================================================
Run after you have trained at multiple data fractions and renamed your logs.

Workflow:
  1. Train at data_fraction: 0.05  → rename logs to *_0.05.csv
  2. Train at data_fraction: 0.25  → rename logs to *_0.25.csv
  3. Train at data_fraction: 0.50  → rename logs to *_0.50.csv
  4. Train at data_fraction: 1.00  → rename logs to *_1.00.csv
  5. Run: python scripts/learning_curve.py

What it tells you:
  - Curve still rising at 1.0  → model is data hungry, get more data
  - Curve flattens at 0.50     → more data won't help, improve the model instead
  - Large gap between models   → architecture matters more than data size
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR    = "logs"
RESULTS_DIR = "results"
MODEL_NAMES = ["custom_cnn", "densenet", "vit"]
COLORS      = {"custom_cnn": "#4C72B0", "densenet": "#DD8452", "vit": "#55A868"}


def get_best_auc(csv_path: str) -> float:
    """Returns the best val AUC from a training log CSV."""
    df = pd.read_csv(csv_path)
    return df["val_auc"].max()


def parse_fraction_from_filename(filename: str) -> float:
    """
    Extracts data fraction from filename.
    e.g. custom_cnn_0.25.csv → 0.25
    """
    base = os.path.basename(filename)
    fraction_str = base.replace(".csv", "").split("_")[-1]
    return float(fraction_str)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    found_any = False

    for model_name in MODEL_NAMES:
        # Find all log files for this model across different fractions
        pattern = os.path.join(LOGS_DIR, f"{model_name}_*.csv")
        log_files = sorted(glob.glob(pattern))

        # Filter out the default log file (no fraction in name)
        log_files = [f for f in log_files
                     if any(c.isdigit() for c in os.path.basename(f).split("_")[-1])]

        if not log_files:
            print(f"No fraction logs found for {model_name} — skipping.")
            print(f"  Expected files like: logs/{model_name}_0.05.csv")
            continue

        fractions = []
        aucs = []

        for log_file in log_files:
            try:
                fraction = parse_fraction_from_filename(log_file)
                best_auc = get_best_auc(log_file)
                fractions.append(fraction)
                aucs.append(best_auc)
                print(f"  {model_name} @ {fraction*100:.0f}% → best AUC: {best_auc:.4f}")
            except Exception as e:
                print(f"  Skipping {log_file}: {e}")
                continue

        if fractions:
            # Sort by fraction so line connects left to right
            pairs = sorted(zip(fractions, aucs))
            fractions, aucs = zip(*pairs)

            ax.plot(
                [f * 100 for f in fractions],
                aucs,
                marker="o",
                label=model_name,
                color=COLORS.get(model_name, "gray"),
                linewidth=2,
                markersize=8
            )
            # Annotate each point with its AUC value
            for f, a in zip(fractions, aucs):
                ax.annotate(f"{a:.3f}", xy=(f * 100, a),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
            found_any = True

    if not found_any:
        print("\nNo learning curve data found.")
        print("Train at multiple fractions and rename logs before running this script.")
        print("Example after 0.05 run:")
        print("  Rename logs/custom_cnn_log.csv → logs/custom_cnn_0.05.csv")
        return

    ax.set_xlabel("Training Data Used (%)", fontsize=12)
    ax.set_ylabel("Best Validation AUC-ROC", fontsize=12)
    ax.set_title("Learning Curve — AUC vs Training Data Size", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    path = os.path.join(RESULTS_DIR, "learning_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nLearning curve saved: {path}")


if __name__ == "__main__":
    main()
