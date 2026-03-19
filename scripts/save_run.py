"""
save_run.py — Save results and checkpoints for the current run.
Reads data_fraction and epochs from configs/config.yaml and names
the backup folders automatically.

Usage:
    python scripts/save_run.py

Output folders:
    results_0.10_2epochs/
    checkpoints_0.10_2epochs/
"""

import shutil
import yaml
import os


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    fraction = config["data"]["data_fraction"]
    epochs   = config["training"]["epochs"]

    tag = f"{fraction:.2f}_{epochs}epochs"

    for src in ["results", "checkpoints"]:
        dst = f"{src}_{tag}"
        if not os.path.exists(src):
            print(f"  Skipping {src}/ — folder does not exist")
            continue
        if os.path.exists(dst):
            print(f"  {dst}/ already exists — skipping (delete it first to overwrite)")
            continue
        shutil.copytree(src, dst)
        print(f"  Saved {src}/ → {dst}/")


if __name__ == "__main__":
    main()
