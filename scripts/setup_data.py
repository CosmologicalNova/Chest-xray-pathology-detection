"""
scripts/setup_data.py — One-time setup: downloads and organizes NIH dataset
Run once from project root: python scripts/setup_data.py
Requires KAGGLE_API_TOKEN in .env file
"""

import os
import shutil
import glob
from dotenv import load_dotenv

load_dotenv()
os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN", "")

OUTPUT_IMAGES_DIR = "data/images"
OUTPUT_CSV_PATH   = "data/Data_Entry_2017.csv"
EXPECTED_IMAGES   = 112120


def main():
    if not os.environ["KAGGLE_API_TOKEN"]:
        print("ERROR: KAGGLE_API_TOKEN not found in .env file.")
        return

    try:
        import kagglehub
    except ImportError:
        os.system("pip install kagglehub")
        import kagglehub

    print("Downloading NIH Chest X-Ray dataset via kagglehub...")
    print("45GB on first run — returns cached path instantly if already downloaded.\n")

    download_path = kagglehub.dataset_download("nih-chest-xrays/data")
    print(f"Dataset located at: {download_path}")

    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Search recursively since kagglehub may extract into subdirectories
    print("\nLocating PNG images...")
    all_pngs = glob.glob(os.path.join(download_path, "**", "*.png"), recursive=True)
    print(f"Found {len(all_pngs):,} PNG files")

    already_moved = len(glob.glob(os.path.join(OUTPUT_IMAGES_DIR, "*.png")))

    if already_moved == len(all_pngs) and already_moved > 0:
        print(f"Images already in {OUTPUT_IMAGES_DIR} — skipping copy step.")
    else:
        print(f"Copying to {OUTPUT_IMAGES_DIR} (runs once, skipped on future runs)...")
        for i, src in enumerate(all_pngs):
            dst = os.path.join(OUTPUT_IMAGES_DIR, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            if i % 5000 == 0:
                print(f"  {i:,} / {len(all_pngs):,} copied...")
        print("  Done.")

    csv_candidates = glob.glob(
        os.path.join(download_path, "**", "Data_Entry_2017.csv"), recursive=True
    )
    if csv_candidates and not os.path.exists(OUTPUT_CSV_PATH):
        shutil.copy2(csv_candidates[0], OUTPUT_CSV_PATH)
        print(f"CSV copied to {OUTPUT_CSV_PATH}")
    elif os.path.exists(OUTPUT_CSV_PATH):
        print(f"CSV already at {OUTPUT_CSV_PATH}")
    else:
        print("WARNING: Data_Entry_2017.csv not found — copy it manually to data/")

    print("\n" + "="*50)
    actual = len(glob.glob(os.path.join(OUTPUT_IMAGES_DIR, "*.png")))
    csv_ok = os.path.exists(OUTPUT_CSV_PATH)
    print(f"Images : {actual:,} / {EXPECTED_IMAGES:,}")
    print(f"CSV    : {'Found' if csv_ok else 'MISSING'}")

    if actual >= EXPECTED_IMAGES and csv_ok:
        print("\nData ready. Set data_fraction: 0.02 and epochs: 2 in config.yaml then run train.py")
    else:
        print(f"\nExpected {EXPECTED_IMAGES:,} images — download may not have completed.")


if __name__ == "__main__":
    main()
