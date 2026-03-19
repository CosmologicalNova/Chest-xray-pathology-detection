"""
dataset.py — NIH Chest X-Ray14 Dataset and DataLoaders
=======================================================
Handles:
  - One-hot encoding of 14 pathology labels from pipe-separated CSV column
  - Stratified train/val/test split (70/15/15) with multi-label fallback
  - Image loading and augmentation pipeline
  - DataLoader construction
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


def get_transforms(split: str, image_size: int = 224):
    """
    Returns image transforms for a given data split.

    Train: augmentations to reduce overfitting on small datasets
    Val/Test: deterministic resize + normalize only (no randomness)

    ImageNet normalization stats are required because DenseNet and ViT
    were pretrained on ImageNet — input must match what the model expected.
    Using different stats will degrade transfer learning performance.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH Chest X-Ray14.

    Each sample:
      image  : Tensor (3, H, W), normalized to ImageNet stats
      labels : Tensor (14,), float32, one value per pathology class
               1.0 = disease present, 0.0 = absent

    Note: X-rays are grayscale but converted to 3-channel RGB
    so they match the expected input shape of pretrained DenseNet/ViT.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["Image Index"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[CLASSES].values.astype(float), dtype=torch.float32
        )
        return image, labels


def prepare_dataframes(csv_path: str, data_fraction: float = 1.0,
                        train_split: float = 0.70, val_split: float = 0.15):
    """
    Reads the NIH CSV, one-hot encodes 14 classes, and performs a
    stratified train/val/test split.

    Stratification strategy:
      1. Build a combined key from all 14 class flags (e.g., "10000001000000")
         This tries to preserve the full multi-label distribution across splits
      2. If any combo appears < 2 times (can't stratify), fall back to
         stratifying on the single most common class
      3. If that also fails, fall back to a plain random split

    Args:
        csv_path      : Path to Data_Entry_2017.csv
        data_fraction : Fraction of full dataset to use (0.01 to 1.0)
        train_split   : Fraction for training (default 0.70)
        val_split     : Fraction for validation (default 0.15)
                        Test = 1 - train_split - val_split = 0.15

    Returns:
        train_df, val_df, test_df
    """
    df = pd.read_csv(csv_path)

    # One-hot encode the pipe-separated "Finding Labels" column
    # e.g., "Atelectasis|Effusion" -> Atelectasis=1, Effusion=1, all others=0
    # "No Finding" -> all 14 classes = 0
    for cls in CLASSES:
        df[cls] = df["Finding Labels"].apply(
            lambda x: 1 if cls in str(x).split("|") else 0
        )

    # Subsample before splitting to preserve class proportions
    if data_fraction < 1.0:
        df = df.sample(frac=data_fraction, random_state=42).reset_index(drop=True)

    print(f"Total samples: {len(df)}")

    # Build a combined stratification key from all 14 class flags
    # "10000001000000" = Atelectasis + Pneumothorax positive, rest negative
    df["stratify_key"] = df[CLASSES].astype(str).agg("".join, axis=1)

    test_size = 1.0 - train_split   # 0.30 of total

    # Try full multi-label stratification
    try:
        key_counts = df["stratify_key"].value_counts()
        rare_keys = key_counts[key_counts < 2].index
        if len(rare_keys) > 0:
            raise ValueError(f"{len(rare_keys)} rare multi-label combos — falling back")

        train_df, temp_df = train_test_split(
            df, test_size=test_size,
            stratify=df["stratify_key"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5,
            stratify=temp_df["stratify_key"], random_state=42
        )
        print("Split strategy: multi-label stratification")

    except Exception as e1:
        # Fall back: stratify on the single most common class
        try:
            dominant_class = df[CLASSES].sum().idxmax()
            train_df, temp_df = train_test_split(
                df, test_size=test_size,
                stratify=df[dominant_class], random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5,
                stratify=temp_df[dominant_class], random_state=42
            )
            print(f"Split strategy: single-class stratification ('{dominant_class}')")

        except Exception as e2:
            # Last resort: random split (no stratification)
            train_df, temp_df = train_test_split(
                df, test_size=test_size, random_state=42
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, random_state=42
            )
            print("Split strategy: random (stratification failed)")

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_dataloaders(csv_path: str, images_dir: str,
                    image_size: int = 224, batch_size: int = 32,
                    num_workers: int = 0, data_fraction: float = 1.0):
    """
    Builds and returns train, val, test DataLoaders.

    Args:
        csv_path      : Path to Data_Entry_2017.csv
        images_dir    : Directory containing the .png X-ray images
        image_size    : Resize target — 224 required for DenseNet/ViT
        batch_size    : Images per batch (reduce to 16 if out of memory)
        num_workers   : Parallel data loading workers
                        Set to 0 on Mac — macOS + PyTorch multiprocessing is buggy
        data_fraction : Fraction of full dataset to use

    Returns:
        train_loader, val_loader, test_loader
    """
    train_df, val_df, test_df = prepare_dataframes(
        csv_path, data_fraction
    )

    train_dataset = ChestXrayDataset(
        train_df, images_dir, get_transforms("train", image_size)
    )
    val_dataset = ChestXrayDataset(
        val_df, images_dir, get_transforms("val", image_size)
    )
    test_dataset = ChestXrayDataset(
        test_df, images_dir, get_transforms("test", image_size)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, test_loader
