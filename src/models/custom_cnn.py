"""
custom_cnn.py — Baseline CNN built from scratch
================================================
Purpose: Establish a performance floor using only basic convolution blocks.
         No pretrained weights — learns entirely from NIH data.

Architecture overview:
  Input (3, 224, 224)
    → Conv Block 1 → Conv Block 2 → Conv Block 3 → Conv Block 4
    → Global Average Pooling
    → Dropout → FC → Sigmoid (14 outputs)

Why Global Average Pooling instead of Flatten?
  Flatten causes a huge parameter explosion at the FC layer.
  GAP averages each feature map to a single value, drastically reducing
  parameters and acting as a built-in regularizer.

Why Batch Normalization after each Conv?
  Stabilizes training by normalizing layer inputs.
  Allows higher learning rates and reduces sensitivity to initialization.
  Remove it and you'll likely see unstable loss curves.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    One convolutional block: Conv → BatchNorm → ReLU → MaxPool

    What to change:
      - kernel_size: larger kernel (5x5) captures bigger spatial patterns
        but costs more memory. 3x3 is standard.
      - pool_size: larger pooling = more aggressive spatial downsampling
        = smaller feature maps downstream = less computation but less detail
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool_size: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            # Conv2d learns spatial filters (edges, textures, shapes)
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,   # Same padding keeps spatial size
                      bias=False),                 # bias=False because BatchNorm handles bias
            nn.BatchNorm2d(out_channels),          # Normalize across the batch
            nn.ReLU(inplace=True),                 # Non-linearity — inplace saves memory
            nn.MaxPool2d(pool_size)                # Halves spatial dimensions
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Lightweight baseline CNN for multi-label chest X-ray classification.

    Channel progression: 3 → 32 → 64 → 128 → 256
    Each block doubles the channels (more feature detectors)
    while MaxPool halves the spatial size (224 → 112 → 56 → 28 → 14)

    What to change:
      - Add more ConvBlocks to go deeper (more capacity, risk of overfitting)
      - Increase channel sizes (e.g., 64→128→256→512) for more capacity
      - Add Dropout after each ConvBlock for stronger regularization
      - Change dropout_rate: higher = more regularization, lower = less
        If train acc >> val acc (overfitting), increase dropout_rate
        If both train and val acc are low (underfitting), decrease dropout_rate
    """
    def __init__(self, num_classes: int = 14, dropout_rate: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32),    # 224×224×3   → 112×112×32
            ConvBlock(32,  64),    # 112×112×32  → 56×56×64
            ConvBlock(64,  128),   # 56×56×64    → 28×28×128
            ConvBlock(128, 256),   # 28×28×128   → 14×14×256
        )

        # Global Average Pooling: 14×14×256 → 1×1×256 → (256,)
        # Why: avoids giant FC layer, forces each channel to be globally meaningful
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),       # Randomly zeros neurons during training
                                            # During MC Dropout inference, keep dropout ON
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)     # Raw logits — no sigmoid here
                                            # BCEWithLogitsLoss applies sigmoid internally
                                            # (numerically more stable than manual sigmoid)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def enable_mc_dropout(self):
        """
        Keeps dropout layers active during inference for Monte Carlo Dropout.
        Call this before running uncertainty estimation forward passes.
        Normally model.eval() disables dropout — this overrides that behavior.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()   # Force dropout ON even in eval mode
