"""
densenet.py — DenseNet-121 with Transfer Learning (CheXNet approach)
====================================================================
Purpose: Fine-tune a DenseNet-121 pretrained on ImageNet for NIH chest X-ray
         multi-label classification. This is the direct replication of CheXNet.

Why DenseNet-121 specifically?
  DenseNet = Dense Connections. Every layer receives feature maps from ALL
  previous layers, not just the one before it. This means:
    - Stronger gradient flow during backprop (no vanishing gradient)
    - Feature reuse: early low-level features (edges) stay accessible to
      deep layers (which normally lose them in regular CNNs)
    - Fewer parameters than ResNet for the same depth

Why Transfer Learning?
  ImageNet-pretrained weights already encode rich visual features:
  edges, textures, shapes. Fine-tuning adapts these to X-ray features
  much faster than training from scratch. Without pretrained weights,
  you'd need far more data and compute.

Fine-tuning strategy:
  Phase 1 (freeze_epochs): Freeze all DenseNet layers, only train
    the new classification head. This prevents pretrained weights
    from being destroyed before the head learns basic features.
  Phase 2 (unfreeze): Unfreeze all layers and train end-to-end
    with a low learning rate so pretrained features adjust gently.
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNet121(nn.Module):
    """
    DenseNet-121 with a custom 14-class multi-label head.

    What to change:
      - dropout_rate: increase if overfitting (val AUC drops while train AUC rises)
      - Replace DenseNet121 with DenseNet169 or DenseNet201 for more capacity
        (will require more VRAM and training time)
      - freeze_backbone=True/False: set via freeze() method below
    """
    def __init__(self, num_classes: int = 14, dropout_rate: float = 0.5):
        super().__init__()

        # Load DenseNet-121 pretrained on ImageNet
        # weights=IMAGENET1K_V1 downloads pretrained weights automatically
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Keep everything except the original classifier
        # densenet.features = all convolutional blocks
        self.features = densenet.features

        # DenseNet-121 outputs 1024 feature maps before pooling
        # Why to change: if you switch to DenseNet169, this becomes 1664
        in_features = 1024

        # New classification head for 14-class multi-label output
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # Global Average Pooling: H×W×1024 → 1×1×1024
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)   # Raw logits, no sigmoid
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        """
        Freeze all pretrained DenseNet layers — only classifier trains.
        Call this for the first few epochs (freeze_epochs in config).
        Why: protects pretrained weights from being overwritten before
        the head stabilizes. If you skip this, you may destroy ImageNet features.
        """
        for param in self.features.parameters():
            param.requires_grad = False
        print("DenseNet backbone frozen — only classifier training")

    def unfreeze_backbone(self):
        """
        Unfreeze all layers for full end-to-end fine-tuning.
        Call this after freeze_epochs. Use a low learning rate (1e-5)
        so pretrained features adjust gradually instead of being overwritten.
        """
        for param in self.features.parameters():
            param.requires_grad = True
        print("DenseNet backbone unfrozen — full fine-tuning active")

    def enable_mc_dropout(self):
        """
        Activates dropout during inference for Monte Carlo uncertainty estimation.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
