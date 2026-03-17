"""
vit.py — Vision Transformer (ViT) for multi-label classification
================================================================
Purpose: Attention-based alternative to CNN. Treats the image as a
         sequence of patches and uses self-attention to model global
         relationships between distant regions of the X-ray.

Why ViT vs CNN?
  CNNs: excellent at capturing LOCAL features (edges, textures, small nodules)
        via sliding convolutional filters. But a filter at position (0,0) cannot
        directly "see" what is at position (200,200) — deep layers eventually
        aggregate this but it's indirect.

  ViT:  splits image into patches (e.g., 16×16 pixels each = 196 patches for 224px).
        Each patch attends to ALL other patches simultaneously via self-attention.
        This means a patch in the upper-left lung can directly influence the
        prediction based on what it sees in the lower-right — useful for global
        pathologies like Cardiomegaly (enlarged heart affecting overall structure).

  In practice on medical imaging:
    CNNs tend to win on small local findings (Nodule, Mass)
    ViT tends to win on global structural patterns (Cardiomegaly, Effusion)
    Ensemble wins overall — captures both
"""

import torch
import torch.nn as nn

try:
    import timm   # PyTorch Image Models — best library for ViT variants
except ImportError:
    raise ImportError("Install timm: pip install timm")


class ViTClassifier(nn.Module):
    """
    Vision Transformer for multi-label chest X-ray classification.

    Uses timm's pretrained ViT which is pretrained on ImageNet-21k
    then fine-tuned on ImageNet-1k — a much larger pretraining than DenseNet.

    What to change:
      - model_name: swap to a different ViT variant
          "vit_small_patch16_224"  → smaller, faster, less VRAM (good for debugging)
          "vit_base_patch16_224"   → standard (current)
          "vit_large_patch16_224"  → more capacity, needs more VRAM (may OOM on 4060)
          "vit_base_patch32_224"   → larger patches = fewer tokens = faster but less detail

      - dropout_rate: ViT uses lower dropout than CNN (0.1 is typical)
        Increase to 0.2-0.3 if overfitting

      - patch size (in model_name): patch16 = 16×16 pixel patches = 196 tokens
        patch32 = 32×32 pixel patches = 49 tokens (much faster, less detail)
    """
    def __init__(self,
                 num_classes: int = 14,
                 model_name: str = "vit_base_patch16_224",
                 dropout_rate: float = 0.1):
        super().__init__()

        # Load pretrained ViT from timm
        # pretrained=True downloads ImageNet weights
        # num_classes=0 removes the original classification head
        # so we can attach our own 14-class head
        self.vit = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0       # Remove original head, we attach our own below
        )

        # ViT base outputs a 768-dimensional embedding (the [CLS] token)
        # Why CLS token? ViT prepends a learnable [CLS] token to the patch sequence.
        # After all attention layers, this token aggregates global image information.
        # It's then passed to the classifier — similar to BERT's [CLS] in NLP.
        # If you switch to vit_small, this becomes 384. vit_large = 1024.
        embed_dim = self.vit.num_features   # Automatically gets correct dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),        # ViT uses LayerNorm, not BatchNorm
                                            # Why: BatchNorm behaves poorly on sequences
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # vit() returns the [CLS] token embedding (batch_size, embed_dim)
        features = self.vit(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """
        Freeze ViT transformer blocks — only train classification head.
        Useful for first few epochs to stabilize the head before full fine-tuning.
        """
        for param in self.vit.parameters():
            param.requires_grad = False
        print("ViT backbone frozen — only classifier training")

    def unfreeze_backbone(self):
        """
        Unfreeze all ViT layers for full fine-tuning.
        Use a very low LR (1e-5) — ViT is sensitive to LR during fine-tuning.
        """
        for param in self.vit.parameters():
            param.requires_grad = True
        print("ViT backbone unfrozen — full fine-tuning active")

    def enable_mc_dropout(self):
        """
        Enable Monte Carlo Dropout at inference time.
        ViT uses dropout inside attention layers — enabling those gives
        uncertainty estimates over the attention patterns themselves.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
