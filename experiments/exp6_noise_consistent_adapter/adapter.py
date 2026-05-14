"""Adapter modules for Exp6 noise-consistent adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualAdapterClassifier(nn.Module):
    """Residual MLP adapter followed by a multi-label classifier."""

    def __init__(self, feat_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )
        self.norm = nn.LayerNorm(feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.adapter(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.adapt(x)
        return self.classifier(z), z
