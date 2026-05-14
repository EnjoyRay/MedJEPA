"""
Nuisance perturbations for Exp1 (robustness experiment).

Each perturbation is a callable that takes a (B, C, H, W) float32 tensor
(already normalized to [0,1] or ImageNet-normalized) and returns a tensor
of the same shape.

Usage:
    from shared.perturbations import get_perturbation_schedule, apply_perturbation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Individual perturbation functions
# ---------------------------------------------------------------------------

def gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add i.i.d. Gaussian noise N(0, sigma^2) to the image."""
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(0., 1.)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur with the given sigma (kernel computed on the fly)."""
    # Kernel size: 2 * ceil(3*sigma) + 1, minimum 3
    ks = max(3, 2 * int(np.ceil(3 * sigma)) + 1)
    if ks % 2 == 0:
        ks += 1

    # Build 2-D Gaussian kernel
    coords = torch.arange(ks, dtype=torch.float32, device=x.device) - ks // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g /= g.sum()
    kernel_2d = g[:, None] * g[None, :]                       # (ks, ks)
    kernel_2d = kernel_2d.expand(x.shape[1], 1, ks, ks)      # (C, 1, ks, ks)

    padding = ks // 2
    blurred = F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])
    return blurred.clamp(0., 1.)


def brightness_shift(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Shift pixel brightness by delta (positive = brighter)."""
    return (x + delta).clamp(0., 1.)


def contrast_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale contrast around mid-grey (0.5). factor > 1 increases contrast."""
    return ((x - 0.5) * factor + 0.5).clamp(0., 1.)


# ---------------------------------------------------------------------------
# Perturbation schedule: name → list of (label, callable)
# ---------------------------------------------------------------------------

PERTURBATION_SCHEDULE: Dict[str, List[Tuple[str, callable]]] = {
    'gaussian_noise': [
        ('sigma=0.05', lambda x: gaussian_noise(x, 0.05)),
        ('sigma=0.10', lambda x: gaussian_noise(x, 0.10)),
        ('sigma=0.20', lambda x: gaussian_noise(x, 0.20)),
        ('sigma=0.30', lambda x: gaussian_noise(x, 0.30)),
    ],
    'gaussian_blur': [
        ('sigma=0.5', lambda x: gaussian_blur(x, 0.5)),
        ('sigma=1.0', lambda x: gaussian_blur(x, 1.0)),
        ('sigma=2.0', lambda x: gaussian_blur(x, 2.0)),
        ('sigma=3.0', lambda x: gaussian_blur(x, 3.0)),
    ],
    'brightness': [
        ('delta=+0.1', lambda x: brightness_shift(x, +0.10)),
        ('delta=+0.2', lambda x: brightness_shift(x, +0.20)),
        ('delta=-0.1', lambda x: brightness_shift(x, -0.10)),
        ('delta=-0.2', lambda x: brightness_shift(x, -0.20)),
    ],
    'contrast': [
        ('factor=1.5', lambda x: contrast_scale(x, 1.5)),
        ('factor=2.0', lambda x: contrast_scale(x, 2.0)),
        ('factor=0.5', lambda x: contrast_scale(x, 0.5)),
        ('factor=0.25', lambda x: contrast_scale(x, 0.25)),
    ],
}


def get_perturbation_schedule(
    types: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[str, callable]]]:
    """
    Return the perturbation schedule.

    Args:
        types: list of perturbation type names to include.
               If None, all types are returned.
    Returns:
        dict mapping perturbation_type -> [(label, fn), ...]
    """
    if types is None:
        return PERTURBATION_SCHEDULE
    return {k: v for k, v in PERTURBATION_SCHEDULE.items() if k in types}
