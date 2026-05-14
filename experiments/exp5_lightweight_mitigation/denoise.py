"""Simple denoising and augmentation utilities for Exp5."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F


def gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)


def brightness(x: torch.Tensor, delta: float) -> torch.Tensor:
    return (x + delta).clamp(0.0, 1.0)


def contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return ((x - mean) * factor + mean).clamp(0.0, 1.0)


def gaussian_smooth(x: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.shape[1]).clamp(0.0, 1.0)


def median_filter(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    return patches.contiguous().view(*x.shape, -1).median(dim=-1).values.clamp(0.0, 1.0)


def get_denoise_methods() -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        "none": lambda x: x,
        "gaussian_smooth": lambda x: gaussian_smooth(x, sigma=1.0, kernel_size=5),
        "median_filter": lambda x: median_filter(x, kernel_size=3),
    }


def get_robust_train_augmentations() -> List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    return [
        ("clean", lambda x: x),
        ("noise_sigma=0.05", lambda x: gaussian_noise(x, 0.05)),
        ("brightness=-0.10", lambda x: brightness(x, -0.10)),
        ("brightness=+0.10", lambda x: brightness(x, +0.10)),
        ("contrast=0.50", lambda x: contrast(x, 0.50)),
        ("contrast=1.50", lambda x: contrast(x, 1.50)),
    ]


def get_eval_conditions() -> List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    return [
        ("clean", lambda x: x),
        ("noise_sigma=0.05", lambda x: gaussian_noise(x, 0.05)),
        ("noise_sigma=0.10", lambda x: gaussian_noise(x, 0.10)),
        ("brightness=-0.20", lambda x: brightness(x, -0.20)),
        ("brightness=+0.20", lambda x: brightness(x, +0.20)),
        ("contrast=0.50", lambda x: contrast(x, 0.50)),
        ("contrast=1.50", lambda x: contrast(x, 1.50)),
    ]
