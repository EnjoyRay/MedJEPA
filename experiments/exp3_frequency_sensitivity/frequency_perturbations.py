"""Frequency-domain perturbations for Exp3.

All functions expect images in [0, 1] with shape (B, C, H, W) and return
clamped tensors in the same range.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch


Perturbation = Callable[[torch.Tensor], torch.Tensor]


def _radius_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    yy = torch.fft.fftshift(torch.fft.fftfreq(h, device=device)).view(h, 1)
    xx = torch.fft.fftshift(torch.fft.fftfreq(w, device=device)).view(1, w)
    radius = torch.sqrt(xx * xx + yy * yy)
    return radius / radius.max().clamp_min(1e-8)


def _fft_filter(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    freq = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    filtered = freq * mask.view(1, 1, *mask.shape)
    out = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)).real
    return out.clamp(0.0, 1.0)


def low_pass(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Keep frequencies with normalized radius <= cutoff."""
    _, _, h, w = x.shape
    radius = _radius_grid(h, w, x.device)
    return _fft_filter(x, (radius <= cutoff).float())


def high_suppression(x: torch.Tensor, cutoff: float, keep: float) -> torch.Tensor:
    """Suppress frequencies above cutoff by multiplying them with keep."""
    _, _, h, w = x.shape
    radius = _radius_grid(h, w, x.device)
    mask = torch.ones_like(radius)
    mask = torch.where(radius > cutoff, torch.full_like(mask, keep), mask)
    return _fft_filter(x, mask)


def band_corrupt(x: torch.Tensor, low: float, high: float, sigma: float) -> torch.Tensor:
    """Add Gaussian noise only to a frequency band."""
    _, _, h, w = x.shape
    radius = _radius_grid(h, w, x.device)
    band = ((radius >= low) & (radius < high)).float()
    noise = torch.randn_like(x) * sigma
    freq_x = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    freq_n = torch.fft.fftshift(torch.fft.fft2(noise, dim=(-2, -1)), dim=(-2, -1))
    out_freq = freq_x + freq_n * band.view(1, 1, h, w)
    out = torch.fft.ifft2(torch.fft.ifftshift(out_freq, dim=(-2, -1)), dim=(-2, -1)).real
    return out.clamp(0.0, 1.0)


def gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)


def get_frequency_schedule() -> Dict[str, List[Tuple[str, Perturbation]]]:
    """Return fixed Exp3 perturbation schedule."""
    return {
        "low_pass": [
            ("cutoff=0.35", lambda x: low_pass(x, 0.35)),
            ("cutoff=0.25", lambda x: low_pass(x, 0.25)),
            ("cutoff=0.15", lambda x: low_pass(x, 0.15)),
        ],
        "high_suppression": [
            ("cutoff=0.35_keep=0.50", lambda x: high_suppression(x, 0.35, 0.50)),
            ("cutoff=0.25_keep=0.25", lambda x: high_suppression(x, 0.25, 0.25)),
            ("cutoff=0.15_keep=0.00", lambda x: high_suppression(x, 0.15, 0.00)),
        ],
        "band_corrupt": [
            ("band=0.00-0.20", lambda x: band_corrupt(x, 0.00, 0.20, 0.08)),
            ("band=0.20-0.45", lambda x: band_corrupt(x, 0.20, 0.45, 0.08)),
            ("band=0.45-1.00", lambda x: band_corrupt(x, 0.45, 1.00, 0.08)),
        ],
        "gaussian_noise": [
            ("sigma=0.05", lambda x: gaussian_noise(x, 0.05)),
            ("sigma=0.10", lambda x: gaussian_noise(x, 0.10)),
        ],
    }
