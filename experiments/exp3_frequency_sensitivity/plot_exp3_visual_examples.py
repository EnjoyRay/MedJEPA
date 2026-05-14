"""Create visual examples for Exp3 frequency perturbations."""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp3_frequency_sensitivity.frequency_perturbations import (
    band_corrupt,
    gaussian_noise,
    high_suppression,
    low_pass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Optional direct path to a CXR image.")
    parser.add_argument("--tar", default="data/VinBigData_ChestXray.tar", help="VinBigData tar archive.")
    parser.add_argument(
        "--member",
        default="VinBigData_ChestXray/images_1024/train/006501b11e04aec2d403177b9ae0f34c.png",
        help="Image member inside the tar archive.",
    )
    parser.add_argument("--output_dir", default="results/paper_figures")
    return parser.parse_args()


def _load_image(args: argparse.Namespace) -> Image.Image:
    if args.image:
        return Image.open(args.image).convert("L")
    with tarfile.open(args.tar, "r") as tar:
        f = tar.extractfile(args.member)
        if f is None:
            raise FileNotFoundError(args.member)
        return Image.open(f).convert("L")


def _to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.resize((224, 224), Image.Resampling.BICUBIC)
    data = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, 1, 224, 224)
    return data / 255.0


def _show(ax, x: torch.Tensor, title: str) -> None:
    ax.imshow(x.squeeze().detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    x = _to_tensor(_load_image(args))

    examples: list[tuple[str, torch.Tensor]] = [
        ("Original", x),
        ("Low-pass\ncutoff 0.25", low_pass(x, 0.25)),
        ("High-frequency suppression\ncutoff 0.25, keep 0.25", high_suppression(x, 0.25, 0.25)),
        ("Band corruption\nlow band 0.00-0.20", band_corrupt(x, 0.00, 0.20, 0.08)),
        ("Band corruption\nmid band 0.20-0.45", band_corrupt(x, 0.20, 0.45, 0.08)),
        ("Band corruption\nhigh band 0.45-1.00", band_corrupt(x, 0.45, 1.00, 0.08)),
        ("Gaussian noise\nsigma 0.05", gaussian_noise(x, 0.05)),
        ("Gaussian noise\nsigma 0.10", gaussian_noise(x, 0.10)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 8.4))
    for ax, (title, img) in zip(axes.flatten(), examples):
        _show(ax, img, title)
    fig.suptitle("Figure 14a. Visual effect of Exp3 frequency perturbations", fontsize=15)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.86, wspace=0.18, hspace=0.36)
    path = out / "fig14a_exp3_frequency_visual_examples.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
