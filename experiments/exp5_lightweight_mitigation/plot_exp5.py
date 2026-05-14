"""Build Exp5 paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="model_name=path/to/mitigation_results.csv")
    parser.add_argument("--output_dir", default="results/paper_figures")
    return parser.parse_args()


def _load(items: list[str]) -> pd.DataFrame:
    frames = []
    for item in items:
        name, path = item.split("=", 1)
        df = pd.read_csv(path)
        df["model"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _grouped_bar(ax, df: pd.DataFrame, x_col: str, y_col: str, hue_col: str) -> None:
    order = list(dict.fromkeys(df[x_col].astype(str)))
    hues = list(dict.fromkeys(df[hue_col].astype(str)))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(hues), 1)))
    x = np.arange(len(order))
    width = min(0.8 / max(len(hues), 1), 0.18)
    offsets = (np.arange(len(hues)) - (len(hues) - 1) / 2) * width
    for i, hue in enumerate(hues):
        vals = []
        for item in order:
            s = df[(df[x_col].astype(str) == item) & (df[hue_col].astype(str) == hue)][y_col]
            vals.append(float(s.mean()) if len(s) else np.nan)
        ax.bar(x + offsets[i], vals, width=width, label=hue, color=cmap[i])
    ax.set_xticks(x)
    ax.set_xticklabels(order)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _load(args.inputs)
    df.to_csv(out / "table_exp5_lightweight_mitigation.csv", index=False)

    sub = df[df["condition"].isin(["clean", "noise_sigma=0.05", "noise_sigma=0.10"])].copy()
    sub["method"] = sub["probe"] + " / " + sub["mitigation"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    _grouped_bar(axes[0], sub, "condition", "macro_auroc", "method")
    axes[0].set_title("Lightweight mitigation: AUROC")
    axes[0].set_ylabel("Macro AUROC")
    axes[0].set_xlabel("")
    axes[0].legend(frameon=False, fontsize=7)

    _grouped_bar(axes[1], sub[sub["condition"] != "clean"], "condition", "cosine_drift", "method")
    axes[1].set_title("Representation drift after mitigation")
    axes[1].set_ylabel("Cosine drift vs clean")
    axes[1].set_xlabel("")
    axes[1].legend(frameon=False, fontsize=7)

    fig.suptitle("Figure 16. Exp5 lightweight mitigation for noise sensitivity")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig16_exp5_lightweight_mitigation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(out / "fig16_exp5_lightweight_mitigation.png")


if __name__ == "__main__":
    main()
