"""Build Exp4 paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = {
    "I-JEPA-H/95": "#2563EB",
    "I-JEPA-H/201": "#0F766E",
    "MAE-H/97": "#DC2626",
    "MAE-H/300": "#7C2D12",
}


def _palette(df: pd.DataFrame) -> dict[str, str] | None:
    models = set(df["model"].dropna().astype(str))
    if models.issubset(MODEL_COLORS):
        return {model: MODEL_COLORS[model] for model in models}
    return None


def _grouped_bar(ax, df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, order: list[str] | None = None) -> None:
    if order is None:
        order = list(dict.fromkeys(df[x_col].astype(str)))
    hues = list(dict.fromkeys(df[hue_col].astype(str)))
    colors = _palette(df) or {}
    fallback = plt.cm.Set2(np.linspace(0, 1, max(len(hues), 1)))
    x = np.arange(len(order))
    width = min(0.8 / max(len(hues), 1), 0.26)
    offsets = (np.arange(len(hues)) - (len(hues) - 1) / 2) * width
    for i, hue in enumerate(hues):
        vals = []
        for item in order:
            s = df[(df[x_col].astype(str) == item) & (df[hue_col].astype(str) == hue)][y_col]
            vals.append(float(s.mean()) if len(s) else np.nan)
        ax.bar(x + offsets[i], vals, width=width, label=hue, color=colors.get(hue, fallback[i]))
    ax.set_xticks(x)
    ax.set_xticklabels(order)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="model_name=path/to/mechanism_per_sample.csv")
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


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _load(args.inputs)
    df.to_csv(out / "table_exp4_mechanism_per_sample.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    _grouped_bar(axes[0], df, "model", "inside_saliency_ratio", "disease_group")
    axes[0].set_title("Saliency inside target bbox")
    axes[0].set_ylabel("Inside-bbox saliency ratio")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False, fontsize=8)

    drift = df.melt(
        id_vars=["model", "disease_group"],
        value_vars=["noise_drift_mean", "lesion_drift_mean", "control_drift_mean"],
        var_name="condition",
        value_name="token_drift",
    )
    _grouped_bar(axes[1], drift, "condition", "token_drift", "model")
    axes[1].set_title("Token drift by perturbation")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean token cosine drift")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Figure 15. Exp4 token drift and saliency-bbox alignment")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig15_exp4_mechanism_alignment.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(out / "fig15_exp4_mechanism_alignment.png")


if __name__ == "__main__":
    main()
