"""Build figures for Exp7/Exp8 probe-capacity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODEL_ORDER = ["I-JEPA-H/95", "I-JEPA-H/201", "MAE-H/97", "MAE-H/300"]
METHOD_ORDER = ["Linear", "MLP", "Partial FT"]
COLORS = {"Linear": "#64748B", "MLP": "#2563EB", "Partial FT": "#DC2626"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear_table", default="results/paper_figures/table_exp1_robustness_long.csv")
    parser.add_argument("--exp7_inputs", nargs="+", required=True, help="MODEL=DIR")
    parser.add_argument("--exp8_inputs", nargs="+", required=True, help="MODEL=DIR")
    parser.add_argument("--output_dir", default="results/paper_figures")
    parser.add_argument("--model_order", nargs="*", default=None)
    parser.add_argument("--prefix", default="", help="Optional filename prefix for output artifacts.")
    return parser.parse_args()


def _parse(entries: list[str]) -> dict[str, Path]:
    out = {}
    for item in entries:
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def _normalize_linear(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[(df["perturbation"].isin(["clean", "gaussian_noise"]))].copy()
    keep["method"] = "Linear"
    return keep[["model", "method", "condition", "perturbation", "x_value", "macro_auroc", "auroc_drop", "cosine_drift"]]


def _load_method(inputs: dict[str, Path], filename: str, method: str) -> pd.DataFrame:
    frames = []
    for model, path in inputs.items():
        csv_path = path / filename
        if not csv_path.exists():
            print(f"[probe-capacity] missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["model"] = model
        df["method"] = method
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[["model", "method", "condition", "perturbation", "x_value", "macro_auroc", "auroc_drop", "cosine_drift"]]


def _load_all(args: argparse.Namespace, model_order: list[str]) -> pd.DataFrame:
    frames = [_normalize_linear(pd.read_csv(args.linear_table))]
    exp7 = _load_method(_parse(args.exp7_inputs), "mlp_probe_results.csv", "MLP")
    exp8 = _load_method(_parse(args.exp8_inputs), "partial_ft_results.csv", "Partial FT")
    if not exp7.empty:
        frames.append(exp7)
    if not exp8.empty:
        frames.append(exp8)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["model"].isin(model_order) & df["method"].isin(METHOD_ORDER)]
    df["x_value"] = df["x_value"].astype(float)
    return df


def make_table(df: pd.DataFrame, out: Path, model_order: list[str], prefix: str) -> Path:
    rows = []
    for condition in ["clean", "gaussian_noise/sigma=0.05", "gaussian_noise/sigma=0.10"]:
        row = {"Setting": condition}
        sub = df[df["condition"] == condition]
        for model in model_order:
            for method in METHOD_ORDER:
                hit = sub[(sub["model"] == model) & (sub["method"] == method)]
                row[f"{model} {method}"] = "" if hit.empty else f"{float(hit.iloc[0]['macro_auroc']):.3f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{prefix}table4_probe_capacity_summary.csv", index=False)

    fig_width = max(12, 1.5 + 1.35 * (1 + len(model_order) * len(METHOD_ORDER)))
    fig, ax = plt.subplots(figsize=(fig_width, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.55)
    ax.set_title("Table 4. Probe capacity and partial fine-tuning summary", fontsize=12, pad=12)
    path = out / f"{prefix}table4_probe_capacity_summary.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_clean_noise(df: pd.DataFrame, out: Path, model_order: list[str], prefix: str) -> Path:
    conditions = ["clean", "gaussian_noise/sigma=0.05", "gaussian_noise/sigma=0.10"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    width = 0.22
    for ax, condition in zip(axes, conditions):
        sub = df[df["condition"] == condition]
        x = np.arange(len(model_order))
        for i, method in enumerate(METHOD_ORDER):
            vals = []
            for model in model_order:
                hit = sub[(sub["model"] == model) & (sub["method"] == method)]
                vals.append(float(hit.iloc[0]["macro_auroc"]) if not hit.empty else np.nan)
            ax.bar(x + (i - 1) * width, vals, width, label=method, color=COLORS[method])
        ax.set_title(condition.replace("gaussian_noise/", "noise "))
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=20, ha="right")
        ax.set_ylim(0.45, 0.98)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Macro AUROC")
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 21. Linear vs MLP probe vs partial fine-tuning")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / f"{prefix}fig21_probe_capacity_clean_noise.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_noise_drop(df: pd.DataFrame, out: Path, model_order: list[str], prefix: str) -> Path:
    sub = df[df["perturbation"] == "gaussian_noise"].copy()
    n = len(model_order)
    fig, axes = plt.subplots(1, n, figsize=(max(4.2 * n, 8), 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, model in zip(axes, model_order):
        part = sub[sub["model"] == model]
        for method in METHOD_ORDER:
            line = part[part["method"] == method].sort_values("x_value")
            if line.empty:
                continue
            ax.plot(line["x_value"], line["auroc_drop"], marker="o", label=method, color=COLORS[method])
        ax.set_title(model)
        ax.set_xlabel("Gaussian sigma")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AUROC drop from clean")
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 22. Probe capacity under increasing Gaussian noise")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / f"{prefix}fig22_probe_capacity_noise_drop.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_gap(df: pd.DataFrame, out: Path, prefix: str, model_order: list[str]) -> Path:
    clean = df[df["condition"] == "clean"]
    jepa_model = next((m for m in model_order if m.startswith("I-JEPA-H/300")), None)
    if jepa_model is None:
        jepa_model = next((m for m in model_order if m.startswith("I-JEPA-H/")), "I-JEPA-H/201")
    mae_model = next((m for m in model_order if m == "MAE-H/300"), "MAE-H/300")
    rows = []
    for method in METHOD_ORDER:
        j = clean[(clean["model"] == jepa_model) & (clean["method"] == method)]
        m = clean[(clean["model"] == mae_model) & (clean["method"] == method)]
        if not j.empty and not m.empty:
            rows.append({"method": method, "gap": float(j.iloc[0]["macro_auroc"] - m.iloc[0]["macro_auroc"])})
    gap = pd.DataFrame(rows)
    path = out / f"{prefix}fig23_mae_gap_decomposition.png"
    if gap.empty:
        return path
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    ax.bar(gap["method"], gap["gap"], color=[COLORS[m] for m in gap["method"]])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(f"{jepa_model} AUROC - {mae_model} AUROC")
    ax.set_title("Figure 23. Does stronger adaptation close the MAE gap?")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_order = args.model_order or DEFAULT_MODEL_ORDER
    prefix = args.prefix
    df = _load_all(args, model_order)
    df.to_csv(out / f"{prefix}table_probe_capacity_long.csv", index=False)
    paths = [make_table(df, out, model_order, prefix), fig_clean_noise(df, out, model_order, prefix), fig_noise_drop(df, out, model_order, prefix)]
    if any(m.startswith("I-JEPA-H/") for m in model_order) and "MAE-H/300" in set(model_order):
        paths.append(fig_gap(df, out, prefix, model_order))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
