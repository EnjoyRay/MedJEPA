"""Build Exp3 paper figures from frequency_sensitivity.csv outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


MODEL_COLORS = {
    "I-JEPA-H/95": "#2563EB",
    "I-JEPA-H/201": "#0F766E",
    "MAE-H/97": "#DC2626",
    "MAE-H/300": "#7C2D12",
}


PERTURBATION_TITLES = {
    "low_pass": "Low-pass filtering\n(remove high frequency detail)",
    "high_suppression": "High-frequency suppression\n(progressively weaken high frequencies)",
    "band_corrupt": "Band corruption\n(corrupt selected frequency bands)",
    "gaussian_noise": "Gaussian noise\n(pixel-domain high-frequency noise)",
}

PERTURBATION_ORDER = ["low_pass", "high_suppression", "band_corrupt", "gaussian_noise"]


def _palette(df: pd.DataFrame) -> dict[str, str] | None:
    models = set(df["model"].dropna().astype(str))
    if models.issubset(MODEL_COLORS):
        return {model: MODEL_COLORS[model] for model in models}
    return None


def _pretty_level(level: str) -> str:
    level = str(level)
    return (
        level.replace("cutoff=", "cutoff ")
        .replace("_keep=", "\nkeep ")
        .replace("band=", "band ")
        .replace("sigma=", "sigma ")
    )


def _grouped_bar(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    order: list[str],
    show_legend: bool = False,
) -> None:
    hues = list(dict.fromkeys(df[hue_col].astype(str)))
    colors = _palette(df) or {}
    x = np.arange(len(order))
    width = min(0.76 / max(len(hues), 1), 0.18)
    offsets = (np.arange(len(hues)) - (len(hues) - 1) / 2) * width
    for i, hue in enumerate(hues):
        vals = []
        for item in order:
            s = df[(df[x_col] == item) & (df[hue_col].astype(str) == hue)][y_col]
            vals.append(float(s.mean()) if len(s) else np.nan)
        ax.bar(x + offsets[i], vals, width=width, label=hue, color=colors.get(hue))
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    if show_legend:
        ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="model_name=path/to/frequency_sensitivity.csv")
    parser.add_argument("--output_dir", default="results/paper_figures")
    return parser.parse_args()


def _load(items: list[str]) -> pd.DataFrame:
    frames = []
    for item in items:
        name, path = item.split("=", 1)
        df = pd.read_csv(path)
        if "repeat_count" in df.columns and "repeat" in df.columns:
            repeat_text = df["repeat"].astype(str)
            df = df[(repeat_text == "mean") | (df["repeat_count"].fillna(1).astype(float) <= 1)].copy()
        df["model"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _setting_label(row: pd.Series) -> str:
    perturbation = str(row["perturbation"])
    level = str(row["level"])
    if perturbation == "clean":
        return "Clean"
    if perturbation == "low_pass":
        return "Low-pass\n" + _pretty_level(level)
    if perturbation == "high_suppression":
        return "High-suppression\n" + _pretty_level(level)
    if perturbation == "band_corrupt":
        return "Band corrupt\n" + _pretty_level(level)
    if perturbation == "gaussian_noise":
        return "Gaussian noise\n" + _pretty_level(level)
    return perturbation + "\n" + _pretty_level(level)


def _make_complete_table(df: pd.DataFrame, out: Path) -> None:
    model_order = [model for model in MODEL_COLORS if model in set(df["model"])]
    condition_order = (
        ["clean"]
        + [f"low_pass/{level}" for level in ["cutoff=0.35", "cutoff=0.25", "cutoff=0.15"]]
        + [
            f"high_suppression/{level}"
            for level in ["cutoff=0.35_keep=0.50", "cutoff=0.25_keep=0.25", "cutoff=0.15_keep=0.00"]
        ]
        + [f"band_corrupt/{level}" for level in ["band=0.00-0.20", "band=0.20-0.45", "band=0.45-1.00"]]
        + [f"gaussian_noise/{level}" for level in ["sigma=0.05", "sigma=0.10"]]
    )

    rows = []
    best_by_row: list[str | None] = []
    for condition in condition_order:
        part = df[df["condition"] == condition]
        if part.empty:
            continue
        first = part.iloc[0]
        row = {"Setting": _setting_label(first)}
        best_model = None
        best_auroc = -float("inf")
        for model in model_order:
            s = part[part["model"] == model]
            if not s.empty and float(s.iloc[0]["macro_auroc"]) > best_auroc:
                best_auroc = float(s.iloc[0]["macro_auroc"])
                best_model = model
        for model in model_order:
            s = part[part["model"] == model]
            if s.empty:
                row[model] = ""
                continue
            r = s.iloc[0]
            if condition == "clean":
                row[model] = f"{r['macro_auroc']:.3f}"
            else:
                auroc_std = r.get("macro_auroc_std", np.nan)
                drop_std = r.get("auroc_drop_std", np.nan)
                if pd.notna(auroc_std):
                    row[model] = (
                        f"{r['macro_auroc']:.3f}±{auroc_std:.3f}\n"
                        f"(drop {r['auroc_drop']:.3f}±{drop_std:.3f}; d {r['cosine_drift']:.2f})"
                    )
                else:
                    row[model] = f"{r['macro_auroc']:.3f}\n(drop {r['auroc_drop']:.3f}; d {r['cosine_drift']:.2f})"
        rows.append(row)
        best_by_row.append(best_model)

    table_df = pd.DataFrame(rows)
    table_df.to_csv(out / "table_exp3_complete_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.axis("off")
    ax.set_title("Table 3. Complete Exp3 frequency sensitivity summary", fontsize=16, pad=18)
    subtitle = "Cells show Macro AUROC; parentheses show AUROC drop from clean and cosine drift d."
    ax.text(0.5, 0.965, subtitle, ha="center", va="center", transform=ax.transAxes, fontsize=10)

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.01, 0.02, 0.98, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(cell.get_height() * 1.18)
        else:
            cell.set_facecolor("#F9FAFB" if row % 2 == 0 else "white")
            if col == 0:
                cell.set_text_props(weight="bold", ha="left")
                cell.set_facecolor("#EEF2FF" if row % 2 == 0 else "#F5F7FF")
            elif col > 0 and best_by_row[row - 1] == table_df.columns[col]:
                cell.set_facecolor("#FEF3C7")
                cell.set_text_props(weight="bold")
    fig.savefig(out / "table3_exp3_complete_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _load(args.inputs)
    df.to_csv(out / "table_exp3_frequency_sensitivity.csv", index=False)
    _make_complete_table(df, out)

    sub = df[df["condition"] != "clean"].copy()
    sub["display"] = sub["level"].map(_pretty_level)

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), sharey=True)
    axes = axes.flatten()
    for ax, perturbation in zip(axes, PERTURBATION_ORDER):
        part = sub[sub["perturbation"] == perturbation].copy()
        order = part.drop_duplicates("display")["display"].tolist()
        _grouped_bar(ax, part, "display", "auroc_drop", "model", order, show_legend=perturbation == "low_pass")
        ax.set_title(PERTURBATION_TITLES.get(perturbation, perturbation), fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("Macro AUROC drop" if ax in (axes[0], axes[2]) else "")
        ax.tick_params(axis="x", labelrotation=0, labelsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(0.36, float(sub["auroc_drop"].max()) * 1.15))

    fig.suptitle("Figure 14. Exp3 frequency sensitivity: AUROC drop by perturbation type", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig14_exp3_frequency_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 7.6))

    markers = {"low_pass": "o", "high_suppression": "s", "band_corrupt": "^", "gaussian_noise": "D"}
    colors = _palette(sub) or {}
    for (model, perturbation), part in sub.groupby(["model", "perturbation"]):
        ax.scatter(
            part["cosine_drift"],
            part["auroc_drop"],
            marker=markers.get(str(perturbation), "o"),
            s=95,
            color=colors.get(str(model)),
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
        )
    ax.set_title("Exp3: functional drop vs representation drift", fontsize=15)
    ax.set_xlabel("Cosine drift")
    ax.set_ylabel("Macro AUROC drop")
    ax.grid(alpha=0.25)
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=9, label=model, color=color)
        for model, color in MODEL_COLORS.items()
        if model in set(sub["model"])
    ]
    perturbation_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[name],
            linestyle="",
            markersize=9,
            label=name.replace("_", " "),
            color="#4B5563",
        )
        for name in PERTURBATION_ORDER
        if name in set(sub["perturbation"])
    ]
    first = ax.legend(handles=model_handles, title="Model", frameon=False, fontsize=9, title_fontsize=10, loc="upper left")
    ax.add_artist(first)
    ax.legend(
        handles=perturbation_handles,
        title="Perturbation",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.72),
    )
    fig.tight_layout()
    fig.savefig(out / "fig14b_exp3_drop_drift_scatter.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(out / "fig14_exp3_frequency_sensitivity.png")
    print(out / "fig14b_exp3_drop_drift_scatter.png")
    print(out / "table3_exp3_complete_summary.png")


if __name__ == "__main__":
    main()
