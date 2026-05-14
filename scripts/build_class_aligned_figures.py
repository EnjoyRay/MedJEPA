"""Build paper figures for the redesigned class-aligned Exp2b."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_figures"

RUNS = {
    "I-JEPA-H/95": RESULTS / "exp2b_class_aligned_ijepa_vith_ep95_fixedsplit",
    "I-JEPA-H/201": RESULTS / "exp2b_class_aligned_ijepa_vith_ep201_fixedsplit",
    "MAE-H/97": RESULTS / "exp2b_class_aligned_mae_huge_mimic_97ep_fixedsplit",
    "MAE-H/300": RESULTS / "exp2b_class_aligned_mae_huge_mimic_300ep_fixedsplit",
}

MODEL_COLORS = {
    "I-JEPA-H/95": "#2563EB",
    "I-JEPA-H/201": "#0F766E",
    "MAE-H/97": "#DC2626",
    "MAE-H/300": "#7C2D12",
}
AREA_BIN_ORDER = ["<=5%", "5-15%", "15-35%", "35-60%", ">60%"]


def _existing_runs() -> dict[str, Path]:
    return {name: path for name, path in RUNS.items() if (path / "class_aligned_overall.csv").exists()}


def setup() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    OUT.mkdir(parents=True, exist_ok=True)


def load_overall(runs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for model, path in runs.items():
        df = pd.read_csv(path / "class_aligned_overall.csv")
        row = df.iloc[0].to_dict()
        row["model"] = model
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table_exp2b_class_aligned_overall.csv", index=False)
    return out


def load_groups(runs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for model, path in runs.items():
        df = pd.read_csv(path / "class_aligned_summary_by_group.csv")
        df["model"] = model
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "table_exp2b_class_aligned_groups.csv", index=False)
    return out


def load_classes(runs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for model, path in runs.items():
        df = pd.read_csv(path / "class_aligned_summary_by_class.csv")
        df["model"] = model
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "table_exp2b_class_aligned_classes.csv", index=False)
    return out


def load_samples(runs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for model, path in runs.items():
        sample = path / "class_aligned_analysis" / "class_aligned_per_sample_with_bins.csv"
        if not sample.exists():
            sample = path / "class_aligned_per_sample.csv"
        df = pd.read_csv(sample)
        df["model"] = model
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "table_exp2b_class_aligned_samples.csv", index=False)
    return out


def build_overall(overall: pd.DataFrame) -> Path:
    order = [m for m in RUNS if m in set(overall["model"])]
    df = overall.set_index("model").loc[order].reset_index()
    x = np.arange(len(df))
    y = df["delta_logit_drop_mean"].to_numpy()
    err = np.vstack([
        y - df["delta_logit_ci95_low"].to_numpy(),
        df["delta_logit_ci95_high"].to_numpy() - y,
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].bar(x - 0.18, df["target_logit_drop_mean"], 0.36, label="Lesion", color="#7C3AED")
    axes[0].bar(x + 0.18, df["control_logit_drop_mean"], 0.36, label="Control", color="#F97316")
    axes[0].set_xticks(x, df["model"], rotation=25, ha="right")
    axes[0].set_ylabel("Target-class logit drop")
    axes[0].set_title("Target disease score drop")
    axes[0].legend(frameon=False)

    axes[1].errorbar(x, y, yerr=err, fmt="o", color="#111827", capsize=4)
    for i, row in df.iterrows():
        axes[1].scatter(i, row["delta_logit_drop_mean"], s=70, color=MODEL_COLORS[row["model"]], zorder=3)
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_xticks(x, df["model"], rotation=25, ha="right")
    axes[1].set_ylabel("Lesion - matched control")
    axes[1].set_title("Class-aligned lesion-specific effect")
    fig.suptitle("Figure 9. Exp2b class-aligned lesion evidence test")
    fig.text(
        0.5,
        0.015,
        "Note: logit drop = original - occluded; negative drop means the target logit increased after occlusion. "
        "Delta is a relative lesion-control effect, not an absolute 'better' score.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    path = OUT / "fig9_exp2b_class_aligned_overall.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_groups(groups: pd.DataFrame) -> Path:
    order = ["localized", "diffuse_or_global", "other"]
    models = [m for m in RUNS if m in set(groups["model"])]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    x = np.arange(len(order))
    offsets = np.linspace(-0.28, 0.28, max(len(models), 1))
    for offset, model in zip(offsets, models):
        sub = groups[groups["model"] == model].set_index("group").reindex(order).reset_index()
        ax.errorbar(
            x + offset,
            sub["delta_logit_drop_mean"],
            yerr=np.vstack([
                sub["delta_logit_drop_mean"] - sub["delta_logit_ci95_low"],
                sub["delta_logit_ci95_high"] - sub["delta_logit_drop_mean"],
            ]),
            fmt="o",
            capsize=3,
            color=MODEL_COLORS[model],
            label=model,
        )
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x, order)
    ax.set_ylabel("Delta target-class logit drop")
    ax.set_title("Figure 10. Localized and diffuse findings behave differently")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUT / "fig10_exp2b_group_effects.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_class_heatmap(classes: pd.DataFrame) -> Path:
    pivot = classes.pivot(index="group", columns="model", values="delta_logit_drop_mean")
    pivot = pivot[[m for m in RUNS if m in pivot.columns]]
    pivot = pivot.reindex(pivot.abs().mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(8.0, 7.6))
    vmax = max(0.05, float(np.nanmax(np.abs(pivot.to_numpy()))))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Delta target-class logit drop"},
        ax=ax,
    )
    ax.set_title("Figure 11. Class-aligned lesion effects by disease class")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    path = OUT / "fig11_exp2b_class_heatmap.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_area(samples: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for model, sub in samples.groupby("model"):
        axes[0].scatter(
            sub["bbox_area_frac"],
            sub["delta_logit_drop"],
            s=11,
            alpha=0.25,
            color=MODEL_COLORS.get(model, "#6B7280"),
            label=model,
            edgecolors="none",
        )
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_xlabel("Bbox area fraction")
    axes[0].set_ylabel("Delta target-class logit drop")
    axes[0].set_title("Sample-level area response")
    axes[0].legend(frameon=False, fontsize=7)

    if "bbox_area_bin" not in samples.columns:
        bins = [-1e-9, 0.05, 0.15, 0.35, 0.60, 1.0]
        samples = samples.copy()
        samples["bbox_area_bin"] = pd.cut(samples["bbox_area_frac"], bins=bins, labels=AREA_BIN_ORDER)
    else:
        samples = samples.copy()
        samples["bbox_area_bin"] = pd.Categorical(samples["bbox_area_bin"], categories=AREA_BIN_ORDER, ordered=True)
    summary = samples.groupby(["model", "bbox_area_bin"], observed=False)["delta_logit_drop"].mean().reset_index()
    summary["bbox_area_bin"] = pd.Categorical(summary["bbox_area_bin"], categories=AREA_BIN_ORDER, ordered=True)
    summary = summary.sort_values(["model", "bbox_area_bin"])
    sns.lineplot(
        data=summary,
        x="bbox_area_bin",
        y="delta_logit_drop",
        hue="model",
        marker="o",
        palette=MODEL_COLORS,
        sort=False,
        ax=axes[1],
    )
    axes[1].set_xticks(range(len(AREA_BIN_ORDER)))
    axes[1].set_xticklabels(AREA_BIN_ORDER)
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_xlabel("Bbox area bin")
    axes[1].set_ylabel("Mean delta logit drop")
    axes[1].set_title("Binned area effect")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Figure 12. Exp2b separates lesion evidence from mask area")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "fig12_exp2b_area_response.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_case_sheet(runs: dict[str, Path]) -> Path | None:
    images = []
    for model, path in runs.items():
        case_dir = path / "class_aligned_cases"
        candidates = sorted(case_dir.glob("case_*.png"))
        if candidates:
            images.append((model, candidates[0]))
    if not images:
        return None
    cols = min(2, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.8 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax, (model, image_path) in zip(axes.ravel(), images):
        ax.imshow(Image.open(image_path))
        ax.set_title(model)
        ax.axis("off")
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.suptitle("Figure 13. Exp2b representative class-aligned occlusion cases")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "fig13_exp2b_case_sheet.png"
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    setup()
    runs = _existing_runs()
    if not runs:
        raise FileNotFoundError("No class-aligned Exp2b results found.")
    paths = [
        build_overall(load_overall(runs)),
        build_groups(load_groups(runs)),
        build_class_heatmap(load_classes(runs)),
        build_area(load_samples(runs)),
    ]
    case = build_case_sheet(runs)
    if case is not None:
        paths.append(case)
    manifest = ["# Exp2b Class-Aligned Figure Manifest", ""]
    for path in paths:
        manifest.append(f"- `{path.relative_to(ROOT)}`")
    (OUT / "exp2b_figure_manifest.md").write_text("\n".join(manifest), encoding="utf-8")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
