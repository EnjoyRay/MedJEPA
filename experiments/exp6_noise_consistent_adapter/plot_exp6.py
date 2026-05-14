"""Plot Exp6 NCA results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "I-JEPA-H/95": "#2563EB",
    "I-JEPA-H/201": "#0F766E",
    "MAE-H/97": "#7C3AED",
    "MAE-H/300": "#DC2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Exp6 paper figures")
    parser.add_argument("--inputs", nargs="+", required=True, help="MODEL=RESULT_DIR entries")
    parser.add_argument("--exp5_inputs", nargs="*", default=[], help="MODEL=EXP5_DIR entries")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _parse_mapping(entries: list[str]) -> dict[str, Path]:
    out = {}
    for item in entries:
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def _load_nca(inputs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for model, path in inputs.items():
        csv_path = path / "nca_results.csv"
        if not csv_path.exists():
            print(f"[Exp6] missing {csv_path}, skipping")
            continue
        df = pd.read_csv(csv_path)
        df["model"] = model
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No nca_results.csv files found.")
    return pd.concat(frames, ignore_index=True)


def _load_exp5(exp5_inputs: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for model, path in exp5_inputs.items():
        csv_path = path / "mitigation_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["model"] = model
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig_main(nca: pd.DataFrame, exp5: pd.DataFrame, out: Path) -> Path:
    conditions = ["clean", "noise_sigma=0.05", "noise_sigma=0.10"]
    rows = []
    for _, row in nca[(nca["ablation"] == "full") & (nca["condition"].isin(conditions))].iterrows():
        rows.append({"model": row["model"], "method": "NCA", "condition": row["condition"], "macro_auroc": row["macro_auroc"]})
    if not exp5.empty:
        keep = exp5[(exp5["condition"].isin(conditions)) & (
            ((exp5["probe"] == "clean_probe") & (exp5["mitigation"] == "none"))
            | ((exp5["probe"] == "robust_probe") & (exp5["mitigation"] == "train_aug"))
        )].copy()
        keep["method"] = np.where(keep["probe"] == "robust_probe", "Robust probe", "Linear probe")
        rows.extend(keep[["model", "method", "condition", "macro_auroc"]].to_dict("records"))
    df = pd.DataFrame(rows)

    models = list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(1, len(conditions), figsize=(5.2 * len(conditions), 4.2), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    width = 0.22
    methods = ["Linear probe", "Robust probe", "NCA"]
    for ax, condition in zip(axes, conditions):
        sub = df[df["condition"] == condition]
        x = np.arange(len(models))
        for i, method in enumerate(methods):
            vals = []
            for model in models:
                hit = sub[(sub["model"] == model) & (sub["method"] == method)]
                vals.append(float(hit["macro_auroc"].iloc[0]) if len(hit) else np.nan)
            ax.bar(x + (i - 1) * width, vals, width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_title(condition)
        ax.set_ylabel("Macro AUROC")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 17. Noise-Consistent Adapter improves noisy robustness")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = out / "fig17_exp6_nca_main.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_tradeoff(nca: pd.DataFrame, out: Path) -> Path:
    df = nca[nca["ablation"] == "full"].copy()
    clean = df[df["condition"] == "clean"][["model", "macro_auroc"]].rename(columns={"macro_auroc": "clean_auroc"})
    noise = df[df["condition"].isin(["noise_sigma=0.05", "noise_sigma=0.10"])].merge(clean, on="model")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for model, sub in noise.groupby("model"):
        ax.scatter(
            sub["clean_auroc"] - sub["macro_auroc"],
            sub["adapter_cosine_drift"],
            s=95,
            label=model,
            color=COLORS.get(model),
            alpha=0.9,
        )
        for _, row in sub.iterrows():
            ax.annotate(row["condition"].replace("noise_sigma=", "s="), (row["clean_auroc"] - row["macro_auroc"], row["adapter_cosine_drift"]), fontsize=8)
    ax.set_xlabel("AUROC drop from clean")
    ax.set_ylabel("Adapter cosine drift")
    ax.set_title("Figure 18. NCA clean-noisy tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    path = out / "fig18_exp6_nca_tradeoff.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_ablation(nca: pd.DataFrame, out: Path) -> Path:
    condition = "noise_sigma=0.05"
    df = nca[nca["condition"] == condition].copy()
    pivot = df.pivot_table(index="ablation", columns="model", values="macro_auroc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < np.nanmean(pivot.values) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Macro AUROC")
    ax.set_title("Figure 19. Exp6 ablation under Gaussian noise sigma=0.05")
    fig.tight_layout()
    path = out / "fig19_exp6_nca_ablation.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_per_class(inputs: dict[str, Path], out: Path) -> Path | None:
    frames = []
    for model, path in inputs.items():
        pc = path / "nca_per_class.csv"
        if not pc.exists():
            continue
        df = pd.read_csv(pc)
        df["model"] = model
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    clean = df[(df["ablation"] == "full") & (df["condition"] == "clean")][["model", "class_name", "auroc"]].rename(columns={"auroc": "clean_auroc"})
    noise = df[(df["ablation"] == "full") & (df["condition"] == "noise_sigma=0.05")][["model", "class_name", "auroc"]]
    merged = noise.merge(clean, on=["model", "class_name"])
    merged["drop"] = merged["clean_auroc"] - merged["auroc"]
    pivot = merged.pivot_table(index="class_name", columns="model", values="drop", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(pivot.values, cmap="magma_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="Clean - noise AUROC")
    ax.set_title("Figure 20. Per-class residual noise drop after NCA")
    fig.tight_layout()
    path = out / "fig20_exp6_per_class_noise_gain.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = _parse_mapping(args.inputs)
    exp5_inputs = _parse_mapping(args.exp5_inputs)
    nca = _load_nca(inputs)
    exp5 = _load_exp5(exp5_inputs)
    nca.to_csv(out / "table_exp6_nca_results.csv", index=False)
    paths = [fig_main(nca, exp5, out), fig_tradeoff(nca, out), fig_ablation(nca, out)]
    pc_path = fig_per_class(inputs, out)
    if pc_path is not None:
        paths.append(pc_path)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
