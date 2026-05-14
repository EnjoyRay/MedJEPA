"""Build paper-style figures from the completed JEPA/MAE experiments.

Outputs are written to results/paper_figures.  The script only reads existing
experiment CSVs/PNGs, so it is safe to rerun after new checkpoints finish.
"""

from __future__ import annotations

import math
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_figures"

MODEL_COLORS = {
    "I-JEPA-L": "#6B7280",
    "I-JEPA-H/95": "#2563EB",
    "I-JEPA-H/201": "#0F766E",
    "MAE-H/97": "#DC2626",
    "MAE-H/300": "#7C2D12",
}

PERT_COLORS = {
    "gaussian_noise": "#B91C1C",
    "gaussian_blur": "#2563EB",
    "brightness": "#D97706",
    "contrast": "#059669",
}

def result_dir(*names: str) -> Path:
    """Prefer fixed-split thesis results, falling back to legacy folders."""
    for name in names:
        path = RESULTS / name
        if path.exists():
            return path
    return RESULTS / names[0]


EXP1_RUNS = {
    "I-JEPA-H/95": result_dir("exp1_ijepa_vith_ep95_fixedsplit", "exp1_ijepa_vith_epoch97"),
    "I-JEPA-H/201": result_dir("exp1_ijepa_vith_ep201_fixedsplit"),
    "MAE-H/97": result_dir("exp1_mae_huge_mimic_97ep_fixedsplit", "exp1_mae_huge_mimic_97ep"),
    "MAE-H/300": result_dir("exp1_mae_huge_mimic_300ep_fixedsplit"),
}

EXP2_SUMMARY_RUNS = {
    "I-JEPA-H/95": result_dir(
        "exp2_ijepa_vith_ep95_fixedsplit",
        "exp2_ijepa_vith_epoch97_per_sample",
        "exp2_ijepa_vith_epoch97",
    ),
    "I-JEPA-H/201": result_dir("exp2_ijepa_vith_ep201_fixedsplit", "exp2_ijepa_vith_ep201_per_sample"),
    "MAE-H/97": result_dir("exp2_mae_huge_mimic_97ep_fixedsplit", "exp2_mae_huge_mimic_97ep"),
    "MAE-H/300": result_dir("exp2_mae_huge_mimic_300ep_fixedsplit"),
}

EXP2_PER_SAMPLE_RUNS = {
    "I-JEPA-H/95": result_dir("exp2_ijepa_vith_ep95_fixedsplit", "exp2_ijepa_vith_epoch97_per_sample"),
    "I-JEPA-H/201": result_dir("exp2_ijepa_vith_ep201_fixedsplit", "exp2_ijepa_vith_ep201_per_sample"),
    "MAE-H/97": result_dir("exp2_mae_huge_mimic_97ep_fixedsplit", "exp2_mae_huge_mimic_97ep"),
    "MAE-H/300": result_dir("exp2_mae_huge_mimic_300ep_fixedsplit"),
}

PERT_CONFIG = {
    "gaussian_noise": {
        "title": "Gaussian noise",
        "xlabel": "sigma",
        "clean_x": 0.0,
        "xlim": (-0.02, 0.32),
    },
    "gaussian_blur": {
        "title": "Gaussian blur",
        "xlabel": "sigma",
        "clean_x": 0.0,
        "xlim": (-0.15, 3.15),
    },
    "brightness": {
        "title": "Brightness shift",
        "xlabel": "delta",
        "clean_x": 0.0,
        "xlim": (-0.23, 0.23),
    },
    "contrast": {
        "title": "Contrast scaling",
        "xlabel": "factor",
        "clean_x": 1.0,
        "xlim": (0.15, 2.1),
    },
}

PERTURBATION_SAMPLE_ID = "d3637a1935a905b3c326af31389cb846"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def load_demo_image() -> np.ndarray:
    """Load one raw VinBigData CXR image for visualizing perturbations."""
    image_rel = Path("VinBigData_ChestXray") / "images_1024" / "train" / f"{PERTURBATION_SAMPLE_ID}.png"
    candidates = [
        ROOT / "data" / image_rel,
        ROOT.parent / "data" / image_rel,
    ]
    extracted = next((path for path in candidates if path.exists()), None)
    if extracted is not None:
        image = Image.open(extracted).convert("L")
    else:
        tar_path = ROOT / "data" / "VinBigData_ChestXray.tar"
        if not tar_path.exists():
            raise FileNotFoundError(
                "Cannot find raw VinBigData image or data/VinBigData_ChestXray.tar"
            )
        with tarfile.open(tar_path, "r") as tf:
            member = tf.getmember(str(image_rel).replace("\\", "/"))
            fileobj = tf.extractfile(member)
            if fileobj is None:
                raise FileNotFoundError(member.name)
            image = Image.open(fileobj).convert("L")

    image = image.resize((384, 384), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    low, high = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)
    return arr


def apply_visual_perturbation(image: np.ndarray, kind: str, value: float) -> np.ndarray:
    if kind == "clean":
        return image.copy()
    if kind == "gaussian_noise":
        rng = np.random.default_rng(7)
        return np.clip(image + rng.normal(0.0, value, size=image.shape), 0.0, 1.0)
    if kind == "gaussian_blur":
        pil = Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255))
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=value))).astype(np.float32) / 255.0
    if kind == "brightness":
        return np.clip(image + value, 0.0, 1.0)
    if kind == "contrast":
        return np.clip((image - 0.5) * value + 0.5, 0.0, 1.0)
    raise ValueError(kind)


def build_perturbation_visual_examples() -> Path:
    image = load_demo_image()
    rows = [
        (
            "Gaussian noise",
            "gaussian_noise",
            [("clean", 0.0), ("sigma=0.05", 0.05), ("sigma=0.10", 0.10), ("sigma=0.20", 0.20), ("sigma=0.30", 0.30)],
        ),
        (
            "Gaussian blur",
            "gaussian_blur",
            [("clean", 0.0), ("sigma=0.5", 0.5), ("sigma=1.0", 1.0), ("sigma=2.0", 2.0), ("sigma=3.0", 3.0)],
        ),
        (
            "Brightness shift",
            "brightness",
            [("clean", 0.0), ("delta=-0.2", -0.20), ("delta=-0.1", -0.10), ("delta=+0.1", 0.10), ("delta=+0.2", 0.20)],
        ),
        (
            "Contrast scaling",
            "contrast",
            [("clean", 1.0), ("factor=0.25", 0.25), ("factor=0.5", 0.50), ("factor=1.5", 1.50), ("factor=2.0", 2.00)],
        ),
    ]

    fig, axes = plt.subplots(len(rows), 5, figsize=(13.0, 10.4))
    for r, (row_title, kind, variants) in enumerate(rows):
        for c, (label, value) in enumerate(variants):
            ax = axes[r, c]
            perturbed = apply_visual_perturbation(image, "clean" if label == "clean" else kind, value)
            ax.imshow(perturbed, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=9)
            if c == 0:
                ax.set_ylabel(row_title, fontsize=10)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#D1D5DB")

    fig.suptitle(
        "Figure 0. Visual examples of nuisance perturbations applied to the same chest X-ray",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.02,
        f"Source image: VinBigData train/{PERTURBATION_SAMPLE_ID}.png. "
        "The perturbation levels match the Exp1 evaluation schedule.",
        ha="center",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = OUT / "fig0_perturbation_visual_examples.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def split_condition(condition: str) -> tuple[str, float | None]:
    if condition == "clean":
        return "clean", None
    perturbation, value = condition.split("/", 1)
    raw = value.split("=", 1)[1]
    return perturbation, float(raw.replace("+", ""))


def load_exp1_long() -> pd.DataFrame:
    rows = []
    for model, directory in EXP1_RUNS.items():
        if not (directory / "robustness_curve.csv").exists():
            print(f"Skipping missing Exp1 run for {model}: {directory}")
            continue
        robust = pd.read_csv(directory / "robustness_curve.csv")
        drift = pd.read_csv(directory / "embedding_drift.csv")
        drift_map = drift.set_index("condition")[["cosine_drift", "l2_drift"]].to_dict("index")
        clean = robust.loc[robust["condition"] == "clean"].iloc[0]
        for perturbation, cfg in PERT_CONFIG.items():
            rows.append(
                {
                    "model": model,
                    "condition": "clean",
                    "perturbation": perturbation,
                    "x_value": cfg["clean_x"],
                    "macro_auroc": clean["macro_auroc"],
                    "f1": clean["f1"],
                    "auroc_drop": 0.0,
                    "f1_drop": 0.0,
                    "cosine_drift": 0.0,
                    "l2_drift": 0.0,
                }
            )
        for _, row in robust.loc[robust["condition"] != "clean"].iterrows():
            perturbation, x_value = split_condition(row["condition"])
            d = drift_map.get(row["condition"], {"cosine_drift": np.nan, "l2_drift": np.nan})
            rows.append(
                {
                    "model": model,
                    "condition": row["condition"],
                    "perturbation": perturbation,
                    "x_value": x_value,
                    "macro_auroc": row["macro_auroc"],
                    "f1": row["f1"],
                    "auroc_drop": row["auroc_drop"],
                    "f1_drop": row["f1_drop"],
                    "cosine_drift": d["cosine_drift"],
                    "l2_drift": d["l2_drift"],
                }
            )
    if not rows:
        raise FileNotFoundError("No Exp1 robustness results found.")
    df = pd.DataFrame(rows)
    return df.sort_values(["model", "perturbation", "x_value"])


def build_exp1_robustness_matrix(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.8), sharey="row")
    for col, (perturbation, cfg) in enumerate(PERT_CONFIG.items()):
        sub = df[df["perturbation"] == perturbation]
        for model, mdf in sub.groupby("model"):
            mdf = mdf.sort_values("x_value")
            color = MODEL_COLORS[model]
            axes[0, col].plot(
                mdf["x_value"],
                mdf["macro_auroc"],
                marker="o",
                linewidth=2,
                color=color,
                label=model,
            )
            axes[1, col].plot(
                mdf["x_value"],
                mdf["cosine_drift"],
                marker="o",
                linewidth=2,
                color=color,
                label=model,
            )
        axes[0, col].set_title(cfg["title"])
        axes[1, col].set_xlabel(cfg["xlabel"])
        axes[0, col].set_xlim(*cfg["xlim"])
        axes[1, col].set_xlim(*cfg["xlim"])
        axes[0, col].axvline(cfg["clean_x"], color="#111827", linewidth=0.8, alpha=0.35)
        axes[1, col].axvline(cfg["clean_x"], color="#111827", linewidth=0.8, alpha=0.35)
        axes[0, col].set_ylim(0.48, 0.96)
        axes[1, col].set_ylim(-0.02, max(1.02, sub["cosine_drift"].max() * 1.08))
    axes[0, 0].set_ylabel("Macro AUROC")
    axes[1, 0].set_ylabel("Cosine drift")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Figure 1. Robustness curves and representation drift")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / "fig1_exp1_robustness_matrix.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_exp1_tradeoff(df: pd.DataFrame) -> Path:
    sub = df[df["condition"] != "clean"].copy()
    sub["perturbation_label"] = sub["perturbation"].map(
        {
            "gaussian_noise": "noise",
            "gaussian_blur": "blur",
            "brightness": "brightness",
            "contrast": "contrast",
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for perturbation, pdf in sub.groupby("perturbation"):
        for model, mdf in pdf.groupby("model"):
            marker = "o" if model.startswith("I-JEPA") else "s"
            ax.scatter(
                mdf["cosine_drift"],
                mdf["auroc_drop"],
                s=68,
                marker=marker,
                color=PERT_COLORS[perturbation],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
                label=f"{model} / {pdf['perturbation_label'].iloc[0]}",
            )
    key_conditions = {
        ("I-JEPA-H/95", "gaussian_noise/sigma=0.05"): "I-JEPA mild noise",
        ("MAE-H/97", "gaussian_noise/sigma=0.05"): "MAE mild noise",
        ("I-JEPA-H/95", "brightness/delta=+0.2"): "I-JEPA +brightness",
        ("MAE-H/97", "brightness/delta=+0.2"): "MAE +brightness",
        ("I-JEPA-H/95", "contrast/factor=0.25"): "I-JEPA low contrast",
        ("MAE-H/97", "contrast/factor=0.25"): "MAE low contrast",
    }
    for (model, condition), text in key_conditions.items():
        row = sub[(sub["model"] == model) & (sub["condition"] == condition)]
        if row.empty:
            continue
        r = row.iloc[0]
        ax.annotate(
            text,
            xy=(r["cosine_drift"], r["auroc_drop"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color="#111827",
        )
    ax.set_xlabel("Cosine drift from clean embedding")
    ax.set_ylabel("Macro AUROC drop")
    ax.set_title("Figure 2. Functional degradation is not identical to feature drift")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.grid(True, alpha=0.25)
    legend_items = [
        plt.Line2D([0], [0], marker="o", color="w", label="I-JEPA-H/95", markerfacecolor="#6B7280", markersize=7),
        plt.Line2D([0], [0], marker="s", color="w", label="MAE-H/97", markerfacecolor="#6B7280", markersize=7),
    ]
    perturb_items = [
        plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=color, markersize=7)
        for name, color in [
            ("noise", PERT_COLORS["gaussian_noise"]),
            ("blur", PERT_COLORS["gaussian_blur"]),
            ("brightness", PERT_COLORS["brightness"]),
            ("contrast", PERT_COLORS["contrast"]),
        ]
    ]
    ax.legend(handles=legend_items + perturb_items, frameon=True, loc="lower right")
    fig.tight_layout()
    path = OUT / "fig2_exp1_drop_drift_tradeoff.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_exp1_complete_table(df: pd.DataFrame) -> Path:
    model_order = ["MAE-H/97", "MAE-H/300", "I-JEPA-H/95", "I-JEPA-H/201"]
    model_order = [model for model in model_order if model in set(df["model"])]
    rows = [
        ("Clean", "clean", None),
        ("Noise\nsigma 0.05", "gaussian_noise", 0.05),
        ("Noise\nsigma 0.10", "gaussian_noise", 0.10),
        ("Noise\nsigma 0.20", "gaussian_noise", 0.20),
        ("Noise\nsigma 0.30", "gaussian_noise", 0.30),
        ("Blur\nsigma 0.50", "gaussian_blur", 0.50),
        ("Blur\nsigma 1.00", "gaussian_blur", 1.00),
        ("Blur\nsigma 2.00", "gaussian_blur", 2.00),
        ("Blur\nsigma 3.00", "gaussian_blur", 3.00),
        ("Brightness\ndelta -0.20", "brightness", -0.20),
        ("Brightness\ndelta -0.10", "brightness", -0.10),
        ("Brightness\ndelta +0.10", "brightness", 0.10),
        ("Brightness\ndelta +0.20", "brightness", 0.20),
        ("Contrast\nfactor 0.25", "contrast", 0.25),
        ("Contrast\nfactor 0.50", "contrast", 0.50),
        ("Contrast\nfactor 1.50", "contrast", 1.50),
        ("Contrast\nfactor 2.00", "contrast", 2.00),
    ]
    table_rows = []
    best_by_row: list[str | None] = []
    for label, perturbation, x_value in rows:
        out_row = {"Condition": label}
        if x_value is None:
            part = df[df["condition"] == "clean"].drop_duplicates("model")
        else:
            part = df[(df["perturbation"] == perturbation) & (np.isclose(df["x_value"], x_value))]
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
                out_row[model] = ""
                continue
            r = s.iloc[0]
            if x_value is None:
                out_row[model] = f"{r['macro_auroc']:.3f}"
            else:
                out_row[model] = f"{r['macro_auroc']:.3f}\n(drop {r['auroc_drop']:.3f}; d {r['cosine_drift']:.2f})"
        table_rows.append(out_row)
        best_by_row.append(best_model)

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(OUT / "table_exp1_complete_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(14.5, 10.2))
    ax.axis("off")
    ax.set_title("Table 1. Complete Exp1 robustness summary", fontsize=16, pad=18)
    ax.text(
        0.5,
        0.965,
        "Cells show Macro AUROC; parentheses show AUROC drop from clean and cosine drift d.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
    )
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.01, 0.02, 0.98, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1.0, 1.2)
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
    path = OUT / "table1_exp1_complete_summary.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def load_exp2_summary() -> pd.DataFrame:
    rows = []
    for model, directory in EXP2_SUMMARY_RUNS.items():
        if not (directory / "prediction_drop.csv").exists():
            print(f"Skipping missing Exp2 summary for {model}: {directory}")
            continue
        pred = pd.read_csv(directory / "prediction_drop.csv")
        values = pred.set_index("condition")
        rows.append(
            {
                "model": model,
                "original_auroc": float(values.loc["original", "macro_auroc"]),
                "lesion_auroc": float(values.loc["lesion_occ", "macro_auroc"]),
                "control_auroc": float(values.loc["control_occ", "macro_auroc"]),
                "lesion_drop": float(values.loc["lesion_occ", "auroc_drop"]),
                "control_drop": float(values.loc["control_occ", "auroc_drop"]),
                "drop_delta_lesion_minus_control": float(values.loc["lesion_occ", "auroc_drop"])
                - float(values.loc["control_occ", "auroc_drop"]),
                "lesion_pred_drop_absmax": float(values.loc["lesion_occ", "mean_pred_drop_max_class"]),
                "control_pred_drop_absmax": float(values.loc["control_occ", "mean_pred_drop_max_class"]),
            }
        )
    if not rows:
        raise FileNotFoundError("No Exp2 prediction_drop.csv files found.")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table_exp2_occlusion_summary.csv", index=False)
    return df


def build_exp2_occlusion_summary(df: pd.DataFrame) -> Path:
    order = [m for m in EXP2_SUMMARY_RUNS.keys() if m in set(df["model"])]
    df = df.set_index("model").loc[order].reset_index()
    x = np.arange(len(df))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))

    axes[0].bar(df["model"], df["original_auroc"], color=[MODEL_COLORS[m] for m in df["model"]])
    axes[0].set_ylim(0.56, 0.84)
    axes[0].set_ylabel("Original macro AUROC")
    axes[0].set_title("Clean Exp2 performance")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(x - width / 2, df["lesion_drop"], width, label="Lesion occlusion", color="#7C3AED")
    axes[1].bar(x + width / 2, df["control_drop"], width, label="Control occlusion", color="#F97316")
    axes[1].set_xticks(x, df["model"], rotation=25, ha="right")
    axes[1].set_ylabel("AUROC drop")
    axes[1].set_title("Occlusion impact")
    axes[1].legend(frameon=False)

    delta_colors = ["#7C3AED" if v > 0 else "#F97316" for v in df["drop_delta_lesion_minus_control"]]
    axes[2].bar(df["model"], df["drop_delta_lesion_minus_control"], color=delta_colors)
    axes[2].axhline(0, color="#111827", linewidth=1)
    axes[2].set_ylabel("Lesion drop - control drop")
    axes[2].set_title("Lesion-specific AUROC effect")
    axes[2].tick_params(axis="x", rotation=25)

    fig.suptitle("Figure 3. Lesion occlusion does not dominate matched control occlusion")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "fig3_exp2_occlusion_summary.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def load_paired_tests() -> pd.DataFrame:
    rows = []
    for model, directory in EXP2_PER_SAMPLE_RUNS.items():
        if not (directory / "analysis" / "paired_tests.csv").exists():
            print(f"Skipping missing paired tests for {model}: {directory}")
            continue
        df = pd.read_csv(directory / "analysis" / "paired_tests.csv")
        df["model"] = model
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No Exp2 paired_tests.csv files found.")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "table_exp2_paired_tests.csv", index=False)
    return out


def build_paired_effects(paired: pd.DataFrame) -> Path:
    metrics = [
        ("cosine_drift", "Cosine drift delta"),
        ("prediction_drop_absmax", "Max prediction drop delta"),
    ]
    order = [m for m in EXP2_PER_SAMPLE_RUNS.keys() if m in set(paired["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        sub = paired[paired["metric"] == metric].set_index("model").loc[order].reset_index()
        x = np.arange(len(sub))
        y = sub["delta_mean_lesion_minus_control"].to_numpy()
        yerr = np.vstack(
            [
                y - sub["delta_ci95_low"].to_numpy(),
                sub["delta_ci95_high"].to_numpy() - y,
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=7,
            linewidth=1.8,
            capsize=4,
            color="#111827",
            ecolor="#374151",
        )
        for idx, row in sub.iterrows():
            color = MODEL_COLORS[row["model"]]
            ax.scatter(idx, row["delta_mean_lesion_minus_control"], s=72, color=color, zorder=3)
            label = "p<0.001" if row["sign_flip_pvalue"] < 0.001 else f"p={row['sign_flip_pvalue']:.3f}"
            ax.annotate(label, (idx, row["delta_ci95_high"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_xticks(x, sub["model"], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Lesion - control")
    fig.suptitle("Figure 4. Paired per-sample lesion-control effects with 95% CIs")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "fig4_exp2_paired_effects.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def load_per_sample() -> pd.DataFrame:
    frames = []
    for model, directory in EXP2_PER_SAMPLE_RUNS.items():
        path = directory / "analysis" / "per_sample_results_with_groups.csv"
        if not path.exists():
            path = directory / "per_sample_results.csv"
        if not path.exists():
            print(f"Skipping missing per-sample results for {model}: {directory}")
            continue
        df = pd.read_csv(path)
        df["model"] = model
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No Exp2 per-sample result files found.")
    return pd.concat(frames, ignore_index=True)


def build_per_sample_distribution(df: pd.DataFrame) -> Path:
    order = [m for m in EXP2_PER_SAMPLE_RUNS.keys() if m in set(df["model"])]
    fig, axes = plt.subplots(2, len(order), figsize=(13.5, 7.2))
    if len(order) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, model in enumerate(order):
        sub = df[df["model"] == model]
        color = MODEL_COLORS[model]
        ax = axes[0, col]
        sns.histplot(sub["delta_pred_drop_absmax"], bins=42, kde=True, ax=ax, color=color)
        ax.axvline(0, color="#111827", linewidth=1)
        ax.axvline(sub["delta_pred_drop_absmax"].mean(), color=color, linewidth=2, linestyle="--")
        ax.set_title(model)
        ax.set_xlabel("Delta max pred drop\n(lesion - control)")
        ax.set_ylabel("Count" if col == 0 else "")

        ax = axes[1, col]
        ax.scatter(
            sub["control_pred_drop_absmax"],
            sub["lesion_pred_drop_absmax"],
            s=10,
            color=color,
            alpha=0.28,
            edgecolors="none",
        )
        maxv = float(max(sub["control_pred_drop_absmax"].max(), sub["lesion_pred_drop_absmax"].max()))
        ax.plot([0, maxv], [0, maxv], color="#111827", linewidth=1, linestyle="--")
        ax.set_xlim(-0.01, maxv * 1.03)
        ax.set_ylim(-0.01, maxv * 1.03)
        ax.set_xlabel("Control max pred drop")
        ax.set_ylabel("Lesion max pred drop" if col == 0 else "")
    fig.suptitle("Figure 5. Per-sample occlusion effects are broad and weakly centered")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "fig5_exp2_per_sample_distribution.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_subgroup_effects() -> Path:
    def load_summary(filename: str) -> pd.DataFrame:
        frames = []
        for model, directory in EXP2_PER_SAMPLE_RUNS.items():
            path = directory / "analysis" / filename
            if not path.exists():
                print(f"Skipping missing subgroup file for {model}: {path}")
                continue
            df = pd.read_csv(path)
            df["model"] = model
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No subgroup summary files found for {filename}")
        return pd.concat(frames, ignore_index=True)

    bbox = load_summary("bbox_size_group_summary.csv")
    disease = load_summary("disease_group_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.7))

    def draw_panel(ax, df: pd.DataFrame, order: list[str], title: str) -> None:
        x = np.arange(len(order))
        models = [m for m in EXP2_PER_SAMPLE_RUNS.keys() if m in set(df["model"])]
        offsets = np.linspace(-0.24, 0.24, len(models))
        for offset, model in zip(offsets, models):
            sub = df[df["model"] == model].set_index("group").reindex(order).reset_index()
            y = sub["delta_pred_drop_absmax_mean"].to_numpy()
            yerr = np.vstack(
                [
                    y - sub["delta_pred_drop_absmax_ci_low"].to_numpy(),
                    sub["delta_pred_drop_absmax_ci_high"].to_numpy() - y,
                ]
            )
            ax.errorbar(
                x + offset,
                y,
                yerr=yerr,
                fmt="o",
                capsize=3,
                linewidth=1.3,
                color=MODEL_COLORS[model],
                label=model,
            )
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_xticks(x, order, rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Delta max pred drop\n(lesion - control)")

    draw_panel(axes[0], bbox, ["small", "medium", "large"], "By bbox area")
    draw_panel(axes[1], disease, ["global_only", "local_only", "mixed_global_local"], "By disease group")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Figure 6. Subgroup analysis exposes context and area effects")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "fig6_exp2_subgroup_effects.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_per_class_heatmap() -> Path:
    frames = []
    for model, directory in EXP2_PER_SAMPLE_RUNS.items():
        path = directory / "prediction_drop_per_class.csv"
        if not path.exists():
            print(f"Skipping missing per-class file for {model}: {path}")
            continue
        df = pd.read_csv(path)
        df["model"] = model
        df["delta_lesion_minus_control"] = df["lesion_pred_drop"] - df["control_pred_drop"]
        frames.append(df[["class", "model", "delta_lesion_minus_control"]])
    if not frames:
        raise FileNotFoundError("No prediction_drop_per_class.csv files found.")
    pc = pd.concat(frames, ignore_index=True)
    pc.to_csv(OUT / "table_exp2_per_class_delta.csv", index=False)
    pivot = pc.pivot(index="class", columns="model", values="delta_lesion_minus_control")
    pivot = pivot[[m for m in EXP2_PER_SAMPLE_RUNS.keys() if m in pivot.columns]]
    pivot = pivot.reindex(pivot.abs().mean(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(7.8, 7.4))
    vmax = max(0.006, float(np.nanmax(np.abs(pivot.to_numpy()))))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Lesion - control signed score drop"},
        ax=ax,
    )
    ax.set_title("Figure 7. Per-class lesion-control effects are inconsistent")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    path = OUT / "fig7_exp2_per_class_heatmap.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def build_case_catalog_and_sheet() -> Path:
    rows = []
    for model, directory in EXP2_PER_SAMPLE_RUNS.items():
        selected = directory / "case_studies" / "selected_cases.csv"
        if not selected.exists():
            continue
        df = pd.read_csv(selected)
        for _, row in df.iterrows():
            pngs = list((directory / "case_studies").glob(f"*_{row['image_id']}.png"))
            if not pngs:
                continue
            delta = float(row["delta_pred_drop_absmax"])
            if float(row["bbox_area_frac"]) <= 0:
                category = "zero-effective-bbox"
            elif delta >= 0.05:
                category = "lesion-dominant"
            elif delta <= -0.05:
                category = "control-dominant"
            else:
                category = "matched/mixed"
            rows.append(
                {
                    "model": model,
                    "image_id": row["image_id"],
                    "lesion_classes": row["lesion_classes"],
                    "bbox_area_frac": row["bbox_area_frac"],
                    "lesion_pred_drop_absmax": row["lesion_pred_drop_absmax"],
                    "control_pred_drop_absmax": row["control_pred_drop_absmax"],
                    "delta_pred_drop_absmax": delta,
                    "lesion_cosine_drift": row["lesion_cosine_drift"],
                    "control_cosine_drift": row["control_cosine_drift"],
                    "delta_cosine_drift": row["delta_cosine_drift"],
                    "category": category,
                    "figure_path": str(pngs[0].relative_to(ROOT)),
                }
            )
    catalog = pd.DataFrame(rows)
    catalog.to_csv(OUT / "case_study_catalog.csv", index=False)

    if catalog.empty:
        raise FileNotFoundError("No case-study images found.")

    chosen = []
    for model, category in [
        ("I-JEPA-H/201", "lesion-dominant"),
        ("I-JEPA-H/201", "control-dominant"),
        ("MAE-H/97", "control-dominant"),
        ("I-JEPA-H/95", "matched/mixed"),
    ]:
        sub = catalog[(catalog["model"] == model) & (catalog["category"] == category)]
        if sub.empty:
            sub = catalog[catalog["model"] == model]
        if sub.empty:
            continue
        chosen.append(sub.iloc[sub["delta_pred_drop_absmax"].abs().argmax()])

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
    for ax, row in zip(axes.ravel(), chosen):
        image = Image.open(ROOT / row["figure_path"])
        ax.imshow(image)
        ax.axis("off")
        title = (
            f"{row['model']} | {row['category']} | "
            f"delta pred={row['delta_pred_drop_absmax']:.3f}, "
            f"delta drift={row['delta_cosine_drift']:.3f}"
        )
        ax.set_title(title, fontsize=9)
    for ax in axes.ravel()[len(chosen) :]:
        ax.axis("off")
    fig.suptitle("Figure 8. Representative qualitative cases")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "fig8_case_study_contact_sheet.png"
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)

    md_lines = [
        "# Case Study Notes",
        "",
        "These cases were selected from the generated per-sample case-study figures.",
        "Delta values are lesion minus control; positive values indicate a stronger lesion-occlusion effect.",
        "",
    ]
    for row in chosen:
        md_lines.extend(
            [
                f"## {row['model']} - {row['category']}",
                f"- Image: `{row['image_id']}`",
                f"- Classes: {row['lesion_classes']}",
                f"- Bbox area fraction: {float(row['bbox_area_frac']):.3f}",
                f"- Max prediction drop: lesion={float(row['lesion_pred_drop_absmax']):.3f}, "
                f"control={float(row['control_pred_drop_absmax']):.3f}, "
                f"delta={float(row['delta_pred_drop_absmax']):.3f}",
                f"- Cosine drift: lesion={float(row['lesion_cosine_drift']):.3f}, "
                f"control={float(row['control_cosine_drift']):.3f}, "
                f"delta={float(row['delta_cosine_drift']):.3f}",
                f"- Figure: `{row['figure_path']}`",
                "",
            ]
        )
    (OUT / "case_study_notes.md").write_text("\n".join(md_lines), encoding="utf-8")
    return path


def build_manifest(paths: list[Path]) -> None:
    captions = {
        "fig0_perturbation_visual_examples.png": "Visual examples of the exact nuisance perturbation schedule applied to one raw chest X-ray.",
        "fig1_exp1_robustness_matrix.png": "Exp1 macro AUROC and cosine drift curves for I-JEPA-H/95, I-JEPA-H/201, MAE-H/97, and MAE-H/300.",
        "fig2_exp1_drop_drift_tradeoff.png": "Scatter plot showing that representation drift and downstream AUROC drop are related but not interchangeable.",
        "table1_exp1_complete_summary.png": "Complete Exp1 robustness table with best AUROC highlighted per condition.",
        "fig3_exp2_occlusion_summary.png": "Exp2 original AUROC, lesion/control AUROC drops, and lesion-control AUROC delta.",
        "fig4_exp2_paired_effects.png": "Per-sample lesion-control deltas with 95% CIs for I-JEPA-H/95, I-JEPA-H/201, and MAE-H/97.",
        "fig5_exp2_per_sample_distribution.png": "Per-sample distributions and paired scatter plots for max-class prediction drop.",
        "fig6_exp2_subgroup_effects.png": "Subgroup deltas by bbox area and disease group.",
        "fig7_exp2_per_class_heatmap.png": "Per-class signed score-drop differences between lesion and control occlusion.",
        "fig8_case_study_contact_sheet.png": "Representative qualitative cases from the generated case-study images.",
    }
    lines = ["# Paper Figure Manifest", ""]
    for path in paths:
        rel = path.relative_to(ROOT)
        lines.append(f"- `{rel}`: {captions.get(path.name, '')}")
    lines.extend(
        [
            "",
            "Generated tables:",
            f"- `{(OUT / 'table_exp2_occlusion_summary.csv').relative_to(ROOT)}`",
            f"- `{(OUT / 'table_exp2_paired_tests.csv').relative_to(ROOT)}`",
            f"- `{(OUT / 'table_exp2_per_class_delta.csv').relative_to(ROOT)}`",
            f"- `{(OUT / 'case_study_catalog.csv').relative_to(ROOT)}`",
            f"- `{(OUT / 'case_study_notes.md').relative_to(ROOT)}`",
        ]
    )
    (OUT / "figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    ensure_out()
    paths: list[Path] = []

    paths.append(build_perturbation_visual_examples())

    exp1 = load_exp1_long()
    exp1.to_csv(OUT / "table_exp1_robustness_long.csv", index=False)
    paths.append(build_exp1_robustness_matrix(exp1))
    paths.append(build_exp1_tradeoff(exp1))
    paths.append(build_exp1_complete_table(exp1))

    exp2_summary = load_exp2_summary()
    paths.append(build_exp2_occlusion_summary(exp2_summary))

    paired = load_paired_tests()
    paths.append(build_paired_effects(paired))

    per_sample = load_per_sample()
    per_sample.to_csv(OUT / "table_exp2_per_sample_combined.csv", index=False)
    paths.append(build_per_sample_distribution(per_sample))
    paths.append(build_subgroup_effects())
    paths.append(build_per_class_heatmap())
    paths.append(build_case_catalog_and_sheet())

    build_manifest(paths)
    print("Generated paper figures:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
