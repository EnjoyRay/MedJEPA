"""Build JEPA300-vs-MAE300 visualizations for EXPERIMENT_REPORT.md.

The older paper figure script predates the JEPA300 fair rerun.  This script is
intentionally narrow: it reads the already materialized CSV outputs for the
current fair-comparison matrix and writes report-ready PNGs.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "report_visualizations"

MODELS = {
    "I-JEPA-H/300": {
        "short": "JEPA300",
        "color": "#0F766E",
        "exp1": RESULTS / "exp1_ijepa_h300",
        "exp2b": RESULTS / "exp2b_class_aligned_ijepa_h300",
        "exp3": RESULTS / "exp3_frequency_ijepa_h300",
        "exp4": RESULTS / "exp4_mechanism_ijepa_h300",
        "exp5": RESULTS / "exp5_mitigation_ijepa_h300",
        "exp6": RESULTS / "exp6_nca_ijepa_h300",
        "exp7": RESULTS / "exp7_mlp_probe_ijepa_h300",
        "exp8": RESULTS / "exp8_partial_ft_ijepa_h300",
    },
    "MAE-H/300": {
        "short": "MAE300",
        "color": "#B45309",
        "exp1": RESULTS / "exp1_mae_huge_mimic_300ep_fixedsplit",
        "exp2b": RESULTS / "exp2b_class_aligned_mae_huge_mimic_300ep_fixedsplit",
        "exp3": RESULTS / "exp3_frequency_mae_huge_mimic_300ep_fixedsplit",
        "exp4": RESULTS / "exp4_mechanism_mae_huge_mimic_300ep_fixedsplit",
        "exp5": RESULTS / "exp5_mitigation_mae_huge_mimic_300ep_fixedsplit",
        "exp6": RESULTS / "exp6_nca_mae_huge_mimic_300ep_fixedsplit",
        "exp7": RESULTS / "exp7_mlp_probe_mae_huge_mimic_300ep_fixedsplit",
        "exp8": RESULTS / "exp8_partial_ft_mae_huge_mimic_300ep_fixedsplit",
    },
}

SIGMA_ORDER = [0.0, 0.05, 0.10, 0.20, 0.30]
FREQ_ORDER = [
    "Clean",
    "Band 0.00-0.20",
    "Band 0.20-0.45",
    "Band 0.45-1.00",
    "Low-pass 0.15",
]
EXAMPLE_IMAGE_ID = "d0a8b798569ce701b96f52058e99e5f4"
EXAMPLE_CLASS = "Pleural effusion"


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
            "savefig.dpi": 240,
        }
    )


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    low, high = np.percentile(arr, [0.5, 99.5])
    return np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)


def load_tar_image(image_id: str) -> np.ndarray:
    image_rel = f"VinBigData_ChestXray/images_1024/train/{image_id}.png"
    with tarfile.open(RESULTS.parent / "data" / "VinBigData_ChestXray.tar", "r") as tf:
        with tf.extractfile(image_rel) as fileobj:
            if fileobj is None:
                raise FileNotFoundError(image_rel)
            image = Image.open(fileobj).convert("L")
            return normalize_image(np.asarray(image))


def load_example_boxes(image_id: str, class_name: str, image_shape: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    tar_path = RESULTS.parent / "data" / "VinBigData_ChestXray.tar"
    with tarfile.open(tar_path, "r") as tf:
        ann = pd.read_csv(tf.extractfile("VinBigData_ChestXray/annotations/train.csv"))
        meta = pd.read_csv(tf.extractfile("VinBigData_ChestXray/images_1024/train_meta.csv"))

    row_meta = meta[meta["image_id"] == image_id].iloc[0]
    sx = image_shape[1] / float(row_meta["dim1"])
    sy = image_shape[0] / float(row_meta["dim0"])
    lesion_rows = ann[(ann["image_id"] == image_id) & (ann["class_name"] == class_name)]
    lesion_boxes = [
        {
            "x_min": float(row["x_min"]) * sx,
            "y_min": float(row["y_min"]) * sy,
            "x_max": float(row["x_max"]) * sx,
            "y_max": float(row["y_max"]) * sy,
            "class_name": class_name,
        }
        for _, row in lesion_rows.iterrows()
    ]

    sample = pd.read_csv(MODELS["I-JEPA-H/300"]["exp2b"] / "class_aligned_per_sample.csv")
    sample_row = sample[(sample["image_id"] == image_id) & (sample["class_name"] == class_name)].iloc[0]
    control_sets = json.loads(sample_row["control_boxes_json"])
    control_boxes = control_sets[0]
    # Exp2b boxes are defined on the 224x224 evaluation canvas.
    cx = image_shape[1] / 224.0
    cy = image_shape[0] / 224.0
    for box in control_boxes:
        box["x_min"] = float(box["x_min"]) * cx
        box["x_max"] = float(box["x_max"]) * cx
        box["y_min"] = float(box["y_min"]) * cy
        box["y_max"] = float(box["y_max"]) * cy
    return lesion_boxes, control_boxes


def apply_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(23)
    return np.clip(image + rng.normal(0.0, sigma, size=image.shape), 0.0, 1.0)


def apply_mask(image: np.ndarray, boxes: list[dict], fill: float = 0.5) -> np.ndarray:
    masked = image.copy()
    h, w = masked.shape
    for box in boxes:
        x0 = max(0, int(round(box["x_min"])))
        x1 = min(w, int(round(box["x_max"])))
        y0 = max(0, int(round(box["y_min"])))
        y1 = min(h, int(round(box["y_max"])))
        masked[y0:y1, x0:x1] = fill
    return masked


def draw_boxes(ax: plt.Axes, boxes: list[dict], color: str, label: str) -> None:
    for i, box in enumerate(boxes):
        rect = Rectangle(
            (box["x_min"], box["y_min"]),
            box["x_max"] - box["x_min"],
            box["y_max"] - box["y_min"],
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            label=label if i == 0 else None,
        )
        ax.add_patch(rect)


def plot_input_perturbation_examples() -> Path:
    image = load_tar_image(EXAMPLE_IMAGE_ID)
    lesion_boxes, control_boxes = load_example_boxes(EXAMPLE_IMAGE_ID, EXAMPLE_CLASS, image.shape)
    panels = [
        ("Clean", image, "none"),
        ("Gaussian noise\nsigma=0.05", apply_noise(image, 0.05), "none"),
        ("Gaussian noise\nsigma=0.10", apply_noise(image, 0.10), "none"),
        ("Gaussian noise\nsigma=0.20", apply_noise(image, 0.20), "none"),
        ("Gaussian noise\nsigma=0.30", apply_noise(image, 0.30), "none"),
        ("Lesion boxes", image, "boxes"),
        ("Lesion occlusion", apply_mask(image, lesion_boxes), "lesion"),
        ("Matched control\nocclusion", apply_mask(image, control_boxes), "control"),
        ("Lesion vs control\nlocations", image, "both"),
        ("Clean crop context", image, "crop"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(14.4, 8.0))
    for ax, (title, panel, mode) in zip(axes.ravel(), panels):
        ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        if mode in {"boxes", "lesion", "both"}:
            draw_boxes(ax, lesion_boxes, "#10B981", "lesion")
        if mode in {"control", "both"}:
            draw_boxes(ax, control_boxes, "#F59E0B", "control")
        if mode == "crop":
            draw_boxes(ax, lesion_boxes, "#10B981", "lesion")
            ax.set_xlim(420, 940)
            ax.set_ylim(850, 80)
        ax.set_title(title, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 3].legend(frameon=True, loc="lower right")
    fig.suptitle(
        "Input-level perturbation examples: Gaussian noise and class-aligned occlusion",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.02,
        f"Example image {EXAMPLE_IMAGE_ID}, target class: {EXAMPLE_CLASS}. "
        "Occlusion uses the neutral-fill masking protocol from Exp2b.",
        ha="center",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), h_pad=2.4, w_pad=0.8)
    return savefig(fig, "fig_input_noise_occlusion_examples.png")


def parse_sigma(condition: str) -> float | None:
    if condition == "clean":
        return 0.0
    if condition.startswith("gaussian_noise/sigma="):
        return float(condition.split("=", 1)[1])
    if condition.startswith("noise_sigma="):
        return float(condition.split("=", 1)[1])
    return None


def load_exp1_noise() -> pd.DataFrame:
    frames = []
    for model, cfg in MODELS.items():
        robust = pd.read_csv(cfg["exp1"] / "robustness_curve.csv")
        drift = pd.read_csv(cfg["exp1"] / "embedding_drift.csv")
        drift = pd.concat(
            [
                pd.DataFrame([{"condition": "clean", "cosine_drift": 0.0, "l2_drift": 0.0}]),
                drift,
            ],
            ignore_index=True,
        )
        df = robust.merge(drift, on="condition", how="left")
        df["sigma"] = df["condition"].map(parse_sigma)
        df = df[df["sigma"].isin(SIGMA_ORDER)].copy()
        df["model"] = model
        df["short"] = cfg["short"]
        df["color"] = cfg["color"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["model", "sigma"])


def plot_exp1_noise() -> Path:
    df = load_exp1_noise()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharex=True)
    panels = [
        ("macro_auroc", "Macro AUROC", (0.42, 0.96)),
        ("auroc_drop", "AUROC drop", (-0.02, 0.46)),
        ("cosine_drift", "Cosine drift", (-0.03, 0.98)),
    ]
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for model, sub in df.groupby("model"):
            color = MODELS[model]["color"]
            ax.plot(sub["sigma"], sub[metric], marker="o", linewidth=2.0, color=color, label=MODELS[model]["short"])
        ax.set_title(ylabel)
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(SIGMA_ORDER)
        ax.axvline(0.05, color="#9CA3AF", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Exp1: JEPA300 has a light-noise failure threshold; MAE300 is smoother at sigma=0.05")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return savefig(fig, "fig_exp1_jepa300_mae300_noise.png")


def load_exp2b_overall() -> pd.DataFrame:
    rows = []
    for model, cfg in MODELS.items():
        row = pd.read_csv(cfg["exp2b"] / "class_aligned_overall.csv").iloc[0].to_dict()
        row["model"] = model
        row["short"] = cfg["short"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_exp2b_lesion() -> Path:
    overall = load_exp2b_overall()
    classes = []
    for model, cfg in MODELS.items():
        df = pd.read_csv(cfg["exp2b"] / "class_aligned_summary_by_class.csv")
        df["model"] = model
        df["short"] = cfg["short"]
        classes.append(df)
    classes = pd.concat(classes, ignore_index=True)
    top_classes = (
        classes[classes["model"] == "I-JEPA-H/300"]
        .sort_values("delta_logit_drop_mean", ascending=False)
        .head(8)["group"]
        .tolist()
    )
    heat = classes[classes["group"].isin(top_classes)].pivot(index="group", columns="short", values="delta_logit_drop_mean")
    heat = heat.reindex(top_classes)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), gridspec_kw={"width_ratios": [0.9, 1.6]})
    ax = axes[0]
    x = np.arange(len(overall))
    y = overall["delta_logit_drop_mean"].to_numpy()
    yerr = np.vstack(
        [
            y - overall["delta_logit_ci95_low"].to_numpy(),
            overall["delta_logit_ci95_high"].to_numpy() - y,
        ]
    )
    colors = [MODELS[m]["color"] for m in overall["model"]]
    ax.bar(x, y, yerr=yerr, color=colors, capsize=4)
    ax.set_xticks(x, overall["short"])
    ax.set_ylabel("Delta logit drop\n(lesion - control)")
    ax.set_title("Overall paired effect")
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.018, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

    sns.heatmap(
        heat,
        ax=axes[1],
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Delta logit drop"},
    )
    axes[1].set_title("Strongest JEPA300 class-aligned effects")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.suptitle("Exp2b: JEPA300 is much more class-aligned lesion-sensitive than MAE300")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return savefig(fig, "fig_exp2b_jepa300_mae300_lesion.png")


def condition_label(condition: str) -> str:
    mapping = {
        "clean": "Clean",
        "band_corrupt/band=0.00-0.20": "Band 0.00-0.20",
        "band_corrupt/band=0.20-0.45": "Band 0.20-0.45",
        "band_corrupt/band=0.45-1.00": "Band 0.45-1.00",
        "low_pass/cutoff=0.15": "Low-pass 0.15",
    }
    return mapping.get(condition, condition)


def load_exp3_key_conditions() -> pd.DataFrame:
    frames = []
    for model, cfg in MODELS.items():
        df = pd.read_csv(cfg["exp3"] / "frequency_sensitivity_summary.csv")
        clean = pd.DataFrame(
            [
                {
                    "condition": "clean",
                    "macro_auroc": load_exp1_noise().query("model == @model and sigma == 0.0")["macro_auroc"].iloc[0],
                    "auroc_drop": 0.0,
                    "cosine_drift": 0.0,
                }
            ]
        )
        df = pd.concat([clean, df], ignore_index=True)
        df["label"] = df["condition"].map(condition_label)
        df = df[df["label"].isin(FREQ_ORDER)].copy()
        df["label"] = pd.Categorical(df["label"], categories=FREQ_ORDER, ordered=True)
        df["model"] = model
        df["short"] = cfg["short"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["label", "model"])


def plot_exp3_frequency() -> Path:
    df = load_exp3_key_conditions()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "macro_auroc", "Macro AUROC"),
        (axes[1], "cosine_drift", "Cosine drift"),
    ]:
        sns.barplot(data=df, x="label", y=metric, hue="short", palette=[MODELS[m]["color"] for m in MODELS], ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(frameon=False, title="")
    fig.suptitle("Exp3: JEPA300 is most vulnerable to mid/high-frequency corruption")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return savefig(fig, "fig_exp3_jepa300_mae300_frequency.png")


def plot_exp4_mechanism() -> Path:
    frames = []
    for model, cfg in MODELS.items():
        df = pd.read_csv(cfg["exp4"] / "mechanism_summary_by_group.csv")
        df["model"] = model
        df["short"] = cfg["short"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    melted = df.melt(
        id_vars=["short", "disease_group"],
        value_vars=["inside_saliency_ratio_mean", "pointing_hit_mean", "noise_drift_mean", "lesion_drift_mean"],
        var_name="metric",
        value_name="value",
    )
    titles = {
        "inside_saliency_ratio_mean": "Inside saliency ratio",
        "pointing_hit_mean": "Pointing hit",
        "noise_drift_mean": "Noise drift",
        "lesion_drift_mean": "Lesion drift",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0))
    for ax, metric in zip(axes.ravel(), titles):
        sub = melted[melted["metric"] == metric]
        sns.barplot(data=sub, x="disease_group", y="value", hue="short", palette=[MODELS[m]["color"] for m in MODELS], ax=ax)
        ax.set_title(titles[metric])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(frameon=False, title="")
    fig.suptitle("Exp4: mechanism signals separate lesion localization from nuisance drift")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return savefig(fig, "fig_exp4_jepa300_mae300_mechanism.png")


def plot_exp5_exp6_mitigation() -> Path:
    exp5_frames = []
    exp6_frames = []
    for model, cfg in MODELS.items():
        m5 = pd.read_csv(cfg["exp5"] / "mitigation_results.csv")
        m5 = m5[m5["condition"].isin(["clean", "noise_sigma=0.05", "noise_sigma=0.10"])].copy()
        m5["model"] = model
        m5["short"] = cfg["short"]
        exp5_frames.append(m5)

        m6 = pd.read_csv(cfg["exp6"] / "nca_results.csv")
        m6 = m6[m6["condition"].isin(["clean", "noise_sigma=0.05", "noise_sigma=0.10"])].copy()
        m6["model"] = model
        m6["short"] = cfg["short"]
        exp6_frames.append(m6)

    exp5 = pd.concat(exp5_frames, ignore_index=True)
    exp6 = pd.concat(exp6_frames, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    noise5 = exp5[exp5["condition"] == "noise_sigma=0.05"].copy()
    sns.barplot(data=noise5, x="mitigation", y="macro_auroc", hue="short", palette=[MODELS[m]["color"] for m in MODELS], ax=axes[0])
    axes[0].set_title("Exp5 preprocessing at sigma=0.05")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Macro AUROC")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False, title="")

    sns.lineplot(data=exp6, x="condition", y="macro_auroc", hue="short", marker="o", palette=[MODELS[m]["color"] for m in MODELS], ax=axes[1])
    axes[1].set_title("Exp6 NCA AUROC")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Macro AUROC")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, title="")

    drift = exp6[exp6["condition"].isin(["noise_sigma=0.05", "noise_sigma=0.10"])].melt(
        id_vars=["short", "condition"],
        value_vars=["raw_cosine_drift", "adapter_cosine_drift"],
        var_name="space",
        value_name="drift",
    )
    drift["space"] = drift["space"].map({"raw_cosine_drift": "Raw encoder", "adapter_cosine_drift": "NCA output"})
    sns.barplot(data=drift, x="condition", y="drift", hue="space", palette=["#6B7280", "#7C3AED"], ax=axes[2])
    axes[2].set_title("NCA collapses drift after adaptation")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Cosine drift")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].legend(frameon=False, title="")
    fig.suptitle("Exp5/Exp6: preprocessing helps lightly; NCA gives the clearest robustness gain")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return savefig(fig, "fig_exp5_exp6_jepa300_mae300_mitigation.png")


def load_method_noise() -> pd.DataFrame:
    frames = []
    for model, cfg in MODELS.items():
        linear = load_exp1_noise().query("model == @model")[["condition", "sigma", "macro_auroc", "auroc_drop", "cosine_drift"]].copy()
        linear["method"] = "Linear"
        mlp = pd.read_csv(cfg["exp7"] / "mlp_probe_results.csv")
        mlp["sigma"] = mlp["condition"].map(parse_sigma)
        mlp["method"] = "MLP"
        pft = pd.read_csv(cfg["exp8"] / "partial_ft_results.csv")
        pft["sigma"] = pft["condition"].map(parse_sigma)
        pft["method"] = "Partial FT"
        df = pd.concat([linear, mlp, pft], ignore_index=True)
        df = df[df["sigma"].isin(SIGMA_ORDER)].copy()
        df["model"] = model
        df["short"] = cfg["short"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_exp7_exp8_capacity() -> Path:
    df = load_method_noise()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        sns.lineplot(data=sub, x="sigma", y="macro_auroc", hue="method", style="method", markers=True, dashes=False, ax=ax)
        ax.set_title(MODELS[model]["short"])
        ax.set_xlabel("Gaussian noise sigma")
        ax.set_ylabel("Macro AUROC")
        ax.set_xticks(SIGMA_ORDER)
        ax.set_ylim(0.42, 0.98)
        ax.legend(frameon=False, title="")
    fig.suptitle("Exp7/Exp8: more supervised capacity improves clean AUROC but does not remove the JEPA300 noise jump")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return savefig(fig, "fig_exp7_exp8_jepa300_mae300_capacity.png")


def plot_dashboard() -> Path:
    exp1 = load_exp1_noise()
    exp2 = load_exp2b_overall()
    exp3 = load_exp3_key_conditions()
    method = load_method_noise()

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    for model, sub in exp1.groupby("model"):
        axes[0, 0].plot(sub["sigma"], sub["macro_auroc"], marker="o", color=MODELS[model]["color"], label=MODELS[model]["short"])
    axes[0, 0].set_title("Exp1 Gaussian-noise AUROC")
    axes[0, 0].set_xlabel("sigma")
    axes[0, 0].set_ylabel("Macro AUROC")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(exp2["short"], exp2["delta_logit_drop_mean"], color=[MODELS[m]["color"] for m in exp2["model"]])
    axes[0, 1].set_title("Exp2b class-aligned lesion sensitivity")
    axes[0, 1].set_ylabel("Delta logit drop")

    mid = exp3[exp3["label"].astype(str) == "Band 0.20-0.45"]
    axes[1, 0].bar(mid["short"], mid["macro_auroc"], color=[MODELS[m]["color"] for m in mid["model"]])
    axes[1, 0].set_title("Exp3 mid-frequency corruption")
    axes[1, 0].set_ylabel("Macro AUROC")
    axes[1, 0].set_ylim(0.50, 0.80)

    sigma005 = method[method["sigma"] == 0.05]
    sns.barplot(data=sigma005, x="method", y="macro_auroc", hue="short", palette=[MODELS[m]["color"] for m in MODELS], ax=axes[1, 1])
    axes[1, 1].set_title("Exp7/8 capacity at sigma=0.05")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Macro AUROC")
    axes[1, 1].legend(frameon=False, title="")

    fig.suptitle("JEPA300 vs MAE300 fair-comparison dashboard")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return savefig(fig, "fig0_jepa300_fair_dashboard.png")


def write_manifest(paths: list[Path]) -> None:
    lines = ["# JEPA300 Report Visualization Manifest", ""]
    captions = {
        "fig_input_noise_occlusion_examples.png": "Input-level visual examples of Gaussian noise and class-aligned lesion/control occlusion.",
        "fig0_jepa300_fair_dashboard.png": "One-page visual summary of the fair JEPA300-vs-MAE300 rerun.",
        "fig_exp1_jepa300_mae300_noise.png": "Gaussian-noise AUROC, drop, and drift curves for Exp1.",
        "fig_exp2b_jepa300_mae300_lesion.png": "Overall and per-class class-aligned lesion sensitivity for Exp2b.",
        "fig_exp3_jepa300_mae300_frequency.png": "Frequency-domain vulnerability comparison for Exp3.",
        "fig_exp4_jepa300_mae300_mechanism.png": "Mechanism metrics by disease group for Exp4.",
        "fig_exp5_exp6_jepa300_mae300_mitigation.png": "Preprocessing and NCA mitigation visual summary for Exp5/Exp6.",
        "fig_exp7_exp8_jepa300_mae300_capacity.png": "Linear, MLP, and partial-finetuning Gaussian-noise curves for Exp7/Exp8.",
    }
    for path in paths:
        lines.append(f"- `{path.relative_to(ROOT).as_posix()}`: {captions[path.name]}")
    (OUT / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    ensure_out()
    paths = [
        plot_input_perturbation_examples(),
        plot_dashboard(),
        plot_exp1_noise(),
        plot_exp2b_lesion(),
        plot_exp3_frequency(),
        plot_exp4_mechanism(),
        plot_exp5_exp6_mitigation(),
        plot_exp7_exp8_capacity(),
    ]
    write_manifest(paths)
    print("Generated report visualizations:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
