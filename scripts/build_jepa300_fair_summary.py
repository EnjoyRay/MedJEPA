"""Build a compact completeness/metric summary for JEPA300 fair-comparison runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPERIMENT_FILES = {
    "exp1": ("exp1_{slug}", "robustness_curve.csv"),
    "exp2b": ("exp2b_class_aligned_{slug}", "class_aligned_overall.csv"),
    "exp3": ("exp3_frequency_{slug}", "frequency_sensitivity.csv"),
    "exp4": ("exp4_mechanism_{slug}", "mechanism_per_sample.csv"),
    "exp5": ("exp5_mitigation_{slug}", "mitigation_results.csv"),
    "exp6": ("exp6_nca_{slug}", "nca_results.csv"),
    "exp7": ("exp7_mlp_probe_{slug}", "mlp_probe_results.csv"),
    "exp8": ("exp8_partial_ft_{slug}", "partial_ft_results.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_matrix", required=True, help="Pipe-separated model matrix from run_jepa300_fair_uic.sh")
    parser.add_argument("--output", default="results/jepa300_fair_summary.csv")
    return parser.parse_args()


def load_models(path: Path) -> list[dict[str, str]]:
    models = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, model_type, slug, weights, epoch, posttrain_type = line.split("|", 5)
        models.append(
            {
                "model": name,
                "model_type": model_type,
                "slug": slug,
                "weights": weights,
                "epoch": epoch,
                "posttrain_type": posttrain_type,
            }
        )
    return models


def add_rows(rows: list[dict], base: dict, experiment: str, csv_path: Path) -> None:
    common = {**base, "experiment": experiment, "result_file": str(csv_path), "status": "complete"}
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        rows.append({**common, "status": "read_error", "metric": "error", "mean": str(exc), "std": "", "seed": ""})
        return

    if experiment in {"exp1", "exp3", "exp7", "exp8"} and "macro_auroc" in df.columns:
        group_cols = [c for c in ["condition", "perturbation", "level"] if c in df.columns]
        for _, row in df.iterrows():
            label = "/".join(str(row[c]) for c in group_cols if pd.notna(row.get(c))) or "overall"
            rows.append({**common, "metric": f"macro_auroc:{label}", "mean": row["macro_auroc"], "std": "", "seed": ""})
        return
    if experiment == "exp2b":
        for col in ["mean_delta", "delta_mean", "lesion_minus_control", "p_value", "n"]:
            if col in df.columns:
                rows.append({**common, "metric": col, "mean": df[col].iloc[0], "std": "", "seed": ""})
        if not any(r["experiment"] == experiment and r["model"] == base["model"] for r in rows):
            rows.append({**common, "metric": "rows", "mean": len(df), "std": "", "seed": ""})
        return
    if experiment == "exp4":
        numeric_cols = [c for c in df.select_dtypes("number").columns if c not in {"image_idx", "sample_idx"}]
        for col in numeric_cols[:12]:
            rows.append({**common, "metric": col, "mean": df[col].mean(), "std": df[col].std(), "seed": ""})
        return
    if experiment in {"exp5", "exp6"}:
        value_col = "macro_auroc" if "macro_auroc" in df.columns else None
        if value_col:
            group_cols = [c for c in ["condition", "probe", "mitigation", "ablation"] if c in df.columns]
            for _, row in df.iterrows():
                label = "/".join(str(row[c]) for c in group_cols if pd.notna(row.get(c))) or "overall"
                rows.append({**common, "metric": f"{value_col}:{label}", "mean": row[value_col], "std": "", "seed": ""})
            return
    rows.append({**common, "metric": "rows", "mean": len(df), "std": "", "seed": ""})


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows: list[dict] = []
    for model in load_models(Path(args.model_matrix)):
        base = {
            "model": model["model"],
            "epoch": model["epoch"],
            "pretrain_type": model["model_type"],
            "posttrain_type": model["posttrain_type"],
            "slug": model["slug"],
        }
        for experiment, (dir_tpl, filename) in EXPERIMENT_FILES.items():
            csv_path = results_dir / dir_tpl.format(slug=model["slug"]) / filename
            if csv_path.exists():
                add_rows(rows, base, experiment, csv_path)
            else:
                rows.append(
                    {
                        **base,
                        "experiment": experiment,
                        "metric": "missing",
                        "mean": "",
                        "std": "",
                        "seed": "",
                        "result_file": str(csv_path),
                        "status": "missing",
                    }
                )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
