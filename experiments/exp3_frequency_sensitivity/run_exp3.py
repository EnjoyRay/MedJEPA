"""Run Exp3 frequency sensitivity analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp1_robustness'))

from exp1_robustness.linear_probe import LinearProbe
from frequency_perturbations import get_frequency_schedule
from shared.metrics import compute_auroc, compute_cosine_drift, compute_f1, compute_l2_drift
from shared.model_wrappers import build_encoder
from shared.vindr_dataset import VINDR_CLASSES, build_vindr_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp3: frequency sensitivity")
    parser.add_argument("--model", required=True, choices=["ijepa", "mae", "moco"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--probe_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--in_chans", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap_iters", default=1000, type=int)
    parser.add_argument("--max_batches", default=None, type=int)
    parser.add_argument(
        "--noise_repeats",
        default=1,
        type=int,
        help="Repeat stochastic Gaussian-noise conditions and report mean/std across repeats.",
    )
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def _optimal_macro_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 19)
    per_class = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue
        best = 0.0
        for thr in thresholds:
            best = max(best, compute_f1(labels[:, [c]], (probs[:, [c]] >= thr).astype(np.float32)))
        per_class.append(best)
    return float(np.mean(per_class)) if per_class else float("nan")


def _bootstrap_macro_ci(labels: np.ndarray, probs: np.ndarray, iters: int, seed: int = 42) -> Tuple[float, float]:
    if iters <= 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    values: List[float] = []
    n = len(labels)
    for _ in range(iters):
        idx = rng.randint(0, n, n)
        try:
            values.append(compute_auroc(labels[idx], probs[idx])["macro_auroc"])
        except Exception:
            continue
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


@torch.no_grad()
def _extract(
    encoder: torch.nn.Module,
    loader,
    device: torch.device,
    perturb: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_batches: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    embs, labels = [], []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if perturb is not None:
            images_01 = (images * 0.5 + 0.5).clamp(0.0, 1.0)
            images = (perturb(images_01) - 0.5) / 0.5
        embs.append(encoder(images).cpu())
        labels.append(batch["label"])
    return torch.cat(embs).float().numpy(), torch.cat(labels).float().numpy()


def _evaluate_condition(
    labels: np.ndarray,
    clean_embs: np.ndarray,
    embs: np.ndarray,
    probe: LinearProbe,
    clean_macro: float,
    bootstrap_iters: int,
) -> Dict[str, float]:
    probs = probe.predict_proba(embs)
    preds = (probs >= 0.5).astype(np.float32)
    auroc = compute_auroc(labels, probs)
    ci_low, ci_high = _bootstrap_macro_ci(labels, probs, bootstrap_iters)
    return {
        "macro_auroc": auroc["macro_auroc"],
        "macro_auroc_ci95_low": ci_low,
        "macro_auroc_ci95_high": ci_high,
        "f1": compute_f1(labels, preds),
        "optimal_f1": _optimal_macro_f1(labels, probs),
        "auroc_drop": clean_macro - auroc["macro_auroc"],
        "cosine_drift": compute_cosine_drift(torch.from_numpy(clean_embs), torch.from_numpy(embs)),
        "l2_drift": compute_l2_drift(torch.from_numpy(clean_embs), torch.from_numpy(embs)),
        **{f"auroc_{VINDR_CLASSES[c]}": auroc.get(f"class_{c}", float("nan")) for c in range(len(VINDR_CLASSES))},
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device)
    encoder = build_encoder(args.model, args.weights, device)
    encoder.eval()

    loader = build_vindr_loader(
        args.data_dir,
        split="test",
        img_size=args.img_size,
        in_chans=args.in_chans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    clean_embs, labels = _extract(encoder, loader, device, max_batches=args.max_batches)
    probe = LinearProbe(clean_embs.shape[1], labels.shape[1], device=device)
    probe.load(args.probe_path)

    clean_probs = probe.predict_proba(clean_embs)
    clean_auroc = compute_auroc(labels, clean_probs)
    clean_macro = clean_auroc["macro_auroc"]
    rows = [{
        "condition": "clean",
        "perturbation": "clean",
        "level": "clean",
        "macro_auroc": clean_macro,
        "macro_auroc_ci95_low": _bootstrap_macro_ci(labels, clean_probs, args.bootstrap_iters)[0],
        "macro_auroc_ci95_high": _bootstrap_macro_ci(labels, clean_probs, args.bootstrap_iters)[1],
        "f1": compute_f1(labels, (clean_probs >= 0.5).astype(np.float32)),
        "optimal_f1": _optimal_macro_f1(labels, clean_probs),
        "auroc_drop": 0.0,
        "cosine_drift": 0.0,
        "l2_drift": 0.0,
        **{f"auroc_{VINDR_CLASSES[c]}": clean_auroc.get(f"class_{c}", float("nan")) for c in range(len(VINDR_CLASSES))},
    }]

    for pert_type, levels in get_frequency_schedule().items():
        for level, fn in levels:
            print(f"Evaluating {pert_type}/{level}", flush=True)
            repeats = max(1, args.noise_repeats if pert_type == "gaussian_noise" else 1)
            repeat_rows: List[Dict[str, float]] = []
            for repeat_idx in range(repeats):
                if pert_type == "gaussian_noise":
                    torch.manual_seed(args.seed + 1009 * repeat_idx)
                    np.random.seed(args.seed + 1009 * repeat_idx)
                pert_embs, _ = _extract(encoder, loader, device, perturb=fn, max_batches=args.max_batches)
                metrics = _evaluate_condition(labels, clean_embs, pert_embs, probe, clean_macro, args.bootstrap_iters)
                repeat_rows.append(metrics)
                if repeats > 1:
                    rows.append({
                        "condition": f"{pert_type}/{level}",
                        "perturbation": pert_type,
                        "level": level,
                        "repeat": repeat_idx,
                        "repeat_count": repeats,
                        **metrics,
                    })
            if repeats == 1:
                rows.append({
                    "condition": f"{pert_type}/{level}",
                    "perturbation": pert_type,
                    "level": level,
                    "repeat": 0,
                    "repeat_count": 1,
                    **repeat_rows[0],
                })
            else:
                agg: Dict[str, float] = {}
                metric_keys = repeat_rows[0].keys()
                for key in metric_keys:
                    vals = np.array([r[key] for r in repeat_rows], dtype=np.float64)
                    agg[key] = float(np.nanmean(vals))
                    agg[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
                rows.append({
                    "condition": f"{pert_type}/{level}",
                    "perturbation": pert_type,
                    "level": level,
                    "repeat": "mean",
                    "repeat_count": repeats,
                    **agg,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out / "frequency_sensitivity.csv", index=False)
    mean_df = df[(df["repeat"] == "mean") | (df["repeat_count"] == 1)].copy()
    mean_df.to_csv(out / "frequency_sensitivity_summary.csv", index=False)
    print(out / "frequency_sensitivity.csv")
    print(out / "frequency_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
