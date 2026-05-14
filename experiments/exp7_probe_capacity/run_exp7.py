"""Run Exp7: frozen encoder with a non-linear MLP probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp1_robustness"))

from evaluate_robustness import evaluate_robustness
from linear_probe import MLPProbe
from run_exp1 import extract_all_embeddings
from shared.model_wrappers import build_encoder
from shared.vindr_dataset import build_vindr_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp7: MLP probe capacity")
    parser.add_argument("--model", required=True, choices=["ijepa", "mae"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--in_chans", default=1, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--probe_epochs", default=50, type=int)
    parser.add_argument("--probe_lr", default=1e-3, type=float)
    parser.add_argument("--probe_wd", default=1e-4, type=float)
    parser.add_argument("--probe_batch_size", default=256, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    return parser.parse_args()


def _results_to_rows(results: dict) -> list[dict]:
    rows = []
    clean_auc = float(results["clean"]["auroc"]["macro_auroc"])
    rows.append(
        {
            "condition": "clean",
            "perturbation": "clean",
            "x_value": 0.0,
            "macro_auroc": clean_auc,
            "f1": float(results["clean"]["f1"]),
            "auroc_drop": 0.0,
            "cosine_drift": 0.0,
            "l2_drift": 0.0,
        }
    )
    for key, val in results.items():
        if key == "clean":
            continue
        pert, level = key.split("/", 1)
        rows.append(
            {
                "condition": key,
                "perturbation": pert,
                "x_value": float(level.split("=")[-1]),
                "macro_auroc": float(val["auroc"]["macro_auroc"]),
                "f1": float(val["f1"]),
                "auroc_drop": float(val["auroc_drop"]),
                "cosine_drift": float(val["cosine_drift"]),
                "l2_drift": float(val["l2_drift"]),
            }
        )
    return rows


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

    train_loader = build_vindr_loader(
        args.data_dir, "train", args.img_size, args.in_chans, args.batch_size, args.num_workers
    )
    test_loader = build_vindr_loader(
        args.data_dir, "test", args.img_size, args.in_chans, args.batch_size, args.num_workers
    )

    train_embs, train_labels = extract_all_embeddings(
        encoder,
        train_loader,
        device,
        save_path=str(out / "train_cache"),
        max_batches=args.max_train_batches,
    )
    probe = MLPProbe(
        feat_dim=train_embs.shape[1],
        num_classes=train_labels.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.probe_lr,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        weight_decay=args.probe_wd,
        device=device,
    )
    probe.fit(train_embs, train_labels)
    probe.save(str(out / "mlp_probe.pth"))

    results = evaluate_robustness(
        encoder=encoder,
        probe=probe,
        loader=test_loader,
        device=device,
        perturbation_types=["gaussian_noise"],
        max_batches=args.max_eval_batches,
    )
    rows = _results_to_rows(results)
    pd.DataFrame(rows).to_csv(out / "mlp_probe_results.csv", index=False)
    print(out / "mlp_probe_results.csv", flush=True)


if __name__ == "__main__":
    main()
