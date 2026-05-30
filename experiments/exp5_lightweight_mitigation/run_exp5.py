"""Run Exp5 lightweight mitigation analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp1_robustness'))

from denoise import get_denoise_methods, get_eval_conditions, get_robust_train_augmentations
from exp1_robustness.linear_probe import LinearProbe
from shared.metrics import compute_auroc, compute_cosine_drift, compute_f1, compute_l2_drift
from shared.model_wrappers import build_encoder
from shared.vindr_dataset import build_vindr_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp5: denoise preprocessing and robust probe")
    parser.add_argument("--model", required=True, choices=["ijepa", "mae", "moco"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--probe_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--in_chans", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--probe_epochs", default=50, type=int)
    parser.add_argument("--probe_lr", default=1e-3, type=float)
    parser.add_argument("--probe_wd", default=1e-4, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    return parser.parse_args()


@torch.no_grad()
def _extract(
    encoder,
    loader,
    device: torch.device,
    transform_01: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_batches: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    embs, labels = [], []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if transform_01 is not None:
            images_01 = (images * 0.5 + 0.5).clamp(0.0, 1.0)
            images = (transform_01(images_01) - 0.5) / 0.5
        embs.append(encoder(images).cpu())
        labels.append(batch["label"])
    return torch.cat(embs).float().numpy(), torch.cat(labels).float().numpy()


def _eval_probe(name: str, probe: LinearProbe, labels: np.ndarray, clean_embs: np.ndarray, embs: np.ndarray) -> dict:
    probs = probe.predict_proba(embs)
    auroc = compute_auroc(labels, probs)["macro_auroc"]
    return {
        "setting": name,
        "macro_auroc": auroc,
        "f1": compute_f1(labels, (probs >= 0.5).astype(np.float32)),
        "cosine_drift": compute_cosine_drift(torch.from_numpy(clean_embs), torch.from_numpy(embs)),
        "l2_drift": compute_l2_drift(torch.from_numpy(clean_embs), torch.from_numpy(embs)),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device)
    encoder = build_encoder(args.model, args.weights, device)
    encoder.eval()

    train_loader = build_vindr_loader(
        args.data_dir, split="train", img_size=args.img_size, in_chans=args.in_chans,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    test_loader = build_vindr_loader(
        args.data_dir, split="test", img_size=args.img_size, in_chans=args.in_chans,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    clean_test_embs, test_labels = _extract(encoder, test_loader, device, max_batches=args.max_eval_batches)
    clean_probe = LinearProbe(clean_test_embs.shape[1], test_labels.shape[1], device=device)
    clean_probe.load(args.probe_path)

    rows = []
    for condition_name, condition_fn in get_eval_conditions():
        if condition_name == "clean":
            cond_embs = clean_test_embs
        else:
            cond_embs, _ = _extract(encoder, test_loader, device, transform_01=condition_fn, max_batches=args.max_eval_batches)
        rows.append({"probe": "clean_probe", "mitigation": "none", "condition": condition_name,
                     **_eval_probe("clean_probe", clean_probe, test_labels, clean_test_embs, cond_embs)})

    # Test-time denoising for noisy images only.
    for denoise_name, denoise_fn in get_denoise_methods().items():
        if denoise_name == "none":
            continue
        for sigma in (0.05, 0.10):
            def noisy_then_denoise(x, sigma=sigma, denoise_fn=denoise_fn):
                return denoise_fn((x + torch.randn_like(x) * sigma).clamp(0.0, 1.0))
            cond_embs, _ = _extract(encoder, test_loader, device, transform_01=noisy_then_denoise, max_batches=args.max_eval_batches)
            rows.append({"probe": "clean_probe", "mitigation": denoise_name, "condition": f"noise_sigma={sigma:.2f}",
                         **_eval_probe("clean_probe", clean_probe, test_labels, clean_test_embs, cond_embs)})

    # Robust probe training on augmented frozen embeddings.
    train_embs_all, train_labels_all = [], []
    for aug_name, aug_fn in get_robust_train_augmentations():
        print(f"Extracting robust-probe train embeddings: {aug_name}", flush=True)
        embs, labels = _extract(encoder, train_loader, device, transform_01=aug_fn, max_batches=args.max_train_batches)
        train_embs_all.append(embs)
        train_labels_all.append(labels)
    train_embs = np.concatenate(train_embs_all, axis=0)
    train_labels = np.concatenate(train_labels_all, axis=0)
    robust_probe = LinearProbe(
        clean_test_embs.shape[1],
        test_labels.shape[1],
        lr=args.probe_lr,
        epochs=args.probe_epochs,
        weight_decay=args.probe_wd,
        device=device,
    )
    robust_probe.fit(train_embs, train_labels)
    robust_probe.save(str(out / "robust_probe.pth"))

    for condition_name, condition_fn in get_eval_conditions():
        if condition_name == "clean":
            cond_embs = clean_test_embs
        else:
            cond_embs, _ = _extract(encoder, test_loader, device, transform_01=condition_fn, max_batches=args.max_eval_batches)
        rows.append({"probe": "robust_probe", "mitigation": "train_aug", "condition": condition_name,
                     **_eval_probe("robust_probe", robust_probe, test_labels, clean_test_embs, cond_embs)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "mitigation_results.csv", index=False)
    print(out / "mitigation_results.csv")


if __name__ == "__main__":
    main()
