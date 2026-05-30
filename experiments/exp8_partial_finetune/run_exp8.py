"""Run Exp8: clean-only partial fine-tuning with the last two ViT blocks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.metrics import compute_auroc, compute_cosine_drift, compute_f1, compute_l2_drift
from shared.model_wrappers import build_encoder
from shared.perturbations import gaussian_noise
from shared.vindr_dataset import build_vindr_loader


NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp8: partial fine-tuning")
    parser.add_argument("--model", required=True, choices=["ijepa", "mae", "moco"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--in_chans", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--lr_encoder", default=1e-5, type=float)
    parser.add_argument("--lr_head", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--unfreeze_blocks", default=2, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    return parser.parse_args()


class FineTuneClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.head(features), features


def _infer_dims(encoder: nn.Module, loader: DataLoader, device: torch.device) -> tuple[int, int]:
    batch = next(iter(loader))
    with torch.no_grad():
        feat = encoder(batch["image"].to(device))
    return int(feat.shape[1]), int(batch["label"].shape[1])


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def configure_partial_finetune(encoder: nn.Module, n_blocks: int) -> list[str]:
    _set_trainable(encoder, False)
    trainable = []
    if hasattr(encoder, "vit") and hasattr(encoder.vit, "blocks"):
        blocks = encoder.vit.blocks
        for blk in blocks[-n_blocks:]:
            _set_trainable(blk, True)
        trainable.append(f"vit.blocks[-{n_blocks}:]")
        for name in ("norm", "fc_norm"):
            mod = getattr(encoder.vit, name, None)
            if mod is not None:
                _set_trainable(mod, True)
                trainable.append(f"vit.{name}")
    elif hasattr(encoder, "mae") and hasattr(encoder.mae, "blocks"):
        blocks = encoder.mae.blocks
        for blk in blocks[-n_blocks:]:
            _set_trainable(blk, True)
        trainable.append(f"mae.blocks[-{n_blocks}:]")
        if hasattr(encoder.mae, "norm"):
            _set_trainable(encoder.mae.norm, True)
            trainable.append("mae.norm")
    else:
        raise ValueError("Cannot locate ViT blocks for partial fine-tuning.")
    return trainable


def _as_01(images: torch.Tensor) -> torch.Tensor:
    return (images * 0.5 + 0.5).clamp(0.0, 1.0)


def _from_01(images: torch.Tensor) -> torch.Tensor:
    return (images - 0.5) / 0.5


def train_one_epoch(
    model: FineTuneClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        bs = images.shape[0]
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate_condition(
    model: FineTuneClassifier,
    loader: DataLoader,
    device: torch.device,
    sigma: float | None,
    max_batches: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all, labels_all, feats_all = [], [], []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if sigma is not None:
            images = _from_01(gaussian_noise(_as_01(images), sigma))
        logits, feats = model(images)
        logits_all.append(logits.cpu())
        feats_all.append(feats.cpu())
        labels_all.append(batch["label"].cpu())
    return (
        torch.cat(logits_all).float().numpy(),
        torch.cat(labels_all).float().numpy(),
        torch.cat(feats_all).float().numpy(),
    )


def evaluate_all(model: FineTuneClassifier, loader: DataLoader, device: torch.device, max_batches: int | None) -> list[dict]:
    clean_logits, labels, clean_feats = evaluate_condition(model, loader, device, None, max_batches)
    clean_probs = 1.0 / (1.0 + np.exp(-clean_logits))
    clean_auc = compute_auroc(labels, clean_probs)["macro_auroc"]
    rows = [
        {
            "condition": "clean",
            "perturbation": "clean",
            "x_value": 0.0,
            "macro_auroc": clean_auc,
            "f1": compute_f1(labels, (clean_probs >= 0.5).astype(np.float32)),
            "auroc_drop": 0.0,
            "cosine_drift": 0.0,
            "l2_drift": 0.0,
        }
    ]
    clean_feats_t = torch.from_numpy(clean_feats)
    for sigma in NOISE_LEVELS:
        logits, labels2, feats = evaluate_condition(model, loader, device, sigma, max_batches)
        if not np.array_equal(labels, labels2):
            raise RuntimeError("Label order changed during noisy evaluation.")
        probs = 1.0 / (1.0 + np.exp(-logits))
        auc = compute_auroc(labels, probs)["macro_auroc"]
        feats_t = torch.from_numpy(feats)
        rows.append(
            {
                "condition": f"gaussian_noise/sigma={sigma:.2f}",
                "perturbation": "gaussian_noise",
                "x_value": sigma,
                "macro_auroc": auc,
                "f1": compute_f1(labels, (probs >= 0.5).astype(np.float32)),
                "auroc_drop": clean_auc - auc,
                "cosine_drift": compute_cosine_drift(clean_feats_t, feats_t),
                "l2_drift": compute_l2_drift(clean_feats_t, feats_t),
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
    train_loader = build_vindr_loader(
        args.data_dir, "train", args.img_size, args.in_chans, args.batch_size, args.num_workers, shuffle=True
    )
    test_loader = build_vindr_loader(
        args.data_dir, "test", args.img_size, args.in_chans, args.batch_size, args.num_workers
    )
    feat_dim, num_classes = _infer_dims(encoder, train_loader, device)
    trainable = configure_partial_finetune(encoder, args.unfreeze_blocks)
    model = FineTuneClassifier(encoder, feat_dim, num_classes).to(device)

    head_params = list(model.head.parameters())
    encoder_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr_encoder},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    logs = []
    best_auc = -float("inf")
    best_path = out / "partial_ft_best.pth"

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.max_train_batches)
        rows = evaluate_all(model, test_loader, device, args.max_eval_batches)
        clean_auc = rows[0]["macro_auroc"]
        logs.append({"epoch": epoch + 1, "train_loss": loss, "clean_macro_auroc": clean_auc})
        print(f"[Exp8] epoch {epoch+1}/{args.epochs} loss={loss:.4f} clean_auc={clean_auc:.4f}", flush=True)
        if clean_auc > best_auc:
            best_auc = clean_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "feat_dim": feat_dim,
                    "num_classes": num_classes,
                    "trainable": trainable,
                    "best_epoch": epoch + 1,
                    "best_clean_macro_auroc": best_auc,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    rows = evaluate_all(model, test_loader, device, args.max_eval_batches)
    pd.DataFrame(rows).to_csv(out / "partial_ft_results.csv", index=False)
    pd.DataFrame(logs).to_csv(out / "partial_ft_training_log.csv", index=False)
    print(out / "partial_ft_results.csv", flush=True)


if __name__ == "__main__":
    main()
