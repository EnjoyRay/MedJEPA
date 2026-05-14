"""Run Exp6: Noise-Consistent Adapter (NCA)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp5_lightweight_mitigation"))

from adapter import ResidualAdapterClassifier
from denoise import brightness, contrast, gaussian_noise, get_eval_conditions
from shared.metrics import compute_auroc, compute_cosine_drift, compute_f1, compute_l2_drift
from shared.model_wrappers import build_encoder
from shared.vindr_dataset import VINDR_CLASSES, build_vindr_loader


ABLATIONS = {
    "full": {
        "pred_weight": 0.5,
        "repr_weight": 0.2,
        "augmentations": ["noise_005", "noise_010", "brightness_neg", "brightness_pos", "contrast_low", "contrast_high"],
    },
    "no_repr": {
        "pred_weight": 0.5,
        "repr_weight": 0.0,
        "augmentations": ["noise_005", "noise_010", "brightness_neg", "brightness_pos", "contrast_low", "contrast_high"],
    },
    "no_pred": {
        "pred_weight": 0.0,
        "repr_weight": 0.2,
        "augmentations": ["noise_005", "noise_010", "brightness_neg", "brightness_pos", "contrast_low", "contrast_high"],
    },
    "noise_only": {
        "pred_weight": 0.5,
        "repr_weight": 0.2,
        "augmentations": ["noise_005", "noise_010"],
    },
    "full_aug": {
        "pred_weight": 0.5,
        "repr_weight": 0.2,
        "augmentations": ["noise_005", "noise_010", "brightness_neg", "brightness_pos", "contrast_low", "contrast_high"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp6: Noise-Consistent Adapter")
    parser.add_argument("--model", required=True, choices=["ijepa", "mae"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--in_chans", default=1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--adapter_batch_size", default=256, type=int)
    parser.add_argument("--adapter_epochs", default=30, type=int)
    parser.add_argument("--adapter_lr", default=1e-3, type=float)
    parser.add_argument("--adapter_wd", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--clean_weight", default=1.0, type=float)
    parser.add_argument("--aug_weight", default=1.0, type=float)
    parser.add_argument("--pred_weight", default=0.5, type=float)
    parser.add_argument("--repr_weight", default=0.2, type=float)
    parser.add_argument("--ablations", default="full", help="Comma-separated ablations or 'all'.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def _train_augmentations() -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        "noise_005": lambda x: gaussian_noise(x, 0.05),
        "noise_010": lambda x: gaussian_noise(x, 0.10),
        "brightness_neg": lambda x: brightness(x, -0.10),
        "brightness_pos": lambda x: brightness(x, +0.10),
        "contrast_low": lambda x: contrast(x, 0.50),
        "contrast_high": lambda x: contrast(x, 1.50),
    }


@torch.no_grad()
def extract_embeddings(
    encoder: torch.nn.Module,
    loader,
    device: torch.device,
    transform_01: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    embs, labels = [], []
    encoder.eval()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if transform_01 is not None:
            images_01 = (images * 0.5 + 0.5).clamp(0.0, 1.0)
            images = (transform_01(images_01) - 0.5) / 0.5
        embs.append(encoder(images).detach().cpu())
        labels.append(batch["label"].detach().cpu())
    return torch.cat(embs).float().numpy(), torch.cat(labels).float().numpy()


def _paired_train_arrays(
    clean_embs: np.ndarray,
    labels: np.ndarray,
    aug_embs_by_name: dict[str, np.ndarray],
    aug_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_pairs, aug_pairs, label_pairs = [], [], []
    for name in aug_names:
        aug = aug_embs_by_name[name]
        clean_pairs.append(clean_embs)
        aug_pairs.append(aug)
        label_pairs.append(labels)
    return (
        np.concatenate(clean_pairs, axis=0),
        np.concatenate(aug_pairs, axis=0),
        np.concatenate(label_pairs, axis=0),
    )


def train_adapter(
    clean_embs: np.ndarray,
    labels: np.ndarray,
    aug_embs_by_name: dict[str, np.ndarray],
    ablation_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ResidualAdapterClassifier, list[dict]]:
    cfg = dict(ABLATIONS[ablation_name])
    if ablation_name == "full":
        cfg["pred_weight"] = args.pred_weight
        cfg["repr_weight"] = args.repr_weight
    clean_pairs, aug_pairs, label_pairs = _paired_train_arrays(
        clean_embs, labels, aug_embs_by_name, list(cfg["augmentations"])
    )
    dataset = TensorDataset(
        torch.from_numpy(clean_pairs).float(),
        torch.from_numpy(aug_pairs).float(),
        torch.from_numpy(label_pairs).float(),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.adapter_batch_size, shuffle=True, generator=generator)

    model = ResidualAdapterClassifier(clean_embs.shape[1], labels.shape[1], dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.adapter_lr, weight_decay=args.adapter_wd)
    bce = torch.nn.BCEWithLogitsLoss()
    logs = []

    model.train()
    for epoch in range(args.adapter_epochs):
        totals = {"loss": 0.0, "clean_bce": 0.0, "aug_bce": 0.0, "pred_cons": 0.0, "repr_cons": 0.0}
        n = 0
        for clean_x, aug_x, y in loader:
            clean_x = clean_x.to(device, non_blocking=True)
            aug_x = aug_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            clean_logits, clean_z = model(clean_x)
            aug_logits, aug_z = model(aug_x)

            clean_bce = bce(clean_logits, y)
            aug_bce = bce(aug_logits, y)
            clean_prob = torch.sigmoid(clean_logits.detach())
            aug_log_prob = F.logsigmoid(aug_logits)
            aug_log_one_minus = F.logsigmoid(-aug_logits)
            pred_cons = -(clean_prob * aug_log_prob + (1.0 - clean_prob) * aug_log_one_minus).mean()
            repr_cons = 1.0 - F.cosine_similarity(clean_z, aug_z, dim=-1).mean()
            loss = (
                args.clean_weight * clean_bce
                + args.aug_weight * aug_bce
                + float(cfg["pred_weight"]) * pred_cons
                + float(cfg["repr_weight"]) * repr_cons
            )
            loss.backward()
            opt.step()

            bs = clean_x.shape[0]
            n += bs
            totals["loss"] += float(loss.item()) * bs
            totals["clean_bce"] += float(clean_bce.item()) * bs
            totals["aug_bce"] += float(aug_bce.item()) * bs
            totals["pred_cons"] += float(pred_cons.item()) * bs
            totals["repr_cons"] += float(repr_cons.item()) * bs

        row = {"ablation": ablation_name, "epoch": epoch + 1, **{k: v / max(n, 1) for k, v in totals.items()}}
        logs.append(row)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[NCA:{ablation_name}] epoch {epoch+1}/{args.adapter_epochs} "
                f"loss={row['loss']:.4f} clean={row['clean_bce']:.4f} aug={row['aug_bce']:.4f} "
                f"pred={row['pred_cons']:.4f} repr={row['repr_cons']:.4f}",
                flush=True,
            )
    model.eval()
    return model, logs


@torch.no_grad()
def predict_adapter(
    model: ResidualAdapterClassifier,
    embs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = TensorDataset(torch.from_numpy(embs).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits_all, z_all = [], []
    model.eval()
    for (xb,) in loader:
        logits, z = model(xb.to(device, non_blocking=True))
        logits_all.append(logits.cpu())
        z_all.append(z.cpu())
    logits_np = torch.cat(logits_all).float().numpy()
    z_np = torch.cat(z_all).float().numpy()
    return logits_np, z_np


def evaluate_ablation(
    model: ResidualAdapterClassifier,
    ablation: str,
    labels: np.ndarray,
    clean_embs: np.ndarray,
    condition_embs: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], list[dict]]:
    clean_logits, clean_z = predict_adapter(model, clean_embs, device, batch_size)
    clean_probs = 1.0 / (1.0 + np.exp(-clean_logits))
    rows, per_class_rows = [], []
    for condition, embs in condition_embs.items():
        logits, z = (clean_logits, clean_z) if condition == "clean" else predict_adapter(model, embs, device, batch_size)
        probs = 1.0 / (1.0 + np.exp(-logits))
        aucs = compute_auroc(labels, probs)
        row = {
            "ablation": ablation,
            "condition": condition,
            "macro_auroc": aucs["macro_auroc"],
            "f1": compute_f1(labels, (probs >= 0.5).astype(np.float32)),
            "raw_cosine_drift": 0.0 if condition == "clean" else compute_cosine_drift(torch.from_numpy(clean_embs), torch.from_numpy(embs)),
            "adapter_cosine_drift": 0.0 if condition == "clean" else compute_cosine_drift(torch.from_numpy(clean_z), torch.from_numpy(z)),
            "adapter_l2_drift": 0.0 if condition == "clean" else compute_l2_drift(torch.from_numpy(clean_z), torch.from_numpy(z)),
        }
        rows.append(row)
        for idx, class_name in enumerate(VINDR_CLASSES):
            per_class_rows.append(
                {
                    "ablation": ablation,
                    "condition": condition,
                    "class_idx": idx,
                    "class_name": class_name,
                    "auroc": aucs.get(f"class_{idx}", float("nan")),
                }
            )
    clean_auc = next(r["macro_auroc"] for r in rows if r["condition"] == "clean")
    for row in rows:
        row["auroc_drop"] = clean_auc - row["macro_auroc"]
    return rows, per_class_rows


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

    print("Extracting clean train embeddings", flush=True)
    clean_train_embs, train_labels = extract_embeddings(
        encoder, train_loader, device, max_batches=args.max_train_batches
    )
    aug_fns = _train_augmentations()
    aug_embs_by_name = {}
    requested_ablations = list(ABLATIONS) if args.ablations == "all" else [x.strip() for x in args.ablations.split(",") if x.strip()]
    unknown = [x for x in requested_ablations if x not in ABLATIONS]
    if unknown:
        raise ValueError(f"Unknown ablations: {unknown}. Valid: {sorted(ABLATIONS)}")
    needed_augs = sorted({a for name in requested_ablations for a in ABLATIONS[name]["augmentations"]})
    for name in needed_augs:
        print(f"Extracting train embeddings: {name}", flush=True)
        aug_embs_by_name[name], labels = extract_embeddings(
            encoder, train_loader, device, transform_01=aug_fns[name], max_batches=args.max_train_batches
        )
        if not np.array_equal(train_labels, labels):
            raise RuntimeError(f"Label order mismatch for augmentation {name}")

    print("Extracting clean test embeddings", flush=True)
    clean_test_embs, test_labels = extract_embeddings(
        encoder, test_loader, device, max_batches=args.max_eval_batches
    )
    condition_embs = {"clean": clean_test_embs}
    for condition_name, condition_fn in get_eval_conditions():
        if condition_name == "clean":
            continue
        print(f"Extracting test embeddings: {condition_name}", flush=True)
        condition_embs[condition_name], labels = extract_embeddings(
            encoder, test_loader, device, transform_01=condition_fn, max_batches=args.max_eval_batches
        )
        if not np.array_equal(test_labels, labels):
            raise RuntimeError(f"Label order mismatch for condition {condition_name}")

    ablations = requested_ablations

    result_rows, per_class_rows, train_logs = [], [], []
    for ablation in ablations:
        model, logs = train_adapter(clean_train_embs, train_labels, aug_embs_by_name, ablation, args, device)
        train_logs.extend(logs)
        rows, pc_rows = evaluate_ablation(
            model, ablation, test_labels, clean_test_embs, condition_embs, device, args.adapter_batch_size
        )
        result_rows.extend(rows)
        per_class_rows.extend(pc_rows)
        ckpt = {
            "model_state": model.state_dict(),
            "feat_dim": clean_train_embs.shape[1],
            "num_classes": train_labels.shape[1],
            "ablation": ablation,
            "args": vars(args),
        }
        torch.save(ckpt, out / f"adapter_probe_{ablation}.pth")
        if ablation == "full":
            torch.save(ckpt, out / "adapter_probe.pth")

    pd.DataFrame(result_rows).to_csv(out / "nca_results.csv", index=False)
    pd.DataFrame([r for r in result_rows if r["ablation"] != "full"]).to_csv(out / "nca_ablation.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "nca_per_class.csv", index=False)
    pd.DataFrame(train_logs).to_csv(out / "nca_training_log.csv", index=False)
    print(out / "nca_results.csv", flush=True)


if __name__ == "__main__":
    main()
