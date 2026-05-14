"""
Exp1 evaluation: robustness curve + embedding drift.

Given a trained linear probe and a frozen encoder, this module:
  1. Extracts embeddings for clean images → baseline performance
  2. For each perturbation type × strength:
     a. Perturbs the images
     b. Extracts embeddings
     c. Measures downstream AUROC/F1 (functional robustness)
     d. Measures cosine drift between clean and perturbed embeddings

Usage (from run_exp1.py):
    results = evaluate_robustness(encoder, probe, loader, perturbation_schedule, device)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.metrics import (
    compute_auroc,
    compute_f1,
    compute_cosine_drift,
    compute_l2_drift,
    embeddings_to_numpy,
)
from shared.perturbations import get_perturbation_schedule


@torch.no_grad()
def _extract_embeddings(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    perturbation_fn=None,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings (and collect labels) from a DataLoader.

    Args:
        encoder: frozen encoder, forward(x) -> (B, D)
        loader: yields dicts with 'image' and 'label' keys
        device: torch device
        perturbation_fn: optional callable (B,C,H,W) -> (B,C,H,W)
    Returns:
        embeddings: (N, D) float32 numpy array
        labels: (N, C) float32 numpy array
    """
    all_embs, all_labels = [], []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label']

        # Denormalize from [-1,1] to [0,1] for perturbation, then re-normalize
        if perturbation_fn is not None:
            images_01 = (images * 0.5 + 0.5).clamp(0., 1.)
            images_01 = perturbation_fn(images_01)
            images = (images_01 - 0.5) / 0.5

        embs = encoder(images)   # (B, D)
        all_embs.append(embs.cpu())
        all_labels.append(labels)

    embs_np, labels_np = embeddings_to_numpy(all_embs, all_labels)
    return embs_np, labels_np


def evaluate_robustness(
    encoder: torch.nn.Module,
    probe,          # LinearProbe
    loader: DataLoader,
    device: torch.device,
    perturbation_types: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full Exp1 evaluation.

    Returns a dict:
    {
        'clean': {'auroc': ..., 'f1': ..., 'embeddings': np.ndarray},
        '<pert_type>/<label>': {
            'auroc': ..., 'f1': ...,
            'cosine_drift': ..., 'l2_drift': ...
        },
        ...
    }
    """
    schedule = get_perturbation_schedule(perturbation_types)
    results = {}

    # --- Clean baseline ---
    print("Extracting clean embeddings...")
    clean_embs, clean_labels = _extract_embeddings(encoder, loader, device, max_batches=max_batches)
    clean_probs = probe.predict_proba(clean_embs)
    clean_preds = (clean_probs > 0.5).astype(np.float32)

    clean_auroc = compute_auroc(clean_labels, clean_probs)
    clean_f1 = compute_f1(clean_labels, clean_preds)
    results['clean'] = {
        'auroc': clean_auroc,
        'f1': clean_f1,
        'embeddings': clean_embs,
    }
    print(f"  Clean → macro AUROC: {clean_auroc['macro_auroc']:.4f}  F1: {clean_f1:.4f}")

    # --- Perturbed evaluations ---
    clean_embs_t = torch.from_numpy(clean_embs)

    for pert_type, levels in schedule.items():
        for label, pert_fn in levels:
            key = f'{pert_type}/{label}'
            print(f"Evaluating perturbation: {key}")

            pert_embs, _ = _extract_embeddings(
                encoder, loader, device, perturbation_fn=pert_fn, max_batches=max_batches
            )
            pert_probs = probe.predict_proba(pert_embs)
            pert_preds = (pert_probs > 0.5).astype(np.float32)

            pert_auroc = compute_auroc(clean_labels, pert_probs)
            pert_f1 = compute_f1(clean_labels, pert_preds)

            pert_embs_t = torch.from_numpy(pert_embs)
            cosine_drift = compute_cosine_drift(clean_embs_t, pert_embs_t)
            l2_drift = compute_l2_drift(clean_embs_t, pert_embs_t)

            results[key] = {
                'auroc': pert_auroc,
                'f1': pert_f1,
                'cosine_drift': cosine_drift,
                'l2_drift': l2_drift,
                'auroc_drop': clean_auroc['macro_auroc'] - pert_auroc['macro_auroc'],
                'f1_drop': clean_f1 - pert_f1,
            }
            print(f"  {key} → AUROC: {pert_auroc['macro_auroc']:.4f}  "
                  f"F1: {pert_f1:.4f}  "
                  f"CosDrift: {cosine_drift:.4f}  "
                  f"L2Drift: {l2_drift:.4f}")

    return results


def save_results_csv(results: Dict[str, Any], output_dir: str) -> None:
    """Save robustness results to CSV files."""
    import csv, os
    os.makedirs(output_dir, exist_ok=True)

    # Robustness curve
    rc_path = os.path.join(output_dir, 'robustness_curve.csv')
    with open(rc_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'macro_auroc', 'f1', 'auroc_drop', 'f1_drop'])
        clean = results['clean']
        writer.writerow(['clean', clean['auroc']['macro_auroc'], clean['f1'], 0.0, 0.0])
        for key, v in results.items():
            if key == 'clean':
                continue
            writer.writerow([
                key,
                v['auroc']['macro_auroc'],
                v['f1'],
                v['auroc_drop'],
                v['f1_drop'],
            ])
    print(f"Saved robustness curve → {rc_path}")

    # Embedding drift
    ed_path = os.path.join(output_dir, 'embedding_drift.csv')
    with open(ed_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'cosine_drift', 'l2_drift'])
        for key, v in results.items():
            if key == 'clean':
                continue
            writer.writerow([key, v['cosine_drift'], v['l2_drift']])
    print(f"Saved embedding drift → {ed_path}")


def plot_results(results: Dict[str, Any], output_dir: str) -> None:
    """Generate robustness curve and drift plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by perturbation type
    pert_types: Dict[str, list] = {}
    for key, v in results.items():
        if key == 'clean':
            continue
        pert_type = key.split('/')[0]
        pert_types.setdefault(pert_type, []).append((key, v))

    for pert_type, items in pert_types.items():
        labels_x = ['clean'] + [k.split('/')[1] for k, _ in items]
        aurocs = [results['clean']['auroc']['macro_auroc']] + [v['auroc']['macro_auroc'] for _, v in items]
        drifts = [0.0] + [v['cosine_drift'] for _, v in items]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(labels_x, aurocs, marker='o')
        ax1.set_title(f'Robustness: {pert_type}')
        ax1.set_xlabel('Perturbation strength')
        ax1.set_ylabel('Macro AUROC')
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='x', rotation=30)

        ax2.plot(labels_x, drifts, marker='s', color='orange')
        ax2.set_title(f'Embedding Drift: {pert_type}')
        ax2.set_xlabel('Perturbation strength')
        ax2.set_ylabel('Cosine Distance')
        ax2.tick_params(axis='x', rotation=30)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'robustness_{pert_type}.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved plot → {fig_path}")
