"""
Shared metric utilities for both experiments.

Functions:
    compute_auroc(labels, scores)  → per-class and macro AUROC
    compute_cosine_drift(emb_a, emb_b) → mean cosine distance (1 - cosine_sim)
    compute_l2_drift(emb_a, emb_b) → mean L2 distance
"""

import numpy as np
import torch
from typing import Dict, Tuple, List


def compute_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute per-class and macro AUROC for multi-label classification.

    Args:
        labels: (N, C) binary ground-truth array
        scores: (N, C) predicted probability/score array
    Returns:
        dict with keys: class AUROC values + 'macro_auroc'
    """
    from sklearn.metrics import roc_auc_score
    results = {}
    n_classes = labels.shape[1]
    valid_aucs = []

    for c in range(n_classes):
        # Skip classes with only one label value in ground truth
        if labels[:, c].sum() == 0 or labels[:, c].sum() == len(labels):
            results[f'class_{c}'] = float('nan')
            continue
        auc = roc_auc_score(labels[:, c], scores[:, c])
        results[f'class_{c}'] = float(auc)
        valid_aucs.append(auc)

    results['macro_auroc'] = float(np.mean(valid_aucs)) if valid_aucs else float('nan')
    return results


def compute_f1(
    labels: np.ndarray,
    preds: np.ndarray,
    average: str = 'macro',
) -> float:
    """Compute F1 score (binary predictions)."""
    from sklearn.metrics import f1_score
    return float(f1_score(labels, preds, average=average, zero_division=0))


def compute_cosine_drift(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
) -> float:
    """
    Mean cosine distance (1 - cosine similarity) between paired embeddings.

    Args:
        emb_a: (N, D) original embeddings
        emb_b: (N, D) perturbed / occluded embeddings
    Returns:
        scalar: mean cosine distance in [0, 2]
    """
    emb_a = torch.nn.functional.normalize(emb_a, dim=-1)
    emb_b = torch.nn.functional.normalize(emb_b, dim=-1)
    cosine_sim = (emb_a * emb_b).sum(dim=-1)   # (N,)
    cosine_dist = 1.0 - cosine_sim              # (N,)
    return float(cosine_dist.mean().item())


def compute_l2_drift(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
) -> float:
    """Mean L2 distance between paired embeddings."""
    diff = emb_a - emb_b
    l2 = diff.norm(dim=-1)   # (N,)
    return float(l2.mean().item())


def embeddings_to_numpy(
    embs: List[torch.Tensor],
    labels: List[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate lists of embedding/label tensors into numpy arrays."""
    return (
        torch.cat(embs, dim=0).cpu().float().numpy(),
        torch.cat(labels, dim=0).cpu().float().numpy(),
    )
