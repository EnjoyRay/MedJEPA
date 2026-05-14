"""
Exp2 evaluation: lesion sensitivity via prediction drop + feature drift.

For each image with annotated lesion boxes:
  1. Extract embedding and prediction for the original image
  2. Extract embedding and prediction for the lesion-occluded image
  3. Extract embedding and prediction for the control-occluded image
  4. Compare:
     - Prediction drop: original_score - occluded_score
     - Feature drift: cosine_distance(original_emb, occluded_emb)

Usage (from run_exp2.py):
    results = evaluate_sensitivity(encoder, probe, loader, device)
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.metrics import compute_cosine_drift, compute_l2_drift, compute_auroc, compute_f1
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from occlusion import occlude_batch


def _box_summary(boxes, img_size):
    """Compact bbox metadata for offline subgroup analysis."""
    areas = []
    class_names = []
    for box in boxes:
        x1 = max(0.0, float(box['x_min']))
        y1 = max(0.0, float(box['y_min']))
        x2 = min(float(img_size), float(box['x_max']))
        y2 = min(float(img_size), float(box['y_max']))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > 0:
            areas.append(area)
        class_names.append(str(box.get('class_name', 'unknown')))

    image_area = float(img_size * img_size)
    return {
        'box_count': len(boxes),
        'bbox_area_frac': float(np.clip(np.sum(areas) / image_area, 0.0, 1.0)),
        'max_bbox_area_frac': float(np.max(areas) / image_area) if areas else 0.0,
        'lesion_classes': '|'.join(sorted(set(class_names))),
    }


@torch.no_grad()
def evaluate_sensitivity(
    encoder: torch.nn.Module,
    probe,              # LinearProbe
    loader: DataLoader,
    device: torch.device,
    control_strategy: str = 'opposite',
) -> Dict[str, Any]:
    """
    Run Exp2: lesion vs. control occlusion sensitivity.

    Returns dict:
    {
        'n_samples': int,
        'n_with_boxes': int,

        # Prediction scores (raw probabilities)
        'original_scores': np.ndarray (N_with_boxes, C)
        'lesion_scores': np.ndarray
        'control_scores': np.ndarray

        # Per-sample metrics (averaged over samples with boxes)
        'lesion_pred_drop': float          # mean drop in max-class confidence
        'control_pred_drop': float
        'lesion_cosine_drift': float
        'control_cosine_drift': float
        'lesion_l2_drift': float
        'control_l2_drift': float

        # Per-class prediction drop
        'lesion_pred_drop_per_class': np.ndarray (C,)
        'control_pred_drop_per_class': np.ndarray (C,)
    }
    """
    all_orig_embs, all_les_embs, all_ctrl_embs = [], [], []
    all_orig_scores, all_les_scores, all_ctrl_scores = [], [], []
    all_labels = []
    per_sample_meta = []
    n_total, n_with_boxes = 0, 0

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)   # (B, C, H, W) normalized
        labels = batch['label']
        batch_boxes = batch['boxes']   # list of B lists of dicts
        batch_image_ids = batch.get('image_id', [None] * images.shape[0])

        B = images.shape[0]
        n_total += B

        # Skip images without any bounding boxes
        has_box = [len(b) > 0 for b in batch_boxes]
        if not any(has_box):
            continue

        # Filter to images with boxes only
        idx_with_box = [i for i, h in enumerate(has_box) if h]
        images_wb = images[idx_with_box]
        boxes_wb = [batch_boxes[i] for i in idx_with_box]
        labels_wb = labels[idx_with_box]
        image_ids_wb = [batch_image_ids[i] for i in idx_with_box]
        n_with_boxes += len(idx_with_box)

        # Denormalize to [0,1] for occlusion
        images_01 = (images_wb * 0.5 + 0.5).clamp(0., 1.)

        # Apply occlusions
        lesion_01 = occlude_batch(images_01, boxes_wb, mode='lesion')
        control_01 = occlude_batch(images_01, boxes_wb, mode='control', strategy=control_strategy)

        # Re-normalize for encoder input
        lesion_norm = (lesion_01 - 0.5) / 0.5
        control_norm = (control_01 - 0.5) / 0.5

        # Extract embeddings
        orig_embs = encoder(images_wb)
        les_embs = encoder(lesion_norm)
        ctrl_embs = encoder(control_norm)

        all_orig_embs.append(orig_embs.cpu())
        all_les_embs.append(les_embs.cpu())
        all_ctrl_embs.append(ctrl_embs.cpu())
        all_labels.append(labels_wb)
        img_size = int(images_wb.shape[-1])
        for image_id, boxes in zip(image_ids_wb, boxes_wb):
            meta = _box_summary(boxes, img_size)
            meta['image_id'] = image_id
            per_sample_meta.append(meta)

    if n_with_boxes == 0:
        print("WARNING: No images with bounding boxes found in this split!")
        return {'n_samples': n_total, 'n_with_boxes': 0}

    # Concatenate
    orig_embs_np = torch.cat(all_orig_embs).float().numpy()
    les_embs_np = torch.cat(all_les_embs).float().numpy()
    ctrl_embs_np = torch.cat(all_ctrl_embs).float().numpy()
    labels_np = torch.cat(all_labels).float().numpy()

    # Predict probabilities
    orig_scores = probe.predict_proba(orig_embs_np)   # (N, C)
    les_scores = probe.predict_proba(les_embs_np)
    ctrl_scores = probe.predict_proba(ctrl_embs_np)

    # Per-class prediction drop
    les_drop_per_class = orig_scores - les_scores       # (N, C)
    ctrl_drop_per_class = orig_scores - ctrl_scores

    # Per-sample max-class prediction drop
    les_drop_per_sample = np.abs(orig_scores - les_scores).max(axis=1)
    ctrl_drop_per_sample = np.abs(orig_scores - ctrl_scores).max(axis=1)
    les_signed_drop_per_sample = (orig_scores - les_scores).mean(axis=1)
    ctrl_signed_drop_per_sample = (orig_scores - ctrl_scores).mean(axis=1)

    # Embedding drift
    orig_t = torch.cat(all_orig_embs).float()
    les_t = torch.cat(all_les_embs).float()
    ctrl_t = torch.cat(all_ctrl_embs).float()

    orig_norm = torch.nn.functional.normalize(orig_t, dim=-1)
    les_norm = torch.nn.functional.normalize(les_t, dim=-1)
    ctrl_norm = torch.nn.functional.normalize(ctrl_t, dim=-1)
    les_cos_per_sample = (1.0 - (orig_norm * les_norm).sum(dim=-1)).numpy()
    ctrl_cos_per_sample = (1.0 - (orig_norm * ctrl_norm).sum(dim=-1)).numpy()
    les_l2_per_sample = (orig_t - les_t).norm(dim=-1).numpy()
    ctrl_l2_per_sample = (orig_t - ctrl_t).norm(dim=-1).numpy()

    les_cos_drift = float(les_cos_per_sample.mean())
    ctrl_cos_drift = float(ctrl_cos_per_sample.mean())
    les_l2_drift = float(les_l2_per_sample.mean())
    ctrl_l2_drift = float(ctrl_l2_per_sample.mean())

    # AUROC for occluded variants (how much do they degrade?)
    orig_auroc = compute_auroc(labels_np, orig_scores)
    les_auroc = compute_auroc(labels_np, les_scores)
    ctrl_auroc = compute_auroc(labels_np, ctrl_scores)

    print(f"\n--- Exp2 Summary ({n_with_boxes} images with annotations) ---")
    print(f"  Original AUROC:         {orig_auroc['macro_auroc']:.4f}")
    print(f"  Lesion-occ AUROC:       {les_auroc['macro_auroc']:.4f}  "
          f"(drop: {orig_auroc['macro_auroc']-les_auroc['macro_auroc']:.4f})")
    print(f"  Control-occ AUROC:      {ctrl_auroc['macro_auroc']:.4f}  "
          f"(drop: {orig_auroc['macro_auroc']-ctrl_auroc['macro_auroc']:.4f})")
    print(f"  Lesion cosine drift:    {les_cos_drift:.4f}")
    print(f"  Control cosine drift:   {ctrl_cos_drift:.4f}")
    print(f"  Lesion pred drop (max): {les_drop_per_sample.mean():.4f}")
    print(f"  Control pred drop(max): {ctrl_drop_per_sample.mean():.4f}")

    return {
        'n_samples': n_total,
        'n_with_boxes': n_with_boxes,
        'original_auroc': orig_auroc,
        'lesion_auroc': les_auroc,
        'control_auroc': ctrl_auroc,
        'lesion_cosine_drift': les_cos_drift,
        'control_cosine_drift': ctrl_cos_drift,
        'lesion_l2_drift': les_l2_drift,
        'control_l2_drift': ctrl_l2_drift,
        'lesion_pred_drop': float(les_drop_per_sample.mean()),
        'control_pred_drop': float(ctrl_drop_per_sample.mean()),
        'lesion_pred_drop_per_class': les_drop_per_class.mean(axis=0),
        'control_pred_drop_per_class': ctrl_drop_per_class.mean(axis=0),
        'lesion_pred_drop_per_sample': les_drop_per_sample,
        'control_pred_drop_per_sample': ctrl_drop_per_sample,
        'lesion_signed_drop_per_sample': les_signed_drop_per_sample,
        'control_signed_drop_per_sample': ctrl_signed_drop_per_sample,
        'lesion_cosine_drift_per_sample': les_cos_per_sample,
        'control_cosine_drift_per_sample': ctrl_cos_per_sample,
        'lesion_l2_drift_per_sample': les_l2_per_sample,
        'control_l2_drift_per_sample': ctrl_l2_per_sample,
        'per_sample_meta': per_sample_meta,
        # Raw scores for offline analysis
        'original_scores': orig_scores,
        'lesion_scores': les_scores,
        'control_scores': ctrl_scores,
        'labels': labels_np,
    }


def save_results_csv(results: Dict[str, Any], output_dir: str) -> None:
    import csv
    os.makedirs(output_dir, exist_ok=True)

    if results.get('n_with_boxes', 0) == 0:
        print("No results to save.")
        return

    from shared.vindr_dataset import VINDR_CLASSES

    # Prediction drop CSV
    pd_path = os.path.join(output_dir, 'prediction_drop.csv')
    with open(pd_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'macro_auroc', 'auroc_drop', 'mean_pred_drop_max_class'])
        orig_auc = results['original_auroc']['macro_auroc']
        writer.writerow(['original', orig_auc, 0.0, 0.0])
        writer.writerow(['lesion_occ', results['lesion_auroc']['macro_auroc'],
                         orig_auc - results['lesion_auroc']['macro_auroc'],
                         results['lesion_pred_drop']])
        writer.writerow(['control_occ', results['control_auroc']['macro_auroc'],
                         orig_auc - results['control_auroc']['macro_auroc'],
                         results['control_pred_drop']])
    print(f"Saved prediction drop → {pd_path}")

    # Per-class prediction drop
    pc_path = os.path.join(output_dir, 'prediction_drop_per_class.csv')
    with open(pc_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'lesion_pred_drop', 'control_pred_drop'])
        for i, cls_name in enumerate(VINDR_CLASSES):
            writer.writerow([
                cls_name,
                results['lesion_pred_drop_per_class'][i],
                results['control_pred_drop_per_class'][i],
            ])
    print(f"Saved per-class drop → {pc_path}")

    # Feature drift CSV
    fd_path = os.path.join(output_dir, 'feature_drift.csv')
    with open(fd_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'cosine_drift', 'l2_drift'])
        writer.writerow(['lesion_occ', results['lesion_cosine_drift'], results['lesion_l2_drift']])
        writer.writerow(['control_occ', results['control_cosine_drift'], results['control_l2_drift']])
    print(f"Saved feature drift → {fd_path}")

    # Save raw scores as .npz for offline analysis
    np.savez(
        os.path.join(output_dir, 'scores.npz'),
        original_scores=results['original_scores'],
        lesion_scores=results['lesion_scores'],
        control_scores=results['control_scores'],
        labels=results['labels'],
        lesion_pred_drop_per_sample=results['lesion_pred_drop_per_sample'],
        control_pred_drop_per_sample=results['control_pred_drop_per_sample'],
        lesion_cosine_drift_per_sample=results['lesion_cosine_drift_per_sample'],
        control_cosine_drift_per_sample=results['control_cosine_drift_per_sample'],
    )
    print(f"Saved raw scores → {output_dir}/scores.npz")

    # Per-sample results for statistical tests, subgroup analysis, and case figures.
    ps_path = os.path.join(output_dir, 'per_sample_results.csv')
    with open(ps_path, 'w', newline='') as f:
        fieldnames = [
            'sample_index', 'image_id', 'lesion_classes', 'box_count',
            'bbox_area_frac', 'max_bbox_area_frac',
            'lesion_pred_drop_absmax', 'control_pred_drop_absmax',
            'delta_pred_drop_absmax',
            'lesion_pred_drop_mean_signed', 'control_pred_drop_mean_signed',
            'delta_pred_drop_mean_signed',
            'lesion_cosine_drift', 'control_cosine_drift', 'delta_cosine_drift',
            'lesion_l2_drift', 'control_l2_drift', 'delta_l2_drift',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, meta in enumerate(results['per_sample_meta']):
            writer.writerow({
                'sample_index': i,
                'image_id': meta.get('image_id'),
                'lesion_classes': meta.get('lesion_classes', ''),
                'box_count': meta.get('box_count', 0),
                'bbox_area_frac': meta.get('bbox_area_frac', 0.0),
                'max_bbox_area_frac': meta.get('max_bbox_area_frac', 0.0),
                'lesion_pred_drop_absmax': results['lesion_pred_drop_per_sample'][i],
                'control_pred_drop_absmax': results['control_pred_drop_per_sample'][i],
                'delta_pred_drop_absmax': (
                    results['lesion_pred_drop_per_sample'][i]
                    - results['control_pred_drop_per_sample'][i]
                ),
                'lesion_pred_drop_mean_signed': results['lesion_signed_drop_per_sample'][i],
                'control_pred_drop_mean_signed': results['control_signed_drop_per_sample'][i],
                'delta_pred_drop_mean_signed': (
                    results['lesion_signed_drop_per_sample'][i]
                    - results['control_signed_drop_per_sample'][i]
                ),
                'lesion_cosine_drift': results['lesion_cosine_drift_per_sample'][i],
                'control_cosine_drift': results['control_cosine_drift_per_sample'][i],
                'delta_cosine_drift': (
                    results['lesion_cosine_drift_per_sample'][i]
                    - results['control_cosine_drift_per_sample'][i]
                ),
                'lesion_l2_drift': results['lesion_l2_drift_per_sample'][i],
                'control_l2_drift': results['control_l2_drift_per_sample'][i],
                'delta_l2_drift': (
                    results['lesion_l2_drift_per_sample'][i]
                    - results['control_l2_drift_per_sample'][i]
                ),
            })
    print(f"Saved per-sample results to {ps_path}")


def plot_results(results: Dict[str, Any], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    if results.get('n_with_boxes', 0) == 0:
        return

    from shared.vindr_dataset import VINDR_CLASSES

    os.makedirs(output_dir, exist_ok=True)

    # Bar plot: cosine drift + prediction drop (summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    conditions = ['Lesion Occ', 'Control Occ']
    drifts = [results['lesion_cosine_drift'], results['control_cosine_drift']]
    bars = ax.bar(conditions, drifts, color=['steelblue', 'coral'])
    ax.set_title('Embedding Drift (Cosine Distance)')
    ax.set_ylabel('Cosine Distance')
    for bar, val in zip(bars, drifts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=9)

    ax = axes[1]
    drops = [results['lesion_pred_drop'], results['control_pred_drop']]
    bars = ax.bar(conditions, drops, color=['steelblue', 'coral'])
    ax.set_title('Prediction Drop (Mean Max-Class)')
    ax.set_ylabel('Score Drop')
    for bar, val in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', fontsize=9)

    plt.suptitle('Exp2: Lesion vs. Control Sensitivity')
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'sensitivity_bar.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved summary plot → {fig_path}")

    # Per-class bar plot
    les_drop = results['lesion_pred_drop_per_class']
    ctrl_drop = results['control_pred_drop_per_class']
    x = np.arange(len(VINDR_CLASSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width / 2, les_drop, width, label='Lesion Occ', color='steelblue')
    ax.bar(x + width / 2, ctrl_drop, width, label='Control Occ', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(VINDR_CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Prediction Drop')
    ax.set_title('Per-Class Prediction Drop: Lesion vs. Control')
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'per_class_drop.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved per-class plot → {fig_path}")
