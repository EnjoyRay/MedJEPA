"""
Class-aligned Exp2 evaluation.

The legacy Exp2 treats each image as one unit and occludes every annotation box
at once.  This file uses the cleaner unit needed for the thesis: one
(image_id, class_name) pair where the class is positive and has matching boxes.
Only boxes for that class are occluded, and the target class logit/probability
is compared with multiple matched control occlusions.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from shared.vindr_dataset import VINDR_CLASSES, CLASS_TO_IDX
from occlusion import (
    box_area_fraction,
    boxes_for_class,
    generate_k_control_box_sets,
    merge_overlapping_boxes,
    occlude_lesion,
)


LOCALIZED_FINDINGS = {
    'Atelectasis',
    'Calcification',
    'Consolidation',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pneumothorax',
}

DIFFUSE_FINDINGS = {
    'Aortic enlargement',
    'Cardiomegaly',
    'ILD',
    'Pleural effusion',
    'Pleural thickening',
    'Pulmonary fibrosis',
}


def disease_group(class_name: str) -> str:
    if class_name in LOCALIZED_FINDINGS:
        return 'localized'
    if class_name in DIFFUSE_FINDINGS:
        return 'diffuse_or_global'
    if class_name == 'No finding':
        return 'no_finding'
    return 'other'


def _fill_value(image_01: torch.Tensor, fill_mode: str) -> float:
    if fill_mode in ('neutral', 'constant'):
        return 0.5
    if fill_mode == 'image_mean':
        return float(image_01.mean().item())
    raise ValueError(f"Unknown fill_mode={fill_mode}")


def _normalize(image_01: torch.Tensor) -> torch.Tensor:
    return (image_01 - 0.5) / 0.5


def _probe_logits(probe, embeddings: torch.Tensor) -> torch.Tensor:
    """Run the existing LinearProbe head on tensor embeddings."""
    head = probe.head
    was_training = head.training
    head.eval()
    logits = head(embeddings.to(next(head.parameters()).device))
    if was_training:
        head.train()
    return logits


def _bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _sign_flip_pvalue(values: np.ndarray, n_perm: int = 20000, seed: int = 42) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, values.size))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def _skipped_summary(skipped: List[Dict[str, Any]]) -> 'pd.DataFrame':
    """Build a DataFrame summarising records skipped due to no valid control."""
    import pandas as pd
    return pd.DataFrame(skipped)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a = torch.nn.functional.normalize(a.float(), dim=-1)
    b = torch.nn.functional.normalize(b.float(), dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).cpu().numpy()


def _encode_in_chunks(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Encode a tensor batch in chunks to keep Exp2b memory bounded."""
    outputs = []
    for start in range(0, images.shape[0], chunk_size):
        outputs.append(encoder(images[start:start + chunk_size]))
    return torch.cat(outputs, dim=0)


def _eligible_records(batch: Dict[str, Any], img_size: int) -> List[Dict[str, Any]]:
    records = []
    labels = batch['label']
    for image_pos, boxes in enumerate(batch['boxes']):
        if not boxes:
            continue
        for class_name in VINDR_CLASSES:
            if class_name == 'No finding':
                continue
            class_idx = CLASS_TO_IDX[class_name]
            if float(labels[image_pos, class_idx]) < 0.5:
                continue
            target_boxes = boxes_for_class(boxes, class_name)
            if not target_boxes:
                continue
            raw_box_count = len(target_boxes)
            target_boxes = merge_overlapping_boxes(target_boxes, img_size)
            if not target_boxes:
                continue
            total_area, max_area = box_area_fraction(target_boxes, img_size)
            if total_area <= 0.0:
                continue
            records.append(
                {
                    'image_pos': image_pos,
                    'image_id': batch['image_id'][image_pos],
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'target_boxes': target_boxes,
                    'all_boxes': boxes,
                    'box_count': len(target_boxes),
                    'raw_box_count': raw_box_count,
                    'bbox_area_frac': total_area,
                    'max_bbox_area_frac': max_area,
                    'disease_group': disease_group(class_name),
                }
            )
    return records


@torch.no_grad()
def evaluate_class_aligned_sensitivity(
    encoder: torch.nn.Module,
    probe,
    loader: DataLoader,
    device: torch.device,
    num_controls: int = 5,
    control_strategy: str = 'matched_random',
    fill_mode: str = 'neutral',
    seed: int = 42,
    max_samples: int | None = None,
    encoder_batch_size: int = 16,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []
    n_images = 0
    n_images_with_boxes = 0
    n_fallback_controls = 0
    n_skipped_no_valid_control = 0

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        images_01 = (images * 0.5 + 0.5).clamp(0.0, 1.0)
        img_size = int(images.shape[-1])
        n_images += int(images.shape[0])
        n_images_with_boxes += sum(1 for b in batch['boxes'] if len(b) > 0)

        records = _eligible_records(batch, img_size)
        if max_samples is not None and len(rows) >= max_samples:
            break
        if max_samples is not None:
            records = records[: max(0, max_samples - len(rows))]
        if not records:
            continue

        original_tensors = []
        lesion_tensors = []
        control_tensors = []
        control_boxes_all = []
        valid_records = []

        for rec_idx, rec in enumerate(records):
            image_01 = images_01[rec['image_pos']]
            fill = _fill_value(image_01, fill_mode)
            lesion_01 = occlude_lesion(image_01, rec['target_boxes'], fill=fill)
            controls = generate_k_control_box_sets(
                target_boxes=rec['target_boxes'],
                forbidden_boxes=rec['all_boxes'],
                img_size=img_size,
                k=num_controls,
                seed=seed + batch_idx * 100000 + rec_idx * 997,
                strategy=control_strategy,
            )
            if len(controls) < num_controls:
                n_skipped_no_valid_control += 1
                skipped_records.append({
                    'image_id': rec['image_id'],
                    'class_name': rec['class_name'],
                    'box_count': rec['box_count'],
                    'bbox_area_frac': rec['bbox_area_frac'],
                    'total_boxes_in_image': len(rec['all_boxes']),
                    'disease_group': rec['disease_group'],
                })
                continue

            original_tensors.append(images[rec['image_pos']])
            lesion_tensors.append(_normalize(lesion_01))
            control_boxes_all.append(controls)
            valid_records.append(rec)
            for box_set in controls:
                if any(bool(box.get('fallback_overlap', False)) for box in box_set):
                    n_fallback_controls += 1
                control_01 = occlude_lesion(image_01, box_set, fill=fill)
                control_tensors.append(_normalize(control_01))

        records = valid_records
        if not records:
            continue

        orig_batch = torch.stack(original_tensors).to(device)
        lesion_batch = torch.stack(lesion_tensors).to(device)
        control_batch = torch.stack(control_tensors).to(device)

        orig_emb = _encode_in_chunks(encoder, orig_batch, encoder_batch_size)
        lesion_emb = _encode_in_chunks(encoder, lesion_batch, encoder_batch_size)
        control_emb = _encode_in_chunks(encoder, control_batch, encoder_batch_size)

        orig_logits = _probe_logits(probe, orig_emb).detach().cpu()
        lesion_logits = _probe_logits(probe, lesion_emb).detach().cpu()
        control_logits = _probe_logits(probe, control_emb).detach().cpu()

        orig_probs = torch.sigmoid(orig_logits).numpy()
        lesion_probs = torch.sigmoid(lesion_logits).numpy()
        control_probs = torch.sigmoid(control_logits).numpy()

        lesion_cos = _cosine_distance(orig_emb.cpu(), lesion_emb.cpu())
        control_cos = _cosine_distance(
            orig_emb.cpu().repeat_interleave(num_controls, dim=0),
            control_emb.cpu(),
        ).reshape(len(records), num_controls)

        control_logits_np = control_logits.numpy().reshape(len(records), num_controls, -1)
        control_probs_np = control_probs.reshape(len(records), num_controls, -1)

        for i, rec in enumerate(records):
            c = rec['class_idx']
            ctrl_logit_vals = control_logits_np[i, :, c]
            ctrl_prob_vals = control_probs_np[i, :, c]
            ctrl_cos_vals = control_cos[i]
            target_logit_drop = float(orig_logits[i, c] - lesion_logits[i, c])
            target_prob_drop = float(orig_probs[i, c] - lesion_probs[i, c])
            control_logit_drop = (float(orig_logits[i, c]) - ctrl_logit_vals).astype(float)
            control_prob_drop = (float(orig_probs[i, c]) - ctrl_prob_vals).astype(float)
            rows.append(
                {
                    'sample_index': len(rows),
                    'image_id': rec['image_id'],
                    'class_name': rec['class_name'],
                    'class_idx': c,
                    'disease_group': rec['disease_group'],
                    'box_count': rec['box_count'],
                    'raw_box_count': rec['raw_box_count'],
                    'bbox_area_frac': rec['bbox_area_frac'],
                    'max_bbox_area_frac': rec['max_bbox_area_frac'],
                    'num_controls': num_controls,
                    'fill_mode': fill_mode,
                    'control_strategy': control_strategy,
                    'original_logit': float(orig_logits[i, c]),
                    'lesion_logit': float(lesion_logits[i, c]),
                    'control_logit_mean': float(ctrl_logit_vals.mean()),
                    'control_logit_std': float(ctrl_logit_vals.std(ddof=0)),
                    'target_logit_drop': target_logit_drop,
                    'control_logit_drop_mean': float(control_logit_drop.mean()),
                    'control_logit_drop_std': float(control_logit_drop.std(ddof=0)),
                    'delta_logit_drop': target_logit_drop - float(control_logit_drop.mean()),
                    'original_prob': float(orig_probs[i, c]),
                    'lesion_prob': float(lesion_probs[i, c]),
                    'control_prob_mean': float(ctrl_prob_vals.mean()),
                    'control_prob_std': float(ctrl_prob_vals.std(ddof=0)),
                    'target_prob_drop': target_prob_drop,
                    'control_prob_drop_mean': float(control_prob_drop.mean()),
                    'control_prob_drop_std': float(control_prob_drop.std(ddof=0)),
                    'delta_prob_drop': target_prob_drop - float(control_prob_drop.mean()),
                    'lesion_cosine_drift': float(lesion_cos[i]),
                    'control_cosine_drift_mean': float(ctrl_cos_vals.mean()),
                    'control_cosine_drift_std': float(ctrl_cos_vals.std(ddof=0)),
                    'delta_cosine_drift': float(lesion_cos[i] - ctrl_cos_vals.mean()),
                    'control_logits_json': json.dumps(ctrl_logit_vals.astype(float).tolist()),
                    'control_probs_json': json.dumps(ctrl_prob_vals.astype(float).tolist()),
                    'control_cosine_json': json.dumps(ctrl_cos_vals.astype(float).tolist()),
                    'control_boxes_json': json.dumps(control_boxes_all[i]),
                }
            )

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  batch {batch_idx + 1}/{len(loader)}: "
                f"class-aligned rows={len(rows)}",
                flush=True,
            )

    skipped_summary = {}
    if skipped_records:
        skipped_df = _skipped_summary(skipped_records)
        skipped_summary = {
            'n_skipped': len(skipped_records),
            'skipped_mean_box_count': float(skipped_df['box_count'].mean()) if len(skipped_df) else 0.0,
            'skipped_mean_bbox_area_frac': float(skipped_df['bbox_area_frac'].mean()) if len(skipped_df) else 0.0,
            'skipped_mean_total_boxes': float(skipped_df['total_boxes_in_image'].mean()) if len(skipped_df) else 0.0,
            'skipped_disease_groups': skipped_df['disease_group'].value_counts().to_dict() if len(skipped_df) else {},
        }

    return {
        'rows': rows,
        'skipped_records': skipped_records,
        'n_images': n_images,
        'n_images_with_boxes': n_images_with_boxes,
        'n_class_aligned_samples': len(rows),
        'n_fallback_controls': n_fallback_controls,
        'n_skipped_no_valid_control': n_skipped_no_valid_control,
        'skipped_summary': skipped_summary,
        'num_controls': num_controls,
        'fill_mode': fill_mode,
        'control_strategy': control_strategy,
        'encoder_batch_size': encoder_batch_size,
    }


def _summary_rows(rows: List[Dict[str, Any]], group_key: str | None, seed: int = 42) -> List[Dict[str, Any]]:
    if group_key is None:
        groups = [('overall', rows)]
    else:
        values = sorted({str(row[group_key]) for row in rows})
        groups = [(value, [row for row in rows if str(row[group_key]) == value]) for value in values]

    out = []
    for name, sub in groups:
        if not sub:
            continue
        delta_logit = np.asarray([r['delta_logit_drop'] for r in sub], dtype=float)
        delta_prob = np.asarray([r['delta_prob_drop'] for r in sub], dtype=float)
        delta_cos = np.asarray([r['delta_cosine_drift'] for r in sub], dtype=float)
        ci_logit = _bootstrap_ci(delta_logit, seed=seed)
        ci_prob = _bootstrap_ci(delta_prob, seed=seed)
        ci_cos = _bootstrap_ci(delta_cos, seed=seed)
        out.append(
            {
                'group': name,
                'n': len(sub),
                'bbox_area_frac_mean': float(np.mean([r['bbox_area_frac'] for r in sub])),
                'target_logit_drop_mean': float(np.mean([r['target_logit_drop'] for r in sub])),
                'control_logit_drop_mean': float(np.mean([r['control_logit_drop_mean'] for r in sub])),
                'delta_logit_drop_mean': float(delta_logit.mean()),
                'delta_logit_ci95_low': ci_logit[0],
                'delta_logit_ci95_high': ci_logit[1],
                'delta_logit_pvalue': _sign_flip_pvalue(delta_logit, seed=seed),
                'target_prob_drop_mean': float(np.mean([r['target_prob_drop'] for r in sub])),
                'control_prob_drop_mean': float(np.mean([r['control_prob_drop_mean'] for r in sub])),
                'delta_prob_drop_mean': float(delta_prob.mean()),
                'delta_prob_ci95_low': ci_prob[0],
                'delta_prob_ci95_high': ci_prob[1],
                'delta_prob_pvalue': _sign_flip_pvalue(delta_prob, seed=seed),
                'lesion_cosine_drift_mean': float(np.mean([r['lesion_cosine_drift'] for r in sub])),
                'control_cosine_drift_mean': float(np.mean([r['control_cosine_drift_mean'] for r in sub])),
                'delta_cosine_drift_mean': float(delta_cos.mean()),
                'delta_cosine_ci95_low': ci_cos[0],
                'delta_cosine_ci95_high': ci_cos[1],
                'delta_cosine_pvalue': _sign_flip_pvalue(delta_cos, seed=seed),
            }
        )
    return out


def save_class_aligned_results(results: Dict[str, Any], output_dir: str, seed: int = 42) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rows = results['rows']
    if not rows:
        raise RuntimeError("No class-aligned rows were produced.")

    per_sample_path = os.path.join(output_dir, 'class_aligned_per_sample.csv')
    fieldnames = list(rows[0].keys())
    with open(per_sample_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def write_summary(filename: str, summary: List[Dict[str, Any]]) -> None:
        path = os.path.join(output_dir, filename)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    write_summary('class_aligned_overall.csv', _summary_rows(rows, None, seed=seed))
    write_summary('class_aligned_summary_by_class.csv', _summary_rows(rows, 'class_name', seed=seed))
    write_summary('class_aligned_summary_by_group.csv', _summary_rows(rows, 'disease_group', seed=seed))

    if results.get('skipped_summary'):
        with open(os.path.join(output_dir, 'class_aligned_skipped_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(results['skipped_summary'], f, indent=2)
        if 'skipped_records' in results:
            skipped_path = os.path.join(output_dir, 'class_aligned_skipped_per_sample.csv')
            skipped_df = _skipped_summary(results['skipped_records'])
            skipped_df.to_csv(skipped_path, index=False)
            print(f"Saved skipped records → {skipped_path} (n={len(skipped_df)})")

    meta = {k: v for k, v in results.items() if k != 'rows'}
    with open(os.path.join(output_dir, 'class_aligned_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved class-aligned rows to {per_sample_path}")
