"""
Create qualitative case-study figures for Exp2.

Requires an Exp2 output directory produced by the updated run_exp2.py, because
it uses per_sample_results.csv to align images with scores.npz.

Example:
    python experiments/exp2_lesion_sensitivity/make_case_studies.py \
        --exp2_dir results/exp2_ijepa_vith_epoch97 \
        --data_dir /path/to/VinBigData_ChestXray \
        --output_dir results/exp2_ijepa_vith_epoch97/cases \
        --num_cases 8
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.vindr_dataset import VinDrCXRDataset, VINDR_CLASSES
from occlusion import occlude_batch


def get_args():
    parser = argparse.ArgumentParser(description='Create Exp2 qualitative case studies')
    parser.add_argument('--exp2_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--in_chans', type=int, default=1)
    parser.add_argument('--num_cases', type=int, default=8)
    parser.add_argument('--control_strategy', default='opposite', choices=['opposite', 'random'])
    parser.add_argument(
        '--selection',
        default='mixed',
        choices=['mixed', 'largest_lesion_effect', 'largest_control_effect', 'near_equal'],
    )
    return parser.parse_args()


def _image_to_numpy(tensor):
    """Convert normalized dataset tensor to HxW grayscale numpy in [0, 1]."""
    img = (tensor * 0.5 + 0.5).clamp(0, 1)
    if img.shape[0] > 1:
        img = img.mean(dim=0, keepdim=True)
    return img.squeeze(0).cpu().numpy()


def _draw_boxes(ax, boxes, color='lime'):
    import matplotlib.patches as patches
    for box in boxes:
        x1, y1 = float(box['x_min']), float(box['y_min'])
        x2, y2 = float(box['x_max']), float(box['y_max'])
        rect = patches.Rectangle(
            (x1, y1), max(0, x2 - x1), max(0, y2 - y1),
            linewidth=1.4, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)


def _select_cases(df, num_cases, mode):
    df = df.copy()
    if mode == 'largest_lesion_effect':
        return df.sort_values('delta_cosine_drift', ascending=False).head(num_cases)
    if mode == 'largest_control_effect':
        return df.sort_values('delta_cosine_drift', ascending=True).head(num_cases)
    if mode == 'near_equal':
        return df.iloc[(df['delta_cosine_drift'].abs()).argsort()].head(num_cases)

    parts = []
    k = max(1, num_cases // 3)
    parts.append(df.sort_values('delta_cosine_drift', ascending=False).head(k))
    parts.append(df.sort_values('delta_cosine_drift', ascending=True).head(k))
    parts.append(df.iloc[(df['delta_cosine_drift'].abs()).argsort()].head(num_cases - 2 * k))
    selected = pd.concat(parts, axis=0).drop_duplicates('sample_index')
    if len(selected) < num_cases:
        extra = df.drop(selected.index, errors='ignore').head(num_cases - len(selected))
        selected = pd.concat([selected, extra], axis=0)
    return selected.head(num_cases)


def _plot_case(sample, dataset, scores, output_path, control_strategy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    image_id = str(sample['image_id'])
    sample_index = int(sample['sample_index'])
    data_index = dataset.image_ids.index(image_id)
    item = dataset[data_index]

    image = item['image']
    boxes = item['boxes']
    image_01 = (image.unsqueeze(0) * 0.5 + 0.5).clamp(0, 1)
    lesion = occlude_batch(image_01, [boxes], mode='lesion')[0]
    control = occlude_batch(
        image_01, [boxes], mode='control', strategy=control_strategy
    )[0]

    orig_scores = scores['original_scores'][sample_index]
    lesion_scores = scores['lesion_scores'][sample_index]
    control_scores = scores['control_scores'][sample_index]
    labels = scores['labels'][sample_index]

    top_idx = np.argsort(orig_scores)[-5:][::-1]
    y = np.arange(len(top_idx))
    class_names = [VINDR_CLASSES[i] for i in top_idx]

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(_image_to_numpy(image), cmap='gray', vmin=0, vmax=1)
    _draw_boxes(ax, boxes, color='lime')
    ax.set_title('Original + lesion bbox')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(_image_to_numpy(lesion), cmap='gray', vmin=0, vmax=1)
    ax.set_title('Lesion occluded')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(_image_to_numpy(control), cmap='gray', vmin=0, vmax=1)
    ax.set_title('Control occluded')
    ax.axis('off')

    ax = fig.add_subplot(gs[1, :])
    width = 0.25
    ax.barh(y - width, orig_scores[top_idx], height=width, label='Original')
    ax.barh(y, lesion_scores[top_idx], height=width, label='Lesion occ')
    ax.barh(y + width, control_scores[top_idx], height=width, label='Control occ')
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{name}{' *' if labels[idx] > 0.5 else ''}"
        for name, idx in zip(class_names, top_idx)
    ])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probe probability')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.2)

    title = (
        f"{image_id} | classes={sample['lesion_classes']} | "
        f"bbox={float(sample['bbox_area_frac']):.3f} | "
        f"delta drift={float(sample['delta_cosine_drift']):+.4f}"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = get_args()
    exp2_dir = Path(args.exp2_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp2_dir / 'case_studies'
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = exp2_dir / 'per_sample_results.csv'
    scores_path = exp2_dir / 'scores.npz'
    if not per_sample_path.exists():
        raise FileNotFoundError(
            f"{per_sample_path} not found. Re-run Exp2 with the updated code first."
        )
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found.")

    df = pd.read_csv(per_sample_path)
    scores = np.load(scores_path)
    dataset = VinDrCXRDataset(
        args.data_dir, split='test', img_size=args.img_size, in_chans=args.in_chans
    )

    selected = _select_cases(df, args.num_cases, args.selection)
    selected.to_csv(output_dir / 'selected_cases.csv', index=False)

    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        output_path = output_dir / f'case_{rank:02d}_{row["image_id"]}.png'
        _plot_case(row, dataset, scores, output_path, args.control_strategy)

    print(f"Wrote {len(selected)} case-study figures to {output_dir}")


if __name__ == '__main__':
    main()
