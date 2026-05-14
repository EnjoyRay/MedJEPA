"""
Create qualitative case figures for class-aligned Exp2b.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from shared.vindr_dataset import VinDrCXRDataset
from occlusion import merge_overlapping_boxes, occlude_lesion


def get_args():
    parser = argparse.ArgumentParser(description='Create class-aligned Exp2 case figures')
    parser.add_argument('--exp2_dir', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--in_chans', default=1, type=int)
    parser.add_argument('--max_cases', default=9, type=int)
    return parser.parse_args()


def _img_np(tensor_01: torch.Tensor):
    arr = tensor_01.detach().cpu().numpy()
    if arr.shape[0] == 1:
        return arr[0]
    return arr.transpose(1, 2, 0)


def _parse_boxes(raw: str):
    try:
        boxes = json.loads(raw)
        return boxes[0] if boxes and isinstance(boxes[0], list) else boxes
    except Exception:
        return []


def _target_boxes(all_boxes, class_name):
    return [box for box in all_boxes if str(box.get('class_name')) == class_name]


def _draw_boxes(ax, boxes, color, label):
    import matplotlib.patches as patches

    first = True
    for box in boxes:
        x1, y1, x2, y2 = box['x_min'], box['y_min'], box['x_max'], box['y_max']
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.8,
            edgecolor=color,
            facecolor='none',
            label=label if first else None,
        )
        ax.add_patch(rect)
        first = False


def main():
    args = get_args()
    exp2_dir = Path(args.exp2_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp2_dir / 'class_aligned_cases'
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob('case_*.png'):
        old.unlink()
    old_manifest = output_dir / 'class_aligned_case_figures.csv'
    if old_manifest.exists():
        old_manifest.unlink()

    per_sample = pd.read_csv(exp2_dir / 'class_aligned_per_sample.csv')
    candidates_path = exp2_dir / 'class_aligned_analysis' / 'class_aligned_case_candidates.csv'
    if candidates_path.exists():
        cases = pd.read_csv(candidates_path).head(args.max_cases)
    else:
        frames = [
            per_sample.sort_values('delta_logit_drop', ascending=False).head(3),
            per_sample.sort_values('delta_logit_drop', ascending=True).head(3),
            per_sample.iloc[(per_sample['delta_logit_drop'].abs()).argsort()].head(3),
        ]
        cases = pd.concat(frames, ignore_index=True).head(args.max_cases)

    dataset = VinDrCXRDataset(
        args.data_dir,
        split='test',
        img_size=args.img_size,
        in_chans=args.in_chans,
    )
    index_by_id = {image_id: i for i, image_id in enumerate(dataset.image_ids)}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    saved = []
    for case_id, row in cases.iterrows():
        image_id = str(row['image_id'])
        if image_id not in index_by_id:
            continue
        sample = dataset[index_by_id[image_id]]
        image_01 = (sample['image'] * 0.5 + 0.5).clamp(0.0, 1.0)
        target = merge_overlapping_boxes(
            _target_boxes(sample['boxes'], row['class_name']),
            args.img_size,
        )
        control = _parse_boxes(row.get('control_boxes_json', '[]'))
        lesion = occlude_lesion(image_01, target, fill=0.5)
        control_img = occlude_lesion(image_01, control, fill=0.5) if control else image_01

        fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.8))
        for ax, img, title in [
            (axes[0], image_01, 'Original + target bbox'),
            (axes[1], lesion, 'Target-class lesion occlusion'),
            (axes[2], control_img, 'Matched control occlusion'),
            (axes[3], image_01, 'Score changes'),
        ]:
            if ax is not axes[3]:
                ax.imshow(_img_np(img), cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title(title, fontsize=9)
        _draw_boxes(axes[0], target, '#7C3AED', 'target')
        _draw_boxes(axes[2], control, '#F97316', 'control')

        axes[3].axis('off')
        text = (
            f"Image: {image_id}\n"
            f"Class: {row['class_name']} ({row['disease_group']})\n"
            f"Bbox area: {float(row['bbox_area_frac']):.3f}\n\n"
            f"Original prob: {float(row['original_prob']):.3f}\n"
            f"Lesion prob: {float(row['lesion_prob']):.3f}\n"
            f"Control prob mean: {float(row['control_prob_mean']):.3f}\n\n"
            f"Target logit drop: {float(row['target_logit_drop']):+.3f}\n"
            f"Control logit drop: {float(row['control_logit_drop_mean']):+.3f}\n"
            f"Delta: {float(row['delta_logit_drop']):+.3f}"
        )
        axes[3].text(0.02, 0.98, text, va='top', ha='left', fontsize=9)
        fig.suptitle('Class-aligned Exp2b qualitative case', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        safe_class = str(row['class_name']).replace('/', '_').replace(' ', '_')
        path = output_dir / f"case_{case_id:02d}_{safe_class}_{image_id}.png"
        fig.savefig(path, dpi=190, bbox_inches='tight')
        plt.close(fig)
        saved.append({'image_id': image_id, 'class_name': row['class_name'], 'path': str(path)})

    pd.DataFrame(saved).to_csv(output_dir / 'class_aligned_case_figures.csv', index=False)
    print(f"Saved {len(saved)} class-aligned case figures to {output_dir}")


if __name__ == '__main__':
    main()
