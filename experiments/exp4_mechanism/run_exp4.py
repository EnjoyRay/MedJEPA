"""Run Exp4 token drift and saliency-bbox alignment analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp1_robustness'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp2_lesion_sensitivity'))

from exp1_robustness.linear_probe import LinearProbe
from exp2_lesion_sensitivity.occlusion import (
    boxes_for_class,
    generate_k_control_box_sets,
    merge_overlapping_boxes,
    occlude_lesion,
)
from shared.model_wrappers import build_encoder
from shared.vindr_dataset import CLASS_TO_IDX, VINDR_CLASSES, build_vindr_loader


LOCALIZED = {
    "Atelectasis",
    "Calcification",
    "Consolidation",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pneumothorax",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp4: token drift and saliency-bbox alignment")
    parser.add_argument("--model", required=True, choices=["ijepa", "mae"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--probe_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--in_chans", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", default=300, type=int)
    parser.add_argument("--max_cases", default=12, type=int)
    parser.add_argument("--noise_sigma", default=0.05, type=float)
    parser.add_argument("--num_controls", default=1, type=int)
    return parser.parse_args()


def _disease_group(class_name: str) -> str:
    return "localized" if class_name in LOCALIZED else "diffuse_or_global"


def _grid_size(num_tokens: int) -> int:
    size = int(round(math.sqrt(num_tokens)))
    if size * size != num_tokens:
        raise ValueError(f"Cannot map {num_tokens} tokens to square grid")
    return size


def _bbox_mask(boxes: List[Dict], grid: int, img_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(grid, grid, device=device)
    cell = img_size / grid
    for box in boxes:
        x1 = max(0, int(math.floor(float(box["x_min"]) / cell)))
        y1 = max(0, int(math.floor(float(box["y_min"]) / cell)))
        x2 = min(grid, int(math.ceil(float(box["x_max"]) / cell)))
        y2 = min(grid, int(math.ceil(float(box["y_max"]) / cell)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask


def _token_drift(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(tokens_a, dim=-1)
    b = F.normalize(tokens_b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


def _saliency_from_tokens(encoder, probe: LinearProbe, image: torch.Tensor, class_idx: int) -> tuple[torch.Tensor, float]:
    encoder.zero_grad(set_to_none=True)
    probe.head.zero_grad(set_to_none=True)
    tokens = encoder.forward_tokens(image)
    tokens.retain_grad()
    pooled = tokens.mean(dim=1)
    logit = probe.logits_from_tensor(pooled)[0, class_idx]
    logit.backward()
    sal = (tokens.grad[0] * tokens.detach()[0]).sum(dim=-1).abs()
    sal = sal / sal.sum().clamp_min(1e-8)
    return sal.detach(), float(logit.detach().cpu().item())


def _alignment_metrics(sal: torch.Tensor, bbox_mask: torch.Tensor) -> Dict[str, float]:
    flat_mask = bbox_mask.flatten().float()
    flat_sal = sal.flatten().float()
    inside = float((flat_sal * flat_mask).sum().item())
    outside = float((flat_sal * (1.0 - flat_mask)).sum().item())
    pointing = float(flat_mask[int(torch.argmax(flat_sal).item())].item()) if flat_mask.numel() else 0.0
    soft_iou = float(torch.minimum(flat_sal, flat_mask).sum().item() / torch.maximum(flat_sal, flat_mask).sum().clamp_min(1e-8).item())
    return {
        "inside_saliency_ratio": inside / max(inside + outside, 1e-8),
        "pointing_hit": pointing,
        "soft_iou": soft_iou,
    }


def _img_np(image_01: torch.Tensor) -> np.ndarray:
    arr = image_01.detach().cpu().numpy()
    return arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)


def _save_case(path: Path, image_01: torch.Tensor, boxes: List[Dict], drift: torch.Tensor, sal: torch.Tensor, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    grid = _grid_size(drift.numel())
    drift_map = drift.view(1, 1, grid, grid)
    sal_map = sal.view(1, 1, grid, grid)
    drift_up = F.interpolate(drift_map, size=image_01.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    sal_up = F.interpolate(sal_map, size=image_01.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    axes[0].imshow(_img_np(image_01), cmap="gray", vmin=0, vmax=1)
    for box in boxes:
        rect = patches.Rectangle(
            (box["x_min"], box["y_min"]),
            box["x_max"] - box["x_min"],
            box["y_max"] - box["y_min"],
            linewidth=1.5,
            edgecolor="#7C3AED",
            facecolor="none",
        )
        axes[0].add_patch(rect)
    axes[0].set_title("Original + bbox")
    axes[1].imshow(_img_np(image_01), cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(drift_up, cmap="magma", alpha=0.55)
    axes[1].set_title("Token drift")
    axes[2].imshow(_img_np(image_01), cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(sal_up, cmap="jet", alpha=0.45)
    axes[2].set_title("Gradient x token saliency")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    case_dir = out / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device)
    encoder = build_encoder(args.model, args.weights, device)
    encoder.eval()

    loader = build_vindr_loader(
        args.data_dir,
        split="test",
        img_size=args.img_size,
        in_chans=args.in_chans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    sample = next(iter(loader))
    feat_dim = encoder(sample["image"].to(device)).shape[1]
    probe = LinearProbe(feat_dim, sample["label"].shape[1], device=device)
    probe.load(args.probe_path)

    rows = []
    saved_cases = 0
    processed = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        images_01 = (images * 0.5 + 0.5).clamp(0.0, 1.0)
        labels = batch["label"]
        for i in range(images.shape[0]):
            if processed >= args.max_samples:
                break
            all_boxes = batch["boxes"][i]
            if not all_boxes:
                continue
            image = images[i:i + 1]
            image_01 = images_01[i]
            image_id = str(batch["image_id"][i])
            with torch.no_grad():
                clean_tokens = encoder.forward_tokens(image)
                grid = _grid_size(clean_tokens.shape[1])
                noise_01 = (image_01.unsqueeze(0) + torch.randn_like(image_01.unsqueeze(0)) * args.noise_sigma).clamp(0, 1)
                noise_tokens = encoder.forward_tokens((noise_01 - 0.5) / 0.5)
                noise_drift = _token_drift(clean_tokens, noise_tokens)[0]

            for class_name in VINDR_CLASSES:
                if class_name == "No finding":
                    continue
                class_idx = CLASS_TO_IDX[class_name]
                if float(labels[i, class_idx]) < 0.5:
                    continue
                target_boxes = merge_overlapping_boxes(boxes_for_class(all_boxes, class_name), args.img_size)
                if not target_boxes:
                    continue

                controls = generate_k_control_box_sets(
                    target_boxes,
                    all_boxes,
                    args.img_size,
                    k=args.num_controls,
                    seed=processed + 17,
                    strategy="matched_random",
                )
                if not controls:
                    continue

                bbox_mask = _bbox_mask(target_boxes, grid, args.img_size, device)
                with torch.no_grad():
                    lesion_01 = occlude_lesion(image_01, target_boxes, fill=0.5).unsqueeze(0)
                    control_01 = occlude_lesion(image_01, controls[0], fill=0.5).unsqueeze(0)
                    lesion_tokens = encoder.forward_tokens((lesion_01 - 0.5) / 0.5)
                    control_tokens = encoder.forward_tokens((control_01 - 0.5) / 0.5)
                    lesion_drift = _token_drift(clean_tokens, lesion_tokens)[0]
                    control_drift = _token_drift(clean_tokens, control_tokens)[0]

                sal, clean_logit = _saliency_from_tokens(encoder, probe, image, class_idx)
                align = _alignment_metrics(sal, bbox_mask)
                bbox_flat = bbox_mask.flatten()
                rows.append({
                    "image_id": image_id,
                    "class_name": class_name,
                    "class_idx": class_idx,
                    "disease_group": _disease_group(class_name),
                    "bbox_token_frac": float(bbox_flat.mean().item()),
                    "noise_drift_mean": float(noise_drift.mean().item()),
                    "noise_drift_inside": float((noise_drift * bbox_flat).sum().item() / bbox_flat.sum().clamp_min(1e-8).item()),
                    "lesion_drift_mean": float(lesion_drift.mean().item()),
                    "lesion_drift_inside": float((lesion_drift * bbox_flat).sum().item() / bbox_flat.sum().clamp_min(1e-8).item()),
                    "control_drift_mean": float(control_drift.mean().item()),
                    "clean_logit": clean_logit,
                    **align,
                })
                if saved_cases < args.max_cases:
                    _save_case(
                        case_dir / f"case_{saved_cases:02d}_{class_name.replace('/', '_').replace(' ', '_')}_{image_id}.png",
                        image_01,
                        target_boxes,
                        noise_drift.detach().cpu(),
                        sal.detach().cpu(),
                        f"{class_name} | {image_id}",
                    )
                    saved_cases += 1
                processed += 1
                break
        if processed >= args.max_samples:
            break

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(out / "mechanism_per_sample.csv", index=False)
    summary = per_sample.groupby("disease_group").agg(
        n=("image_id", "count"),
        inside_saliency_ratio_mean=("inside_saliency_ratio", "mean"),
        pointing_hit_mean=("pointing_hit", "mean"),
        soft_iou_mean=("soft_iou", "mean"),
        noise_drift_mean=("noise_drift_mean", "mean"),
        lesion_drift_mean=("lesion_drift_mean", "mean"),
        control_drift_mean=("control_drift_mean", "mean"),
    ).reset_index()
    summary.to_csv(out / "mechanism_summary_by_group.csv", index=False)
    print(out / "mechanism_per_sample.csv")
    print(out / "mechanism_summary_by_group.csv")


if __name__ == "__main__":
    main()
