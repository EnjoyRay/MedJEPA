"""
Occlusion utilities for Exp2 (lesion sensitivity experiment).

Given an image and a bounding box, this module creates:
  - A lesion-occluded image (bbox filled with a neutral value)
  - A control-occluded image (equivalent-sized region outside the bbox, filled the same way)

The fill value defaults to 0.5 (mid-grey after [0,1] normalization), which is
neutral for chest X-rays and avoids introducing spurious texture artifacts.

Usage:
    from occlusion import occlude_lesion, occlude_control, occlude_batch
"""

import torch
import random
import math
from typing import List, Dict, Optional, Tuple


# Fill value after denormalizing to [0,1]: 0.5 = mid-grey
# After re-normalizing with mean=0.5, std=0.5 this maps to 0.0 (zero-fill in model space)
FILL_VALUE_01 = 0.5


def _box_to_pixels(
    box: dict,
    img_size: int,
) -> Tuple[int, int, int, int]:
    """Clamp and round bounding box coordinates to valid pixel indices."""
    x_min = max(0, int(math.floor(box['x_min'])))
    y_min = max(0, int(math.floor(box['y_min'])))
    x_max = min(img_size, int(math.ceil(box['x_max'])))
    y_max = min(img_size, int(math.ceil(box['y_max'])))
    return x_min, y_min, x_max, y_max


def occlude_lesion(
    image_01: torch.Tensor,
    boxes: List[Dict],
    fill: float = FILL_VALUE_01,
) -> torch.Tensor:
    """
    Fill all annotated lesion bounding boxes with a neutral value.

    Args:
        image_01: (C, H, W) float tensor in [0, 1]
        boxes: list of dicts with x_min, y_min, x_max, y_max (already scaled to img_size)
        fill: fill value in [0, 1]
    Returns:
        occluded image of same shape
    """
    img = image_01.clone()
    _, H, W = img.shape
    for box in boxes:
        x_min, y_min, x_max, y_max = _box_to_pixels(box, W)
        if x_max > x_min and y_max > y_min:
            img[:, y_min:y_max, x_min:x_max] = fill
    return img


def occlude_control(
    image_01: torch.Tensor,
    boxes: List[Dict],
    fill: float = FILL_VALUE_01,
    strategy: str = 'opposite',
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Fill a control region (same total area as lesion boxes, outside the lesion).

    Args:
        image_01: (C, H, W) float tensor in [0, 1]
        boxes: list of lesion bounding boxes (scaled to img_size)
        fill: fill value in [0, 1]
        strategy: 'opposite' mirrors the box to the other side;
                  'random' picks a random non-overlapping region.
        seed: optional RNG seed for 'random' strategy
    Returns:
        control-occluded image of same shape
    """
    img = image_01.clone()
    _, H, W = img.shape

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    for box in boxes:
        x_min, y_min, x_max, y_max = _box_to_pixels(box, W)
        bw = x_max - x_min
        bh = y_max - y_min
        if bw <= 0 or bh <= 0:
            continue

        if strategy == 'opposite':
            # Mirror horizontally: flip x about image center
            cx_flip = W - x_max
            cy_flip = y_min  # keep vertical position
            cx2 = min(W, cx_flip + bw)
            cy2 = min(H, cy_flip + bh)
            if cx2 > cx_flip and cy2 > cy_flip:
                img[:, cy_flip:cy2, cx_flip:cx2] = fill

        elif strategy == 'random':
            # Try up to 20 random placements that don't overlap with any lesion box
            placed = False
            for _ in range(20):
                rx = rng.randint(0, max(0, W - bw))
                ry = rng.randint(0, max(0, H - bh))
                rx2, ry2 = rx + bw, ry + bh
                # Check no overlap with any lesion box
                overlap = False
                for b2 in boxes:
                    bx1, by1, bx2, by2 = _box_to_pixels(b2, W)
                    if rx < bx2 and rx2 > bx1 and ry < by2 and ry2 > by1:
                        overlap = True
                        break
                if not overlap:
                    img[:, ry:ry2, rx:rx2] = fill
                    placed = True
                    break
            if not placed:
                # Fallback: use opposite strategy
                cx_flip = W - x_max
                cy_flip = y_min
                cx2 = min(W, cx_flip + bw)
                cy2 = min(H, cy_flip + bh)
                img[:, cy_flip:cy2, cx_flip:cx2] = fill
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

    return img


def occlude_batch(
    images_01: torch.Tensor,
    batch_boxes: List[List[Dict]],
    mode: str = 'lesion',
    fill: float = FILL_VALUE_01,
    strategy: str = 'opposite',
) -> torch.Tensor:
    """
    Apply occlusion to a batch of images.

    Args:
        images_01: (B, C, H, W) float tensor in [0, 1]
        batch_boxes: list of length B, each element is a list of bbox dicts
        mode: 'lesion' or 'control'
        fill: fill value
        strategy: 'opposite' or 'random' (used when mode='control')
    Returns:
        (B, C, H, W) tensor with occluded regions
    """
    B = images_01.shape[0]
    occluded = []
    for i in range(B):
        boxes = batch_boxes[i]
        if len(boxes) == 0:
            # No annotation: skip this image (copy unchanged)
            occluded.append(images_01[i])
            continue
        if mode == 'lesion':
            out = occlude_lesion(images_01[i], boxes, fill=fill)
        elif mode == 'control':
            out = occlude_control(images_01[i], boxes, fill=fill, strategy=strategy, seed=i)
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        occluded.append(out)
    return torch.stack(occluded)


def boxes_for_class(boxes: List[Dict], class_name: str) -> List[Dict]:
    """Return boxes whose annotation class exactly matches class_name."""
    return [box for box in boxes if str(box.get('class_name', '')) == class_name]


def merge_overlapping_boxes(boxes: List[Dict], img_size: int) -> List[Dict]:
    """Merge overlapping boxes into connected bbox components.

    Multiple radiologists can annotate the same abnormality with highly
    overlapping boxes.  Occluding each box separately would make matched control
    occlusion cover a much larger non-overlapping area than the true lesion
    union.  This function collapses those duplicate/overlapping annotations
    before lesion and control masks are generated.
    """
    if not boxes:
        return []

    components: List[Dict] = []
    for box in boxes:
        x1, y1, x2, y2 = _box_to_pixels(box, img_size)
        if x2 <= x1 or y2 <= y1:
            continue
        current = {
            'class_name': str(box.get('class_name', 'lesion')),
            'x_min': float(x1),
            'y_min': float(y1),
            'x_max': float(x2),
            'y_max': float(y2),
        }

        merged = True
        while merged:
            merged = False
            kept: List[Dict] = []
            for existing in components:
                if boxes_overlap(current, existing, img_size):
                    ex1, ey1, ex2, ey2 = _box_to_pixels(existing, img_size)
                    current['x_min'] = float(min(int(current['x_min']), ex1))
                    current['y_min'] = float(min(int(current['y_min']), ey1))
                    current['x_max'] = float(max(int(current['x_max']), ex2))
                    current['y_max'] = float(max(int(current['y_max']), ey2))
                    merged = True
                else:
                    kept.append(existing)
            components = kept
        components.append(current)

    return components


def box_area_fraction(boxes: List[Dict], img_size: int) -> Tuple[float, float]:
    """Return total and max bbox area fraction for scaled boxes."""
    areas = []
    for box in boxes:
        x_min, y_min, x_max, y_max = _box_to_pixels(box, img_size)
        area = max(0, x_max - x_min) * max(0, y_max - y_min)
        if area > 0:
            areas.append(area)
    image_area = float(img_size * img_size)
    if not areas:
        return 0.0, 0.0
    return min(float(sum(areas)) / image_area, 1.0), float(max(areas)) / image_area


def boxes_overlap(a: Dict, b: Dict, img_size: int) -> bool:
    ax1, ay1, ax2, ay2 = _box_to_pixels(a, img_size)
    bx1, by1, bx2, by2 = _box_to_pixels(b, img_size)
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _candidate_box(x: int, y: int, width: int, height: int, img_size: int, class_name: str) -> Dict:
    return {
        'class_name': class_name,
        'x_min': float(max(0, x)),
        'y_min': float(max(0, y)),
        'x_max': float(min(img_size, x + width)),
        'y_max': float(min(img_size, y + height)),
    }


def _is_valid_control(candidate: Dict, forbidden_boxes: List[Dict], img_size: int) -> bool:
    x1, y1, x2, y2 = _box_to_pixels(candidate, img_size)
    if x2 <= x1 or y2 <= y1:
        return False
    return not any(boxes_overlap(candidate, forbidden, img_size) for forbidden in forbidden_boxes)


def generate_matched_control_boxes(
    target_boxes: List[Dict],
    forbidden_boxes: List[Dict],
    img_size: int,
    seed: int = 0,
    strategy: str = 'matched_random',
    max_tries: int = 100,
    allow_fallback: bool = False,
) -> List[Dict]:
    """
    Generate one matched control set with the same per-box sizes as target_boxes.

    The control boxes are constrained to avoid all forbidden boxes.  The default
    strategy samples in the same vertical band first, because chest X-ray anatomy
    varies strongly with y-position.  It deliberately does not use a mirror box
    by default: a valid mirror can still cover important unlabeled anatomy.
    """
    rng = random.Random(seed)
    controls: List[Dict] = []
    occupied = list(forbidden_boxes)

    for target in target_boxes:
        x_min, y_min, x_max, y_max = _box_to_pixels(target, img_size)
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            continue

        cls = str(target.get('class_name', 'control'))
        candidates: List[Dict] = []

        if strategy in ('mirror', 'opposite'):
            # Anatomical mirror candidate: same y band, opposite x side.
            mx = img_size - x_max
            candidates.append(_candidate_box(mx, y_min, width, height, img_size, cls))

        if strategy in ('matched_random', 'mixed', 'random', 'strict_random'):
            y_low = max(0, y_min - max(8, height // 2))
            y_high = min(max(0, img_size - height), y_min + max(8, height // 2))
            for _ in range(max_tries):
                if y_high >= y_low:
                    ry = rng.randint(y_low, y_high)
                else:
                    ry = rng.randint(0, max(0, img_size - height))
                rx = rng.randint(0, max(0, img_size - width))
                candidates.append(_candidate_box(rx, ry, width, height, img_size, cls))
            for _ in range(max_tries):
                rx = rng.randint(0, max(0, img_size - width))
                ry = rng.randint(0, max(0, img_size - height))
                candidates.append(_candidate_box(rx, ry, width, height, img_size, cls))

        placed = None
        for candidate in candidates:
            if _is_valid_control(candidate, occupied, img_size):
                placed = candidate
                break

        # If the image is almost fully annotated, no non-overlap control may
        # exist.  By default we reject this control set instead of using a
        # possibly lesion-overlapping fallback, because Exp2b is intended as a
        # clean paired lesion-vs-control test.
        if placed is None:
            if not allow_fallback:
                return []
            if candidates:
                placed = dict(candidates[0])
                placed['fallback_overlap'] = True
            else:
                return []
        else:
            placed = dict(placed)
            placed['fallback_overlap'] = False

        controls.append(placed)
        occupied.append(placed)

    return controls


def generate_k_control_box_sets(
    target_boxes: List[Dict],
    forbidden_boxes: List[Dict],
    img_size: int,
    k: int = 5,
    seed: int = 0,
    strategy: str = 'matched_random',
) -> List[List[Dict]]:
    """Generate k independently sampled matched control box sets."""
    controls: List[List[Dict]] = []
    attempts = max(k * 20, k)
    for i in range(attempts):
        box_set = generate_matched_control_boxes(
            target_boxes=target_boxes,
            forbidden_boxes=forbidden_boxes,
            img_size=img_size,
            seed=seed + 1009 * i,
            strategy=strategy,
        )
        if box_set:
            controls.append(box_set)
        if len(controls) >= k:
            break
    return controls
