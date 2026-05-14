"""
VinDr-CXR / VinBigData-ChestXray dataset loader.

Supports two dataset formats:
  1. VinDr-CXR: annotations_{split}.csv with image_id, class_name, x_min, ...
  2. VinBigData-ChestXray: {split}.csv with image_id, class_name, class_id, ...

Directory structure (VinBigData, default):
    <data_dir>/
        images_1024/
            train/           ← PNG files named by image_id
            test/
        annotations/
            train.csv
            train_meta.csv

Directory structure (VinDr-CXR):
    <data_dir>/
        train/
            images/
        test/
            images/
        annotations/
            annotations_train.csv

Usage:
    from shared.vindr_dataset import VinDrCXRDataset, build_vindr_loader
"""

import os
import ast
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Class names in VinDr-CXR (14 findings + "No finding")
# ---------------------------------------------------------------------------

VINDR_CLASSES = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'No finding',
]

CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(VINDR_CLASSES)}
NUM_CLASSES = len(VINDR_CLASSES)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VinDrCXRDataset(Dataset):
    """
    VinDr-CXR dataset.

    Each sample returns:
        image   : (1, H, W) or (3, H, W) float tensor (normalized)
        label   : (NUM_CLASSES,) float binary multi-label vector
        boxes   : list of dicts with keys {class_name, x_min, y_min, x_max, y_max}
        image_id: str
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'test',
        img_size: int = 224,
        in_chans: int = 1,
        transform: Optional[T.Compose] = None,
    ):
        """
        Args:
            data_dir: root directory of VinDr-CXR
            split: 'train' or 'test'
            img_size: resize target (square)
            in_chans: 1 (grayscale) or 3 (RGB duplicate)
            transform: optional override transform
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.in_chans = in_chans

        self.image_dir = self._find_image_dir()
        self.image_meta = self._load_image_meta()
        self.annotations, self.image_ids = self._load_annotations()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform()

    def _find_image_dir(self) -> Path:
        # Support both VinBigData and VinDr-CXR layouts
        candidates = [
            self.data_dir / f'images_1024' / self.split,          # VinBigData
            self.data_dir / self.split / 'images',                 # VinDr-CXR
            self.data_dir / self.split,
            self.data_dir / 'images' / self.split,
            self.data_dir / 'images',
        ]
        for c in candidates:
            if c.is_dir():
                return c
        # Fallback: when test split uses sampled train data, images may be in train dir
        if self.split in ('test', 'val', 'validation'):
            for prefix in ['train', 'training']:
                fallback = self.data_dir / f'images_1024' / prefix
                if fallback.is_dir():
                    print(f"[VinDr-CXR] Image dir for '{self.split}' not found, using '{fallback}'", flush=True)
                    return fallback
                fallback = self.data_dir / prefix / 'images'
                if fallback.is_dir():
                    print(f"[VinDr-CXR] Image dir for '{self.split}' not found, using '{fallback}'", flush=True)
                    return fallback
        raise FileNotFoundError(
            f"Cannot find image directory for split='{self.split}' under {self.data_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def _find_annotation_csv(self) -> Path:
        candidates = [
            self.data_dir / 'annotations' / f'{self.split}.csv',          # VinBigData
            self.data_dir / 'annotations' / f'annotations_{self.split}.csv',  # VinDr-CXR
            self.data_dir / f'annotations_{self.split}.csv',
            self.data_dir / 'annotations' / f'image_labels_{self.split}.csv',
            self.data_dir / f'image_labels_{self.split}.csv',
        ]
        for c in candidates:
            if c.is_file():
                return c
        # If test split not found, fall back to train.csv (VinBigData has no public test labels)
        if self.split in ('test', 'val', 'validation'):
            train_csv = self.data_dir / 'annotations' / 'train.csv'
            if train_csv.is_file():
                print(f"[VinDr-CXR] No annotation CSV for split='{self.split}', "
                      f"falling back to train.csv (will sample subset)")
                return train_csv
        raise FileNotFoundError(
            f"Cannot find annotation CSV for split='{self.split}' under {self.data_dir}."
        )

    def _load_image_meta(self) -> Dict[str, Tuple[float, float]]:
        """Load original image sizes used by VinBigData bbox annotations.

        The local VinBigData release stores resized 1024x1024 PNGs, but
        train.csv bounding boxes are in the original DICOM coordinate system.
        images_1024/train_meta.csv provides that original height/width.
        """
        candidates = [
            self.data_dir / 'images_1024' / f'{self.split}_meta.csv',
            self.data_dir / 'images_1024' / 'train_meta.csv',
            self.data_dir / 'annotations' / f'{self.split}_meta.csv',
            self.data_dir / 'annotations' / 'train_meta.csv',
        ]
        for path in candidates:
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            if not {'image_id', 'dim0', 'dim1'}.issubset(df.columns):
                continue
            meta: Dict[str, Tuple[float, float]] = {}
            for _, row in df.iterrows():
                # VinBigData convention: dim0 = original height, dim1 = original width.
                meta[str(row['image_id'])] = (float(row['dim1']), float(row['dim0']))
            print(f"[VinDr-CXR] Loaded image metadata from {path}", flush=True)
            return meta
        return {}

    def _load_annotations(self) -> Tuple[Dict[str, dict], List[str]]:
        csv_path = self._find_annotation_csv()
        df = pd.read_csv(csv_path)

        # VinBigData test labels are not present in the local release used here.
        # In that case, create a deterministic disjoint train/held-out split from
        # train.csv.  Earlier versions only sampled held-out IDs for split='test'
        # while split='train' still used all IDs, which allowed train/test overlap.
        self._is_fallback = False
        native_test_csv = any([
            (self.data_dir / 'annotations' / 'test.csv').is_file(),
            (self.data_dir / 'annotations' / 'annotations_test.csv').is_file(),
            (self.data_dir / 'annotations_test.csv').is_file(),
            (self.data_dir / 'annotations' / 'image_labels_test.csv').is_file(),
            (self.data_dir / 'image_labels_test.csv').is_file(),
        ])
        using_train_csv = csv_path.name == 'train.csv'
        needs_fallback_split = using_train_csv and not native_test_csv
        if needs_fallback_split and self.split in ('train', 'test', 'val', 'validation'):
            all_image_ids = np.array(sorted(df['image_id'].astype(str).unique()))
            rng = np.random.RandomState(42)
            n_test = max(500, int(len(all_image_ids) * 0.2))
            holdout_ids = set(rng.choice(all_image_ids, size=n_test, replace=False))

            if self.split in ('test', 'val', 'validation'):
                self._is_fallback = True
                # Override image_dir to point to train images (since held-out IDs
                # are sampled from train.csv).
                train_img_dir = self.data_dir / 'images_1024' / 'train'
                if train_img_dir.is_dir():
                    self.image_dir = train_img_dir
                    print(f"[VinDr-CXR] Using image dir: {self.image_dir}", flush=True)
                df = df[df['image_id'].astype(str).isin(holdout_ids)]
                print(
                    f"[VinDr-CXR] Using deterministic held-out split: "
                    f"{len(holdout_ids)} images from train.csv as {self.split} set",
                    flush=True,
                )
            else:
                df = df[~df['image_id'].astype(str).isin(holdout_ids)]
                print(
                    f"[VinDr-CXR] Using deterministic train split: "
                    f"{df['image_id'].nunique()} images "
                    f"(excluded {len(holdout_ids)} held-out images)",
                    flush=True,
                )

        annotations: Dict[str, dict] = {}  # image_id → {label, boxes}

        for image_id, group in df.groupby('image_id'):
            label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            boxes = []
            for _, row in group.iterrows():
                cls_name = row.get('class_name', 'No finding')
                if cls_name in CLASS_TO_IDX:
                    label_vec[CLASS_TO_IDX[cls_name]] = 1.0
                # Boxes may not exist in test split
                if all(c in row.index for c in ['x_min', 'y_min', 'x_max', 'y_max']):
                    if not any(pd.isna(row[c]) for c in ['x_min', 'y_min', 'x_max', 'y_max']):
                        boxes.append({
                            'class_name': cls_name,
                            'x_min': float(row['x_min']),
                            'y_min': float(row['y_min']),
                            'x_max': float(row['x_max']),
                            'y_max': float(row['y_max']),
                        })
            annotations[str(image_id)] = {
                'label': label_vec,
                'boxes': boxes,
            }

        image_ids = list(annotations.keys())
        return annotations, image_ids

    def _default_transform(self) -> T.Compose:
        # Images are always loaded as RGB; we convert to grayscale here if needed.
        mean = [0.5] * self.in_chans
        std = [0.5] * self.in_chans
        transforms_list = [
            T.Grayscale(num_output_channels=self.in_chans),
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        return T.Compose(transforms_list)

    def _load_image(self, image_id: str) -> Image.Image:
        for ext in ['.png', '.jpg', '.jpeg', '.dicom', '.dcm']:
            path = self.image_dir / (image_id + ext)
            if path.is_file():
                if ext in ('.dicom', '.dcm'):
                    return self._load_dicom(path)
                # Always load as RGB first; Grayscale conversion is handled in transform.
                # This avoids ToTensor issues with 'L'-mode images in older torchvision.
                img = Image.open(path).convert('RGB')
                return img
        raise FileNotFoundError(f"Image not found for id={image_id} in {self.image_dir}")

    @staticmethod
    def _load_dicom(path: Path) -> Image.Image:
        try:
            import pydicom
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
            return Image.fromarray(arr.astype(np.uint8)).convert('L')
        except ImportError:
            raise ImportError("pydicom is required to read DICOM files: pip install pydicom")

    def scale_boxes(
        self,
        boxes: List[dict],
        orig_size: Tuple[int, int],
        image_id: Optional[str] = None,
    ) -> List[dict]:
        """Scale bounding boxes from annotation coordinates to img_size."""
        if image_id is not None and image_id in self.image_meta:
            orig_w, orig_h = self.image_meta[image_id]
        else:
            orig_w, orig_h = orig_size
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        scaled = []
        for box in boxes:
            scaled.append({
                'class_name': box['class_name'],
                'x_min': box['x_min'] * scale_x,
                'y_min': box['y_min'] * scale_y,
                'x_max': box['x_max'] * scale_x,
                'y_max': box['y_max'] * scale_y,
            })
        return scaled

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        ann = self.annotations[image_id]

        img = self._load_image(image_id)
        orig_size = img.size  # (W, H)
        scaled_boxes = self.scale_boxes(ann['boxes'], orig_size, image_id=image_id)

        img_tensor = self.transform(img)      # (C, H, W)

        return {
            'image': img_tensor,
            'label': torch.from_numpy(ann['label']),
            'boxes': scaled_boxes,            # list of dicts (scaled to img_size)
            'image_id': image_id,
        }


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_vindr_loader(
    data_dir: str,
    split: str = 'test',
    img_size: int = 224,
    in_chans: int = 1,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    dataset = VinDrCXRDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        in_chans=in_chans,
    )

    def collate_fn(batch):
        images = torch.stack([b['image'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        boxes = [b['boxes'] for b in batch]
        image_ids = [b['image_id'] for b in batch]
        return {
            'image': images,
            'label': labels,
            'boxes': boxes,
            'image_id': image_ids,
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[VinDr-CXR] {split} split: {len(dataset)} images, {len(loader)} batches")
    return loader
