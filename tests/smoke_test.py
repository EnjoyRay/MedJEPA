"""
Smoke test for the experiment pipeline using synthetic (fake) data.

Creates:
  - A tiny synthetic VinDr-CXR-like dataset (50 train + 30 test images, CSV annotations)
  - A mock encoder (random linear projection, no real weights)
  - Runs both Exp1 and Exp2 end-to-end
  - Verifies output files exist

Run with:
    conda run -n mae python tests/smoke_test.py
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

# Make experiments/ importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))

# ─────────────────────────────────────────────────────────────
# 1.  Mock encoder
# ─────────────────────────────────────────────────────────────

class MockEncoder(nn.Module):
    """Deterministic 'encoder' for testing: mean-pools patches then projects."""
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(224 * 224, feat_dim, bias=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 224, 224)
        B = x.shape[0]
        flat = x.reshape(B, -1)          # (B, 224*224)
        return self.proj(flat)           # (B, feat_dim)


# ─────────────────────────────────────────────────────────────
# 2.  Synthetic VinDr-CXR dataset
# ─────────────────────────────────────────────────────────────

CLASSES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass',
    'Other lesion', 'Pleural effusion', 'Pleural thickening',
    'Pneumothorax', 'Pulmonary fibrosis', 'No finding',
]

def make_synthetic_dataset(root: Path, n_train: int = 50, n_test: int = 30):
    """Create a minimal synthetic VinDr-CXR directory layout."""
    rng = np.random.default_rng(42)

    for split, n in [('train', n_train), ('test', n_test)]:
        img_dir = root / split / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(n):
            img_id = f'{split}_{i:04d}'
            # Save a random grayscale 224x224 PNG
            arr = (rng.random((224, 224)) * 255).astype(np.uint8)
            Image.fromarray(arr, mode='L').save(img_dir / f'{img_id}.png')

            # Random label (1-3 classes per image)
            n_classes = rng.integers(1, 4)
            chosen = rng.choice(CLASSES[:-1], size=n_classes, replace=False)
            for cls in chosen:
                # Random bounding box (only for train; test may have boxes too)
                x1 = float(rng.integers(20, 80))
                y1 = float(rng.integers(20, 80))
                x2 = x1 + float(rng.integers(30, 80))
                y2 = y1 + float(rng.integers(30, 80))
                x2, y2 = min(x2, 220.0), min(y2, 220.0)
                rows.append({
                    'image_id': img_id,
                    'class_name': cls,
                    'x_min': x1, 'y_min': y1,
                    'x_max': x2, 'y_max': y2,
                    'rad_id': 'R001',
                })

        ann_dir = root / 'annotations'
        ann_dir.mkdir(exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(ann_dir / f'annotations_{split}.csv', index=False)
        print(f'  [{split}] {n} images, {len(rows)} annotation rows')

    return root


# ─────────────────────────────────────────────────────────────
# 3.  Monkey-patch build_encoder to return MockEncoder
# ─────────────────────────────────────────────────────────────

def mock_build_encoder(model_type, weights_path, device):
    enc = MockEncoder(feat_dim=128)
    enc.eval().to(device)
    return enc


# ─────────────────────────────────────────────────────────────
# 4.  Run Exp1
# ─────────────────────────────────────────────────────────────

def run_exp1(data_dir: Path, output_dir: Path):
    print('\n======  Exp1: Robustness  ======')
    import shared.model_wrappers as mw
    mw.build_encoder = mock_build_encoder        # patch

    from shared.vindr_dataset import build_vindr_loader
    from exp1_robustness.linear_probe import LinearProbe
    from exp1_robustness.evaluate_robustness import (
        evaluate_robustness, save_results_csv, plot_results,
    )

    device = torch.device('cpu')
    encoder = mock_build_encoder('mock', 'fake.pth', device)

    train_loader = build_vindr_loader(str(data_dir), split='train',
                                      img_size=224, in_chans=1,
                                      batch_size=16, num_workers=0)
    test_loader  = build_vindr_loader(str(data_dir), split='test',
                                      img_size=224, in_chans=1,
                                      batch_size=16, num_workers=0)

    # Extract train embeddings
    all_embs, all_labels = [], []
    with torch.no_grad():
        for batch in train_loader:
            embs = encoder(batch['image'].to(device))
            all_embs.append(embs.cpu())
            all_labels.append(batch['label'])
    train_embs  = torch.cat(all_embs).float().numpy()
    train_labels = torch.cat(all_labels).float().numpy()
    print(f'  train embs: {train_embs.shape}  labels: {train_labels.shape}')

    # Train probe (few epochs for speed)
    probe = LinearProbe(feat_dim=128, num_classes=15,
                        lr=1e-2, epochs=5, batch_size=32, device=device)
    probe.fit(train_embs, train_labels)

    # Evaluate robustness (only 2 perturbation types to keep test fast)
    results = evaluate_robustness(
        encoder=encoder, probe=probe, loader=test_loader, device=device,
        perturbation_types=['gaussian_noise', 'gaussian_blur'],
    )

    save_results_csv(results, str(output_dir))
    plot_results(results, str(output_dir))

    # Check output files
    expected = ['robustness_curve.csv', 'embedding_drift.csv']
    for fname in expected:
        fpath = output_dir / fname
        assert fpath.exists(), f'Missing output: {fpath}'
        print(f'  [OK] {fname}')


# ─────────────────────────────────────────────────────────────
# 5.  Run Exp2
# ─────────────────────────────────────────────────────────────

def run_exp2(data_dir: Path, output_dir: Path):
    print('\n======  Exp2: Lesion Sensitivity  ======')
    from shared.vindr_dataset import build_vindr_loader
    from exp1_robustness.linear_probe import LinearProbe
    from exp2_lesion_sensitivity.evaluate_sensitivity import (
        evaluate_sensitivity, save_results_csv, plot_results,
    )

    device = torch.device('cpu')
    encoder = MockEncoder(feat_dim=128).eval()

    train_loader = build_vindr_loader(str(data_dir), split='train',
                                      img_size=224, in_chans=1,
                                      batch_size=16, num_workers=0)
    test_loader  = build_vindr_loader(str(data_dir), split='test',
                                      img_size=224, in_chans=1,
                                      batch_size=8, num_workers=0)

    # Train probe
    all_embs, all_labels = [], []
    with torch.no_grad():
        for batch in train_loader:
            embs = encoder(batch['image'].to(device))
            all_embs.append(embs.cpu())
            all_labels.append(batch['label'])
    train_embs   = torch.cat(all_embs).float().numpy()
    train_labels = torch.cat(all_labels).float().numpy()

    probe = LinearProbe(feat_dim=128, num_classes=15,
                        lr=1e-2, epochs=5, batch_size=32, device=device)
    probe.fit(train_embs, train_labels)

    # Evaluate sensitivity
    results = evaluate_sensitivity(
        encoder=encoder, probe=probe,
        loader=test_loader, device=device,
        control_strategy='opposite',
    )

    save_results_csv(results, str(output_dir))
    plot_results(results, str(output_dir))

    expected = ['prediction_drop.csv', 'feature_drift.csv',
                'prediction_drop_per_class.csv', 'scores.npz']
    for fname in expected:
        fpath = output_dir / fname
        assert fpath.exists(), f'Missing output: {fpath}'
        print(f'  [OK] {fname}')


# ─────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────

def main():
    tmp = Path(tempfile.mkdtemp(prefix='jepa_smoke_'))
    data_dir   = tmp / 'vindr'
    out_exp1   = tmp / 'results_exp1'
    out_exp2   = tmp / 'results_exp2'

    print(f'Temp dir: {tmp}')

    try:
        print('\n── Generating synthetic dataset ──')
        make_synthetic_dataset(data_dir)

        run_exp1(data_dir, out_exp1)
        run_exp2(data_dir, out_exp2)

        print('\n══════════════════════════════════')
        print('  ALL SMOKE TESTS PASSED')
        print('══════════════════════════════════')

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
