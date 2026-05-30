"""
Exp2 — Lesion Region Sensitivity
=================================

Main entry point. Run with:

    python experiments/exp2_lesion_sensitivity/run_exp2.py \\
        --model ijepa \\
        --weights /path/to/ijepa_mimic.pth \\
        --data_dir /path/to/vindr-cxr \\
        --output_dir results/exp2_ijepa

    python experiments/exp2_lesion_sensitivity/run_exp2.py \\
        --model mae \\
        --weights /path/to/mae_mimic.pth \\
        --data_dir /path/to/vindr-cxr \\
        --output_dir results/exp2_mae

    # Use a probe trained in Exp1 (recommended: same model/config)
    python experiments/exp2_lesion_sensitivity/run_exp2.py \\
        --model ijepa \\
        --weights /path/to/ijepa_mimic.pth \\
        --data_dir /path/to/vindr-cxr \\
        --probe_path results/exp1_ijepa/probe.pth \\
        --output_dir results/exp2_ijepa

Pipeline:
  1. Load encoder (frozen)
  2. Train (or load) a linear probe on clean train embeddings
  3. For each test image with lesion annotations:
     a. Compute original embedding + prediction
     b. Occlude lesion region → embedding + prediction
     c. Occlude control region → embedding + prediction
  4. Compute prediction drop and embedding drift for both conditions
  5. Save CSVs and plots
"""

import argparse
import json
import os
import sys

import torch
import numpy as np

# Make shared/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.model_wrappers import build_encoder
from shared.vindr_dataset import build_vindr_loader

# Import linear probe (from exp1; reuse or retrain)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp1_robustness'))
from linear_probe import LinearProbe

from evaluate_sensitivity import evaluate_sensitivity, save_results_csv, plot_results


def get_args():
    parser = argparse.ArgumentParser(description='Exp2: Lesion Region Sensitivity')

    # Model
    parser.add_argument('--model', required=True, choices=['ijepa', 'mae', 'moco'])
    parser.add_argument('--weights', required=True,
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--in_chans', default=1, type=int)

    # Data
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # Probe
    parser.add_argument('--probe_path', default=None,
                        help='Path to pre-trained probe .pth (skip training if provided)')
    parser.add_argument('--probe_epochs', default=50, type=int)
    parser.add_argument('--probe_lr', default=1e-3, type=float)
    parser.add_argument('--probe_wd', default=1e-4, type=float)

    # Occlusion
    parser.add_argument('--control_strategy', default='opposite',
                        choices=['opposite', 'random'],
                        help='Strategy for choosing control occlusion region')

    # Output
    parser.add_argument('--output_dir', required=True)

    # Misc
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', default=42, type=int)

    return parser.parse_args()


@torch.no_grad()
def extract_all_embeddings(encoder, loader, device):
    all_embs, all_labels = [], []
    for i, batch in enumerate(loader):
        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(loader)}...", flush=True)
        try:
            images = batch['image'].to(device, non_blocking=True)
            embs = encoder(images)
            all_embs.append(embs.cpu())
            all_labels.append(batch['label'])
        except Exception as e:
            print(f"  ERROR at batch {i}: {e}, skipping", flush=True)
            continue
    if not all_embs:
        raise RuntimeError("No embeddings extracted!")
    embs_np = torch.cat(all_embs).float().numpy()
    labels_np = torch.cat(all_labels).float().numpy()
    return embs_np, labels_np


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # -------------------------------------------------------------------------
    # 1. Load encoder
    # -------------------------------------------------------------------------
    print(f"\n=== Loading {args.model.upper()} encoder ===")
    encoder = build_encoder(args.model, args.weights, device)
    encoder.eval()

    # -------------------------------------------------------------------------
    # 2. Build data loaders
    # -------------------------------------------------------------------------
    print("\n=== Building data loaders ===")
    train_loader = build_vindr_loader(
        args.data_dir, split='train',
        img_size=args.img_size, in_chans=args.in_chans,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    test_loader = build_vindr_loader(
        args.data_dir, split='test',
        img_size=args.img_size, in_chans=args.in_chans,
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False,
    )

    # -------------------------------------------------------------------------
    # 3. Train (or load) linear probe
    # -------------------------------------------------------------------------
    probe_save_path = os.path.join(args.output_dir, 'probe.pth')

    if args.probe_path is not None:
        print(f"\n=== Loading pre-trained probe from {args.probe_path} ===")
        # Infer dimensions by running one batch
        print("Inferring feature dimension from one batch...")
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            sample_emb = encoder(sample_batch['image'][:1].to(device))
        feat_dim = sample_emb.shape[-1]
        num_classes = sample_batch['label'].shape[-1]
        probe = LinearProbe(feat_dim=feat_dim, num_classes=num_classes, device=device)
        probe.load(args.probe_path)

    else:
        print("\n=== Training linear probe ===")
        train_embs, train_labels = extract_all_embeddings(encoder, train_loader, device)
        feat_dim = train_embs.shape[1]
        num_classes = train_labels.shape[1]
        print(f"Train embeddings: {train_embs.shape}  Labels: {train_labels.shape}")

        probe = LinearProbe(
            feat_dim=feat_dim, num_classes=num_classes,
            lr=args.probe_lr, epochs=args.probe_epochs,
            weight_decay=args.probe_wd, device=device,
        )
        probe.fit(train_embs, train_labels)
        probe.save(probe_save_path)

    # -------------------------------------------------------------------------
    # 4. Evaluate lesion sensitivity
    # -------------------------------------------------------------------------
    print("\n=== Evaluating lesion sensitivity ===")
    results = evaluate_sensitivity(
        encoder=encoder,
        probe=probe,
        loader=test_loader,
        device=device,
        control_strategy=args.control_strategy,
    )

    # -------------------------------------------------------------------------
    # 5. Save results
    # -------------------------------------------------------------------------
    print("\n=== Saving results ===")
    save_results_csv(results, args.output_dir)
    plot_results(results, args.output_dir)

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
