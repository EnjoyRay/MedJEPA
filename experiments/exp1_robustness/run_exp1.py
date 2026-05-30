"""
Exp1 — Robustness to Nuisance Perturbations
============================================

Main entry point. Run with:

    python experiments/exp1_robustness/run_exp1.py \\
        --model ijepa \\
        --weights /path/to/ijepa_mimic.pth \\
        --data_dir /path/to/vindr-cxr \\
        --output_dir results/exp1_ijepa

    python experiments/exp1_robustness/run_exp1.py \\
        --model mae \\
        --weights /path/to/mae_mimic.pth \\
        --data_dir /path/to/vindr-cxr \\
        --output_dir results/exp1_mae

Pipeline:
  1. Load encoder (frozen)
  2. Extract train embeddings → train linear probe
  3. Extract test embeddings (clean + perturbed) → evaluate AUROC / F1 / cosine drift
  4. Save CSVs and plots to output_dir
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
from linear_probe import LinearProbe
from evaluate_robustness import evaluate_robustness, save_results_csv, plot_results


def get_args():
    parser = argparse.ArgumentParser(description='Exp1: Robustness to Nuisance Perturbations')

    # Model
    parser.add_argument('--model', required=True, choices=['ijepa', 'mae', 'moco'],
                        help='Which encoder to evaluate')
    parser.add_argument('--weights', required=True,
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--in_chans', default=1, type=int,
                        help='1 for grayscale, 3 for RGB (ViT trained on 3-channel)')

    # Data
    parser.add_argument('--data_dir', required=True,
                        help='Root directory of VinDr-CXR')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # Probe training
    parser.add_argument('--probe_epochs', default=50, type=int)
    parser.add_argument('--probe_lr', default=1e-3, type=float)
    parser.add_argument('--probe_wd', default=1e-4, type=float)

    # Perturbations
    parser.add_argument('--pert_types', nargs='+',
                        default=['gaussian_noise', 'gaussian_blur', 'brightness', 'contrast'],
                        help='Perturbation types to evaluate')

    # Output
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save results')

    # Misc
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--probe_path', default=None,
                        help='Path to a pre-trained probe (skip probe training)')
    parser.add_argument('--max_train_batches', default=None, type=int,
                        help='Optional smoke-test limit for probe embedding extraction')
    parser.add_argument('--max_eval_batches', default=None, type=int,
                        help='Optional smoke-test limit for robustness evaluation')

    return parser.parse_args()


@torch.no_grad()
def extract_all_embeddings(encoder, loader, device, save_path=None, max_batches=None):
    """Extract embeddings and labels from a DataLoader.

    If save_path is provided, uses numpy memmap to avoid OOM on large datasets.
    """
    import torch
    import numpy as np

    # First pass: determine shapes with one batch
    sample_batch = next(iter(loader))
    sample_img = sample_batch['image'].to(device, non_blocking=True)
    sample_emb = encoder(sample_img)
    feat_dim = sample_emb.shape[1]
    num_labels = sample_batch['label'].shape[1]
    n_samples = len(loader.dataset)
    if max_batches is not None:
        n_samples = min(n_samples, max_batches * loader.batch_size)
    del sample_batch, sample_img, sample_emb

    print(f"  Extracting {n_samples} samples, feat_dim={feat_dim}", flush=True)

    # Use memmap for large datasets
    if save_path and n_samples > 2000:
        embs_np = np.memmap(save_path + '_embs.npy', dtype=np.float32,
                            mode='w+', shape=(n_samples, feat_dim))
        labels_np = np.memmap(save_path + '_labels.npy', dtype=np.float32,
                              mode='w+', shape=(n_samples, num_labels))
        use_memmap = True
    else:
        use_memmap = False
        embs_list, labels_list = [], []

    idx = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(loader)} ({idx}/{n_samples})...", flush=True)
        try:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label']
            embs = encoder(images)
            bs = embs.shape[0]
            if use_memmap:
                embs_np[idx:idx+bs] = embs.cpu().float().numpy()
                labels_np[idx:idx+bs] = labels.numpy()
            else:
                embs_list.append(embs.cpu())
                labels_list.append(labels)
            idx += bs
        except Exception as e:
            print(f"  ERROR at batch {i}: {e}, skipping", flush=True)
            continue

    if use_memmap:
        embs_np = np.array(embs_np, dtype=np.float32)  # load from memmap
        labels_np = np.array(labels_np, dtype=np.float32)
    else:
        if not embs_list:
            raise RuntimeError("No embeddings extracted!")
        embs_np = torch.cat(embs_list).float().numpy()
        labels_np = torch.cat(labels_list).float().numpy()

    print(f"  Done: {embs_np.shape} embeddings extracted", flush=True)
    return embs_np, labels_np


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Save args
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
    )

    # -------------------------------------------------------------------------
    # 3. Extract train embeddings
    # -------------------------------------------------------------------------
    print("\n=== Extracting train embeddings ===")
    train_embs, train_labels = extract_all_embeddings(encoder, train_loader, device,
                                                       save_path=os.path.join(args.output_dir, 'train_cache'),
                                                       max_batches=args.max_train_batches)
    print(f"Train embeddings: {train_embs.shape}  Labels: {train_labels.shape}")

    feat_dim = train_embs.shape[1]
    num_classes = train_labels.shape[1]

    # -------------------------------------------------------------------------
    # 4. Train (or load) linear probe
    # -------------------------------------------------------------------------
    probe = LinearProbe(
        feat_dim=feat_dim, num_classes=num_classes,
        lr=args.probe_lr, epochs=args.probe_epochs,
        weight_decay=args.probe_wd, device=device,
    )

    probe_save_path = os.path.join(args.output_dir, 'probe.pth')

    if args.probe_path is not None:
        print(f"\n=== Loading pre-trained probe from {args.probe_path} ===")
        probe.load(args.probe_path)
    else:
        print("\n=== Training linear probe ===")
        probe.fit(train_embs, train_labels)
        probe.save(probe_save_path)

    # -------------------------------------------------------------------------
    # 5. Evaluate robustness on test set
    # -------------------------------------------------------------------------
    print("\n=== Evaluating robustness on test set ===")
    results = evaluate_robustness(
        encoder=encoder,
        probe=probe,
        loader=test_loader,
        device=device,
        perturbation_types=args.pert_types,
        max_batches=args.max_eval_batches,
    )

    # -------------------------------------------------------------------------
    # 6. Save results
    # -------------------------------------------------------------------------
    print("\n=== Saving results ===")
    save_results_csv(results, args.output_dir)
    plot_results(results, args.output_dir)

    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
