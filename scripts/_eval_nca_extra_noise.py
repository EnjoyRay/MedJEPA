"""Evaluate existing NCA adapters on σ=0.20 and σ=0.30 (additional noise levels).

Run on UIC with: python scripts/_eval_nca_extra_noise.py
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.shared.model_wrappers import build_encoder
from experiments.shared.vindr_dataset import build_vindr_loader
from experiments.exp6_noise_consistent_adapter.adapter import ResidualAdapterClassifier
from experiments.shared.metrics import compute_auroc, compute_cosine_drift


def extract_embeddings(encoder, loader, device, noise_sigma=None, max_batches=None):
    embs, labels = [], []
    encoder.eval()
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = batch['image'].to(device, non_blocking=True)
        if noise_sigma is not None and noise_sigma > 0:
            images_01 = (images * 0.5 + 0.5).clamp(0.0, 1.0)
            images_01 = (images_01 + torch.randn_like(images_01) * noise_sigma).clamp(0.0, 1.0)
            images = (images_01 - 0.5) / 0.5
        embs.append(encoder(images).detach().cpu())
        labels.append(batch['label'].cpu())
    return torch.cat(embs).float(), torch.cat(labels).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['ijepa','mae'])
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--adapter_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Build encoder
    encoder = build_encoder(args.model, args.weights, device)
    encoder.eval()

    # Build test loader
    test_loader = build_vindr_loader(args.data_dir, split='test', img_size=224, in_chans=1, batch_size=32, num_workers=4)

    # Load existing adapter
    ckpt = torch.load(args.adapter_path, map_location=device, weights_only=False)
    adapter = ResidualAdapterClassifier(ckpt['feat_dim'], ckpt['num_classes'], dropout=0.1).to(device)
    adapter.load_state_dict(ckpt['model_state'])
    adapter.eval()

    # Extract clean test embeddings
    print('Extracting clean test embeddings...', flush=True)
    clean_embs, labels = extract_embeddings(encoder, test_loader, device, noise_sigma=None)
    labels_np = labels.numpy()

    rows = []
    for sigma in [0.05, 0.10, 0.20, 0.30]:
        print(f'Evaluating sigma={sigma}...', flush=True)
        noisy_embs, _ = extract_embeddings(encoder, test_loader, device, noise_sigma=sigma)

        # Run adapter
        adapter.eval()
        logits_list, z_list = [], []
        dataset = torch.utils.data.TensorDataset(noisy_embs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for (xb,) in loader:
            with torch.no_grad():
                logits, z = adapter(xb.to(device))
            logits_list.append(logits.detach().cpu())
            z_list.append(z.detach().cpu())
        noisy_logits = torch.cat(logits_list).float().numpy()
        noisy_z = torch.cat(z_list).float().numpy()

        # Also get clean adapter outputs
        if sigma == 0.05:
            logits_c, z_c = [], []
            dataset_c = torch.utils.data.TensorDataset(clean_embs)
            loader_c = torch.utils.data.DataLoader(dataset_c, batch_size=args.batch_size, shuffle=False)
            for (xb,) in loader_c:
                with torch.no_grad():
                    l, z = adapter(xb.to(device))
                logits_c.append(l.detach().cpu())
                z_c.append(z.detach().cpu())
            clean_logits = torch.cat(logits_c).float().numpy()
            clean_z = torch.cat(z_c).float().numpy()
            clean_probs = 1.0 / (1.0 + np.exp(-clean_logits))
            clean_auc = compute_auroc(labels_np, clean_probs)['macro_auroc']

        probs = 1.0 / (1.0 + np.exp(-noisy_logits))
        auc = compute_auroc(labels_np, probs)['macro_auroc']
        adapter_drift = compute_cosine_drift(torch.from_numpy(clean_z), torch.from_numpy(noisy_z))
        raw_drift = compute_cosine_drift(clean_embs, noisy_embs)

        rows.append({
            'condition': f'noise_sigma={sigma:.2f}',
            'macro_auroc': auc,
            'auroc_drop': clean_auc - auc,
            'raw_cosine_drift': raw_drift,
            'adapter_cosine_drift': adapter_drift,
        })
        print(f'  AUROC={auc:.4f}, drop={clean_auc-auc:.4f}, raw_drift={raw_drift:.4f}, adapter_drift={adapter_drift:.4f}', flush=True)

    # Append to existing results
    df_new = pd.DataFrame(rows)
    existing_path = args.output_path
    if os.path.exists(existing_path):
        df_existing = pd.read_csv(existing_path)
        # Remove old noise rows for this ablation
        ablation = ckpt.get('ablation', 'full')
        mask = (df_existing['ablation'] == ablation) & (df_existing['condition'].str.contains('noise'))
        df_existing = df_existing[~mask]
        df_new['ablation'] = ablation
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(existing_path, index=False)
    print(f'Saved to {existing_path}', flush=True)


if __name__ == '__main__':
    main()
