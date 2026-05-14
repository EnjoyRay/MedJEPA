"""
Run class-aligned Exp2 lesion sensitivity.

This entry point mirrors run_exp2.py but writes the redesigned Exp2b outputs:
one row per positive (image_id, class_name) pair with matching bbox annotation.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.model_wrappers import build_encoder
from shared.vindr_dataset import build_vindr_loader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp1_robustness'))
from linear_probe import LinearProbe

from evaluate_class_aligned import (
    evaluate_class_aligned_sensitivity,
    save_class_aligned_results,
)
from run_exp2 import extract_all_embeddings


def get_args():
    parser = argparse.ArgumentParser(description='Exp2b: class-aligned lesion sensitivity')
    parser.add_argument('--model', required=True, choices=['ijepa', 'mae'])
    parser.add_argument('--weights', required=True)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--in_chans', default=1, type=int)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--probe_path', default=None)
    parser.add_argument('--probe_epochs', default=50, type=int)
    parser.add_argument('--probe_lr', default=1e-3, type=float)
    parser.add_argument('--probe_wd', default=1e-4, type=float)
    parser.add_argument('--num_controls', default=5, type=int)
    parser.add_argument('--control_strategy', default='matched_random',
                        choices=['matched_random', 'strict_random', 'mixed', 'random', 'mirror', 'opposite'])
    parser.add_argument('--fill_mode', default='neutral',
                        choices=['neutral', 'constant', 'image_mean'])
    parser.add_argument('--max_samples', default=None, type=int,
                        help='Optional dry-run limit over class-aligned rows')
    parser.add_argument('--encoder_batch_size', default=16, type=int,
                        help='Forward-pass chunk size for original/lesion/control tensors')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'args_class_aligned.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"\n=== Loading {args.model.upper()} encoder ===")
    encoder = build_encoder(args.model, args.weights, device)
    encoder.eval()

    print("\n=== Building data loaders ===")
    train_loader = build_vindr_loader(
        args.data_dir,
        split='train',
        img_size=args.img_size,
        in_chans=args.in_chans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = build_vindr_loader(
        args.data_dir,
        split='test',
        img_size=args.img_size,
        in_chans=args.in_chans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    if args.probe_path is not None:
        print(f"\n=== Loading pre-trained probe from {args.probe_path} ===")
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            sample_emb = encoder(sample_batch['image'][:1].to(device))
        probe = LinearProbe(
            feat_dim=sample_emb.shape[-1],
            num_classes=sample_batch['label'].shape[-1],
            device=device,
        )
        probe.load(args.probe_path)
    else:
        print("\n=== Training linear probe ===")
        train_embs, train_labels = extract_all_embeddings(encoder, train_loader, device)
        probe = LinearProbe(
            feat_dim=train_embs.shape[1],
            num_classes=train_labels.shape[1],
            lr=args.probe_lr,
            epochs=args.probe_epochs,
            weight_decay=args.probe_wd,
            device=device,
        )
        probe.fit(train_embs, train_labels)
        probe.save(os.path.join(args.output_dir, 'probe.pth'))

    print("\n=== Evaluating class-aligned lesion sensitivity ===")
    results = evaluate_class_aligned_sensitivity(
        encoder=encoder,
        probe=probe,
        loader=test_loader,
        device=device,
        num_controls=args.num_controls,
        control_strategy=args.control_strategy,
        fill_mode=args.fill_mode,
        seed=args.seed,
        max_samples=args.max_samples,
        encoder_batch_size=args.encoder_batch_size,
    )

    print("\n=== Saving class-aligned results ===")
    save_class_aligned_results(results, args.output_dir, seed=args.seed)
    print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
