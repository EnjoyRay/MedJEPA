"""Export only the I-JEPA encoder state from a full training checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a smaller encoder-only I-JEPA checkpoint.")
    parser.add_argument("--input", required=True, help="Full I-JEPA checkpoint.")
    parser.add_argument("--output", required=True, help="Destination .pth.tar file.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"SKIP existing {out}", file=sys.stderr)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(checkpoint)!r}")
    for key in ("encoder", "target_encoder", "model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            break
    else:
        state = checkpoint
    payload = {
        "encoder": state,
        "source_checkpoint": str(args.input),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }
    tmp = out.with_suffix(out.suffix + ".part")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(f"Wrote {out} with {len(state)} encoder tensors", file=sys.stderr)


if __name__ == "__main__":
    main()
