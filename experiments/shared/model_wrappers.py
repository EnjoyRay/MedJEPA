"""
Unified encoder wrappers for I-JEPA and MAE.

Both models expose the same interface:
    encoder = build_encoder(model_type, weights_path, device)
    features = encoder(images)   # (B, D) float32 tensor

Usage:
    from shared.model_wrappers import build_encoder
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MAE encoder wrapper
# ---------------------------------------------------------------------------

def _build_mae_encoder(weights_path: str, device: torch.device) -> nn.Module:
    """Load a MAE ViT encoder from a checkpoint produced by mae/main_pretrain.py."""
    import sys, os
    # Allow importing from the sibling mae/ directory
    mae_root = os.path.join(os.path.dirname(__file__), '..', '..', 'mae')
    if mae_root not in sys.path:
        sys.path.insert(0, mae_root)

    import models_mae  # type: ignore

    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model', checkpoint)
    patch_weight = state.get('patch_embed.proj.weight')
    in_chans = int(patch_weight.shape[1]) if patch_weight is not None else 3
    embed_dim = int(patch_weight.shape[0]) if patch_weight is not None else 768
    patch_size = int(patch_weight.shape[-1]) if patch_weight is not None else 16

    if embed_dim == 1280 and patch_size == 14:
        model = models_mae.mae_vit_huge_patch14(in_chans=in_chans)
        arch = "mae_vit_huge_patch14"
    elif embed_dim == 1024 and patch_size == 16:
        model = models_mae.mae_vit_large_patch16(in_chans=in_chans)
        arch = "mae_vit_large_patch16"
    elif embed_dim == 768 and patch_size == 16:
        model = models_mae.mae_vit_base_patch16(in_chans=in_chans)
        arch = "mae_vit_base_patch16"
    else:
        raise ValueError(
            f"Cannot infer MAE architecture from patch_embed.proj.weight: "
            f"embed_dim={embed_dim}, patch_size={patch_size}, in_chans={in_chans}"
        )

    # MAE checkpoints may include decoder weights; load with strict=False
    msg = model.load_state_dict(state, strict=False)
    print(f"[MAE] Loaded weights from {weights_path}: {msg}")
    print(f"[MAE] Detected architecture: {arch}, input channels: {in_chans}")

    # We only need the encoder part
    encoder = _MAEEncoderOnly(model)
    encoder.eval().to(device)
    return encoder


class _MAEEncoderOnly(nn.Module):
    """Wraps a MaskedAutoencoderViT and exposes only its encoder forward pass."""

    def __init__(self, mae: nn.Module):
        super().__init__()
        self.mae = mae

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        expected_chans = self.mae.patch_embed.proj.weight.shape[1]
        if x.shape[1] != expected_chans:
            if x.shape[1] == 1 and expected_chans == 3:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] == 3 and expected_chans == 1:
                x = x.mean(dim=1, keepdim=True)
            else:
                raise ValueError(
                    f"MAE checkpoint expects {expected_chans} input channels, got {x.shape[1]}"
                )
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens with shape (B, N, D), CLS token excluded."""
        x = self._prepare_input(x)
        latent, _, _ = self.mae.forward_encoder(x, mask_ratio=0.0)
        return latent[:, 1:, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) or (B, 3, H, W) tensor, normalized
        Returns:
            (B, D) mean-pooled patch embeddings (CLS excluded)
        """
        return self.forward_tokens(x).mean(dim=1)


# ---------------------------------------------------------------------------
# I-JEPA encoder wrapper
# ---------------------------------------------------------------------------

def _build_ijepa_encoder(weights_path: str, device: torch.device) -> nn.Module:
    """Load an I-JEPA context encoder from a checkpoint.

    Supports the official facebookresearch/ijepa checkpoint format.
    The checkpoint is expected to contain checkpoint['encoder'] with the
    context encoder state dict. Auto-detects vit_base vs vit_large from
    weight shapes (embed_dim: 768=base, 1024=large).
    """
    import sys, os
    ijepa_root = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                               'zhaoyi', 'medical-i-jepa')
    if ijepa_root not in sys.path:
        sys.path.insert(0, ijepa_root)

    from src.models.vision_transformer import vit_base, vit_large, vit_huge

    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state = _extract_ijepa_state(checkpoint, prefix='encoder')

    # Auto-detect model size from embed_dim in pos_embed
    first_key = [k for k in state if 'pos_embed' in k][0]
    embed_dim = state[first_key].shape[-1]

    patch_size = 14  # medical-i-jepa uses patch_size=14

    if embed_dim == 1280:
        model = vit_huge(patch_size=patch_size)
        print(f"[I-JEPA] Detected ViT-Huge/{patch_size} (dim={embed_dim})")
    elif embed_dim == 1024:
        model = vit_large(patch_size=patch_size)
        print(f"[I-JEPA] Detected ViT-Large/{patch_size} (dim={embed_dim})")
    elif embed_dim == 768:
        model = vit_base(patch_size=patch_size)
        print(f"[I-JEPA] Detected ViT-Base/{patch_size} (dim={embed_dim})")
    else:
        raise ValueError(f"Unexpected embed_dim={embed_dim}, cannot determine architecture")

    msg = model.load_state_dict(state, strict=False)
    print(f"[I-JEPA] Loaded weights from {weights_path}: {msg}")
    model.eval().to(device)
    return _IJEPAEncoderWrapper(model)


def _extract_ijepa_state(checkpoint: dict, prefix: str = 'encoder') -> dict:
    """Extract encoder state dict from various I-JEPA checkpoint formats."""
    if prefix in checkpoint:
        state = checkpoint[prefix]
    elif 'target_encoder' in checkpoint:
        state = checkpoint['target_encoder']
    elif 'model' in checkpoint:
        state = checkpoint['model']
    else:
        state = checkpoint

    # Strip common key prefixes (e.g. 'module.', 'encoder.')
    cleaned = {}
    for k, v in state.items():
        for pfx in ('module.', 'encoder.', 'backbone.'):
            if k.startswith(pfx):
                k = k[len(pfx):]
                break
        cleaned[k] = v
    return cleaned


class _IJEPAEncoderWrapper(nn.Module):
    """Wraps the official I-JEPA ViT and exposes mean-pooled patch embeddings.
    Handles single-channel input by replicating to 3 channels (I-JEPA ViT expects RGB)."""

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # If grayscale (1 channel), replicate to 3 channels for I-JEPA ViT
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Official ijepa vit returns patch tokens; shape (B, N, D)
        tokens = self.vit(x)
        if tokens.dim() != 3:
            raise ValueError(f"I-JEPA encoder returned non-token tensor with shape {tuple(tokens.shape)}")
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        if tokens.dim() == 3:
            return tokens.mean(dim=1)   # (B, D)
        return tokens                   # already (B, D)


class _TimmViTEncoder(nn.Module):
    """Wraps a timm ViT (num_classes=0) as a feature extractor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm with num_classes=0 returns (B, D) directly
        return self.model(x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_encoder(model_type: str, weights_path: str, device: torch.device) -> nn.Module:
    """
    Build a frozen encoder.

    Args:
        model_type: 'ijepa' or 'mae'
        weights_path: path to pretrained checkpoint
        device: torch device

    Returns:
        nn.Module with signature forward(x: Tensor) -> Tensor (B, D)
    """
    model_type = model_type.lower()
    if model_type == 'mae':
        return _build_mae_encoder(weights_path, device)
    elif model_type in ('ijepa', 'i-jepa'):
        return _build_ijepa_encoder(weights_path, device)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'ijepa' or 'mae'.")
