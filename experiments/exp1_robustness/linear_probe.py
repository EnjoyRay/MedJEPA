"""
Linear probe trained on frozen encoder features.

Trains a logistic regression (or small MLP) head on top of fixed embeddings
extracted from a pre-trained encoder.

Usage (from run_exp1.py):
    probe = LinearProbe(feat_dim=768, num_classes=15)
    probe.fit(train_embeddings, train_labels)
    scores = probe.predict_proba(test_embeddings)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional


class LinearProbe:
    """
    Multi-label linear probe (sigmoid output, BCE loss).

    Wraps a small nn.Linear trained with Adam for a fixed number of epochs.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device or torch.device('cpu')

        self.head = nn.Linear(feat_dim, num_classes).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearProbe':
        """
        Train the linear head.

        Args:
            X: (N, D) float32 embeddings
            y: (N, C) float32 binary labels
        """
        X_t = torch.from_numpy(X).float().to(self.device)
        y_t = torch.from_numpy(y).float().to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        self.head.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.head(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            if (epoch + 1) % 10 == 0:
                print(f"  [Probe] epoch {epoch+1}/{self.epochs}  loss={total_loss/len(dataset):.4f}")

        self.head.eval()
        return self

    @torch.no_grad()
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """
        Predict raw class logits.

        Args:
            X: (N, D) float32 embeddings
        Returns:
            (N, C) float32 logits
        """
        X_t = torch.from_numpy(X).float().to(self.device)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        logits_all = []
        for (xb,) in loader:
            logits_all.append(self.head(xb).cpu().numpy())
        return np.concatenate(logits_all, axis=0)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: (N, D) float32 embeddings
        Returns:
            (N, C) float32 probabilities in [0, 1]
        """
        X_t = torch.from_numpy(X).float().to(self.device)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs = []
        for (xb,) in loader:
            probs.append(torch.sigmoid(self.head(xb)).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def logits_from_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Return raw logits for an already-device tensor, preserving gradients."""
        return self.head(X.to(self.device))

    def save(self, path: str) -> None:
        torch.save(self.head.state_dict(), path)
        print(f"[Probe] Saved to {path}")

    def load(self, path: str) -> 'LinearProbe':
        self.head.load_state_dict(torch.load(path, map_location=self.device))
        self.head.eval()
        return self


class MLPProbe(LinearProbe):
    """Two-layer non-linear probe on frozen encoder features."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            feat_dim=feat_dim,
            num_classes=num_classes,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            device=device,
        )
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        ).to(self.device)

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.head.state_dict(),
                "feat_dim": self.feat_dim,
                "num_classes": self.num_classes,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
            },
            path,
        )
        print(f"[MLPProbe] Saved to {path}")

    def load(self, path: str) -> 'MLPProbe':
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        self.head.load_state_dict(state)
        self.head.eval()
        return self
