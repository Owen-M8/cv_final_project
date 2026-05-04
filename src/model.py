"""1.5 V1 model: simple MLP regressor on flattened per-frame features.

Predicts a PCA-reduced cochleagram (10-d). PCA is fit on training-set
cochleagrams (flattened to channels*time) and stored alongside the checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from config import CLIP_FRAMES, HIDDEN1, HIDDEN2, PCA_DIM, PER_FRAME_FEAT_DIM


@dataclass
class PCAState:
    mean: np.ndarray            # (D,)
    components: np.ndarray      # (PCA_DIM, D)
    explained_variance: np.ndarray  # (PCA_DIM,)
    target_shape: tuple[int, int]   # (n_channels, n_time) for un-flattening

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components + self.mean


def fit_pca(cochleagrams: np.ndarray, k: int = PCA_DIM) -> PCAState:
    """cochleagrams: (N, C, T). Returns a PCAState fit on flattened (N, C*T)."""
    n, c, t = cochleagrams.shape
    X = cochleagrams.reshape(n, c * t).astype(np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD-based PCA, keep top-k components.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k].astype(np.float32)
    explained = (S[:k] ** 2) / max(1, n - 1)
    return PCAState(
        mean=mean.astype(np.float32),
        components=components,
        explained_variance=explained.astype(np.float32),
        target_shape=(c, t),
    )


class V1MLP(nn.Module):
    """visual_features (T, F) -> flatten -> MLP -> PCA-coeff prediction."""

    def __init__(
        self,
        n_frames: int = CLIP_FRAMES,
        feat_dim: int = PER_FRAME_FEAT_DIM,
        hidden1: int = HIDDEN1,
        hidden2: int = HIDDEN2,
        out_dim: int = PCA_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(n_frames * feat_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, F)
        return self.net(self.flatten(feats))
