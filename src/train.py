"""1.6 Training. Run end-to-end: discover data -> cache cochleagrams + features
-> fit PCA -> train MLP -> save checkpoint.

After this finishes, run inference.py on a held-out clip to produce a .wav.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLIP_FRAMES,
    EPOCHS,
    LR,
    PCA_DIM,
    PER_FRAME_FEAT_DIM,
)
from dataset import ClipIndex, build_datasets
from model import PCAState, V1MLP, fit_pca
from visual_features import _build_resnet18, features_for_clip, precompute_all


class FeatureCochDataset(Dataset):
    """Wraps cached visual features + cached cochleagrams + PCA targets."""

    def __init__(
        self,
        clips: list[ClipIndex],
        full_coch_lookup,
        pca: PCAState,
        envelope_sr: int,
        clip_duration_s: float,
        onset_frac: float,
        device_for_features: torch.device,
        feature_model,
    ):
        from dataset import slice_cochleagram_for_clip

        self.clips = clips
        self.pca = pca
        self.feats: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        for c in tqdm(clips, desc="building train tensors"):
            feats = features_for_clip(c, feature_model, device_for_features)
            full = full_coch_lookup(c.entry)
            coch = slice_cochleagram_for_clip(
                full, c.onset_time, envelope_sr, clip_duration_s, onset_frac,
            )
            target_z = pca.transform(coch.reshape(1, -1))[0]
            self.feats.append(feats.astype(np.float32))
            self.targets.append(target_z.astype(np.float32))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return torch.from_numpy(self.feats[idx]), torch.from_numpy(self.targets[idx])


def _gather_full_cochleagrams(clips: list[ClipIndex]):
    """Returns (lookup_fn, list_of_clip_cochleagrams) for PCA fitting."""
    from dataset import cochleagram_for_video, slice_cochleagram_for_clip
    from config import CLIP_DURATION_S, ENVELOPE_SR, ONSET_FRAME_INDEX

    cache: dict[str, np.ndarray] = {}

    def lookup(entry):
        if entry.video_id not in cache:
            cache[entry.video_id] = cochleagram_for_video(entry)
        return cache[entry.video_id]

    onset_frac = ONSET_FRAME_INDEX / CLIP_FRAMES
    clip_cochs = []
    for c in tqdm(clips, desc="caching cochleagrams"):
        full = lookup(c.entry)
        clip_cochs.append(
            slice_cochleagram_for_clip(full, c.onset_time, ENVELOPE_SR, CLIP_DURATION_S, onset_frac)
        )
    return lookup, np.stack(clip_cochs, axis=0)


def main(epochs: int = EPOCHS, lr: float = LR, batch_size: int = BATCH_SIZE):
    from config import CLIP_DURATION_S, ENVELOPE_SR, ONSET_FRAME_INDEX

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    _, _, train_clips, test_clips = build_datasets()
    print(f"{len(train_clips)} train clips, {len(test_clips)} test clips")

    # Cache cochleagrams up-front; collect train clip cochleagrams for PCA.
    lookup, train_cochs = _gather_full_cochleagrams(train_clips)
    pca = fit_pca(train_cochs, k=PCA_DIM)
    print(
        f"PCA fit: kept {PCA_DIM} components, "
        f"explained variance ratio = "
        f"{(pca.explained_variance / pca.explained_variance.sum()).round(3).tolist()}"
    )

    # Cache visual features.
    precompute_all(train_clips + test_clips, device=device)
    feature_model = _build_resnet18(device)

    train_set = FeatureCochDataset(
        train_clips, lookup, pca, ENVELOPE_SR, CLIP_DURATION_S,
        ONSET_FRAME_INDEX / CLIP_FRAMES, device, feature_model,
    )
    test_set = FeatureCochDataset(
        test_clips, lookup, pca, ENVELOPE_SR, CLIP_DURATION_S,
        ONSET_FRAME_INDEX / CLIP_FRAMES, device, feature_model,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = V1MLP(n_frames=CLIP_FRAMES, feat_dim=PER_FRAME_FEAT_DIM, out_dim=PCA_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []
    best_val = float("inf")
    ckpt_path = CHECKPOINT_DIR / "v1.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for feats, target in train_loader:
            feats = feats.to(device)
            target = target.to(device)
            pred = model(feats)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for feats, target in test_loader:
                feats = feats.to(device)
                target = target.to(device)
                val_losses.append(loss_fn(model(feats), target).item())

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"epoch {epoch:02d}  train {tr:.4f}  val {va:.4f}")

        if va < best_val:
            best_val = va
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "pca": {
                        "mean": pca.mean,
                        "components": pca.components,
                        "explained_variance": pca.explained_variance,
                        "target_shape": pca.target_shape,
                    },
                    "config": {
                        "n_frames": CLIP_FRAMES,
                        "feat_dim": PER_FRAME_FEAT_DIM,
                        "pca_dim": PCA_DIM,
                    },
                },
                ckpt_path,
            )

    with open(CHECKPOINT_DIR / "v1_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    main(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
