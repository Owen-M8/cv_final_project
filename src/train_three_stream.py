"""Three-stream V2A training.

Pipeline per epoch:
    1. Sample a clip (with optional ±1-frame temporal jitter from the dataset).
    2. Apply pixel-space augmentation: hflip (also flips flow_x sign),
       per-clip color jitter.
    3. Run frozen ResNet50 + R(2+1)D-18 once per batch on the GPU to produce
       (app, m3d) features.
    4. Forward the trainable ThreeStreamV2A: learnable flow encoder + 2-layer
       Transformer fusion + windowed mean-pool readout around the impact
       frame + PCA head.
    5. MSE loss against the PCA-reduced cochleagram target.

After this finishes, run inference (TODO `--variant three_stream`) on a
held-out clip to produce a .wav.

Stream selection for ablations: `--streams app,m3d,flow` (default), or any
non-empty subset like `--streams app,flow` to run the no-3D-CNN ablation.
Each ablation row is a separate training run; checkpoints are named after
the stream tuple by default.
"""
from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cache_visual_features import _pick_device
from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    LR,
    PCA_DIM,
    TS_D_MODEL,
)
from dataset import build_three_stream_datasets
from model import fit_pca
from streams import FrozenStreams, compute_frozen_features
from three_stream_model import ModelConfig, PCAHead, ThreeStreamV2A
from train import _gather_full_cochleagrams


# ---------------------------------------------------------------------------
# Per-sample augmentation. Applied inside the wrapping Dataset so DataLoader
# workers parallelise it and each sample in a batch gets independent params.
# ---------------------------------------------------------------------------

def _hflip(frames_uint8: np.ndarray, flow_fp16: np.ndarray, prob: float = 0.5):
    """Horizontal flip. Flips frames spatially; flips flow spatially AND
    negates flow_x because horizontal motion direction reverses under hflip.
    Easy to get wrong — keep this in one place."""
    if np.random.random() >= prob:
        return frames_uint8, flow_fp16
    frames_uint8 = frames_uint8[:, :, ::-1, :].copy()
    flow_fp16 = flow_fp16[:, :, :, ::-1].copy()
    flow_fp16[:, 0] = -flow_fp16[:, 0]
    return frames_uint8, flow_fp16


def _color_jitter(frames_float: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Per-clip brightness + contrast jitter. Same params across all frames in
    the clip so we don't break temporal coherence (which the 3D and flow
    streams encode and would react to as motion)."""
    if strength <= 0:
        return frames_float
    s = strength
    brightness = 1.0 + np.random.uniform(-s, s)
    contrast = 1.0 + np.random.uniform(-s, s)
    out = frames_float * brightness
    mean = out.mean()
    out = (out - mean) * contrast + mean
    return np.clip(out, 0.0, 1.0)


class _TransformedDataset(Dataset):
    """Wraps a V2AClipDataset; applies augmentation + PCA target conversion."""

    def __init__(self, base, pca, augment: bool, color_jitter_strength: float = 0.2):
        self.base = base
        self.pca = pca
        self.augment = augment
        self.color_jitter_strength = color_jitter_strength

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        frames = sample["frames"]        # uint8 (T, H, W, 3) RGB
        flow = sample["flow"]            # fp16 (T-1, 2, H_f, W_f)
        coch = sample["cochleagram"]     # float32 (C, T_audio)

        if self.augment:
            frames, flow = _hflip(frames, flow)

        frames = frames.astype(np.float32) / 255.0
        if self.augment:
            frames = _color_jitter(frames, self.color_jitter_strength)
        # (T, H, W, 3) -> (T, 3, H, W) for compute_frozen_features
        frames = np.ascontiguousarray(np.transpose(frames, (0, 3, 1, 2)))

        target = self.pca.transform(coch.reshape(1, -1))[0].astype(np.float32)

        return {
            "frames": torch.from_numpy(frames),                      # (T, 3, H, W) float
            "flow": torch.from_numpy(flow.astype(np.float32)),       # (T-1, 2, H_f, W_f) float
            "target": torch.from_numpy(target),                      # (PCA_DIM,) float
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _epoch(
    model: ThreeStreamV2A,
    frozen: FrozenStreams,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    opt: torch.optim.Optimizer | None,
    epoch_num: int,
    n_epochs: int,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """One training or eval epoch. Pass opt=None for eval.

    Mixed precision: when device is CUDA, frozen forward + trainable forward run
    inside torch.amp.autocast('cuda'). On train, gradients are scaled via
    `scaler` to avoid fp16 underflow (caller passes one in). On eval, autocast
    alone is enough — no scaler needed without a backward pass.
    """
    is_train = opt is not None
    model.train(is_train)
    losses = []
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    use_amp = device.type == "cuda"
    autocast_ctx = (
        torch.amp.autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)
    )
    desc = f"ep {epoch_num:02d}/{n_epochs:02d} {'train' if is_train else ' val '}"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    with grad_ctx:
        for batch in pbar:
            frames = batch["frames"].to(device, non_blocking=True)
            flow = batch["flow"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            with autocast_ctx:
                with torch.no_grad():
                    app, m3d = compute_frozen_features(frames, frozen)
                pred = model(appearance=app, motion3d=m3d, flow_field=flow)
                loss = loss_fn(pred, target)

            if is_train:
                opt.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            losses.append(loss.item())
            recent = losses[-50:]
            pbar.set_postfix(loss=f"{float(np.mean(recent)):.4f}")
    return float(np.mean(losses)) if losses else float("nan")


def main(
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    streams: tuple[str, ...] = ("app", "m3d", "flow"),
    temporal_jitter: int = 1,
    color_jitter: float = 0.2,
    num_workers: int = 2,
    weight_decay: float = 1e-4,
    ckpt_name: str | None = None,
    eval_every: int = 1,
) -> None:
    device = _pick_device()  # CUDA -> MPS -> CPU
    print(f"device: {device}", flush=True)
    print(f"streams: {streams}  temporal_jitter: ±{temporal_jitter}  color_jitter: {color_jitter}", flush=True)

    train_base, test_base, train_clips, test_clips = build_three_stream_datasets(
        train_temporal_jitter=temporal_jitter,
    )
    print(f"{len(train_clips)} train clips, {len(test_clips)} test clips", flush=True)

    # PCA on training-set cochleagrams (cached as a side effect so the dataset
    # can serve audio targets quickly during training).
    _, train_cochs = _gather_full_cochleagrams(train_clips)
    pca = fit_pca(train_cochs, k=PCA_DIM)
    print(
        f"PCA fit: kept {PCA_DIM} components, "
        f"explained variance ratio = "
        f"{(pca.explained_variance / pca.explained_variance.sum()).round(3).tolist()}",
        flush=True,
    )

    train_ds = _TransformedDataset(train_base, pca, augment=True, color_jitter_strength=color_jitter)
    test_ds = _TransformedDataset(test_base, pca, augment=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )

    print("building frozen backbones (downloads on first run)...", flush=True)
    frozen = FrozenStreams.build(device)
    print("frozen backbones ready", flush=True)

    model_cfg = ModelConfig(streams=tuple(streams))
    head = PCAHead(d_model=TS_D_MODEL, out_dim=PCA_DIM)
    model = ThreeStreamV2A(head, streams=tuple(streams), cfg=model_cfg).to(device)
    print(
        f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    # GradScaler is only meaningful on CUDA; on MPS/CPU we run plain training.
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    print(f"AMP: {'on (cuda autocast + GradScaler)' if scaler else 'off (non-CUDA device)'}", flush=True)
    print(f"eval cadence: every {eval_every} epoch(s)", flush=True)

    history = []
    best_val = float("inf")
    ckpt_name = ckpt_name or f"three_stream_{'-'.join(streams)}.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    history_path = CHECKPOINT_DIR / (ckpt_path.stem + "_history.json")

    for epoch in range(1, epochs + 1):
        tr = _epoch(model, frozen, train_loader, loss_fn, device, opt,
                    epoch, epochs, scaler=scaler)

        # Skip eval on intermediate epochs unless it's an eval epoch or the
        # final epoch — saves ~25% of wall time per epoch with minimal loss
        # of resolution on the val curve.
        do_eval = (epoch % eval_every == 0) or (epoch == epochs)
        if do_eval:
            va = _epoch(model, frozen, test_loader, loss_fn, device, opt=None,
                        epoch_num=epoch, n_epochs=epochs)
        else:
            va = float("nan")
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if do_eval:
            print(f"epoch {epoch:02d}  train {tr:.4f}  val {va:.4f}", flush=True)
        else:
            print(f"epoch {epoch:02d}  train {tr:.4f}  val   - (skipped)", flush=True)

        if do_eval and va < best_val:
            best_val = va
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": dataclasses.asdict(model_cfg),
                    "pca": {
                        "mean": pca.mean,
                        "components": pca.components,
                        "explained_variance": pca.explained_variance,
                        "target_shape": pca.target_shape,
                    },
                    "training": {
                        "epochs": epochs,
                        "best_val": best_val,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "temporal_jitter": temporal_jitter,
                        "color_jitter": color_jitter,
                        "eval_every": eval_every,
                    },
                },
                ckpt_path,
            )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved checkpoint -> {ckpt_path}", flush=True)
    print(f"saved history    -> {history_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--streams", type=str, default="app,m3d,flow",
        help="Non-empty subset of {app,m3d,flow} (comma-separated). "
        "Use for ablations, e.g. --streams app,flow.",
    )
    parser.add_argument("--temporal-jitter", type=int, default=1)
    parser.add_argument("--color-jitter", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ckpt-name", type=str, default=None)
    parser.add_argument(
        "--eval-every", type=int, default=1,
        help="Run val pass every N epochs (default 1 = every epoch). "
        "Final epoch is always evaluated.",
    )
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        streams=tuple(s.strip() for s in args.streams.split(",") if s.strip()),
        temporal_jitter=args.temporal_jitter,
        color_jitter=args.color_jitter,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        ckpt_name=args.ckpt_name,
        eval_every=args.eval_every,
    )
