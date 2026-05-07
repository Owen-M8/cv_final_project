"""Quantitative eval on the Greatest Hits held-out test set.

Compares the three-stream model against V1 (if a V1 checkpoint exists) and
two baselines (random training cochleagram, mean training cochleagram). All
four scored on the same 7175 held-out clips.

Reuses the metric helpers from eval_zero_shot.py so the scoring is exactly
comparable to the cross-dataset eval — same loudness MAE, same spectral
centroid MSE + Pearson r definitions.

Usage:
    python src/eval_greatest_hits.py \\
        --three-stream-ckpt checkpoints/three_stream_app-m3d-flow.pt \\
        --v1-ckpt checkpoints/v1.pt \\
        --max-clips 1000 \\
        --n-samples 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cache_visual_features import _pick_device
from cochleagram import cochleagram_to_waveform
from config import (
    CHECKPOINT_DIR,
    CLIP_DURATION_S,
    CLIP_FRAMES,
    ENVELOPE_SR,
    ONSET_FRAME_INDEX,
    OUTPUT_DIR,
    PCA_DIM,
    TARGET_SR,
)
from dataset import (
    ThreeStreamGHDataset,
    build_clip_index,
    discover_videos,
    load_clip_index,
    CLIP_INDEX_PATH,
)
from eval_zero_shot import (
    _gather_training_cochleagrams,
    aggregate_metrics,
    baseline_mean_predictions,
    baseline_random_predictions,
)
from model import PCAState, V1MLP
from streams import FrozenStreams, compute_frozen_features
from three_stream_model import ModelConfig, PCAHead, ThreeStreamV2A
from visual_features import _feat_cache_path


def _load_three_stream(checkpoint: Path, device: torch.device):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_cfg"])
    head = PCAHead(d_model=cfg.d_model, out_dim=PCA_DIM)
    model = ThreeStreamV2A(head, streams=tuple(cfg.streams), cfg=cfg).to(device).eval()
    model.load_state_dict(blob["model_state"])
    p = blob["pca"]
    pca = PCAState(
        mean=p["mean"], components=p["components"],
        explained_variance=p["explained_variance"],
        target_shape=tuple(p["target_shape"]),
    )
    return model, pca, cfg


def _load_v1(checkpoint: Path, device: torch.device):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = V1MLP(
        n_frames=cfg["n_frames"],
        feat_dim=cfg["feat_dim"],
        out_dim=cfg["pca_dim"],
    ).to(device).eval()
    model.load_state_dict(blob["model_state"])
    p = blob["pca"]
    pca = PCAState(
        mean=p["mean"], components=p["components"],
        explained_variance=p["explained_variance"],
        target_shape=tuple(p["target_shape"]),
    )
    return model, pca


@torch.no_grad()
def _predict_three_stream(test_clips, model, pca, frozen, device, batch_size=64, num_workers=2):
    """Batched inference over the three-stream test loader. Returns (preds, truths)
    aligned with test_clips order."""
    onset_frac = ONSET_FRAME_INDEX / CLIP_FRAMES
    ds = ThreeStreamGHDataset(
        test_clips, ENVELOPE_SR, CLIP_DURATION_S, onset_frac, temporal_jitter=0,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    for batch in tqdm(loader, desc="three-stream", dynamic_ncols=True):
        frames = batch["frames"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True)
        # frames in __getitem__ is uint8 (T, H, W, 3); convert to float (B, T, 3, H, W).
        frames_chw = (frames.float() / 255.0).permute(0, 1, 4, 2, 3).contiguous()
        flow_f = flow.float()
        autocast_ctx = torch.amp.autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)
        with autocast_ctx:
            app, m3d = compute_frozen_features(frames_chw, frozen)
            z = model(appearance=app, motion3d=m3d, flow_field=flow_f)
        z_np = z.float().cpu().numpy()
        for zi in z_np:
            coch_flat = pca.inverse_transform(zi[None])[0]
            preds.append(np.clip(coch_flat.reshape(pca.target_shape), 0.0, None).astype(np.float32))
        for c in batch["cochleagram"]:
            truths.append(c.numpy().astype(np.float32))
    return preds, truths


@torch.no_grad()
def _predict_v1(test_clips, model, pca, device):
    """V1 reads from cached _feat.npy directly — fast, no DataLoader needed."""
    preds = []
    for clip in tqdm(test_clips, desc="v1", dynamic_ncols=True):
        feats = np.load(_feat_cache_path(clip))
        z = model(torch.from_numpy(feats).unsqueeze(0).to(device)).cpu().numpy()[0]
        coch_flat = pca.inverse_transform(z[None])[0]
        preds.append(np.clip(coch_flat.reshape(pca.target_shape), 0.0, None).astype(np.float32))
    return preds


def main(
    three_stream_ckpt: Path,
    v1_ckpt: Path | None,
    max_clips: int | None,
    n_samples: int,
    out_dir: Path,
    batch_size: int = 64,
    num_workers: int = 2,
) -> None:
    device = _pick_device()
    print(f"device: {device}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Test clips ------------------------------------------------------
    if CLIP_INDEX_PATH.exists():
        train_clips, test_clips = load_clip_index()
    else:
        clips = build_clip_index(discover_videos())
        from dataset import video_level_split
        train_clips, test_clips = video_level_split(clips)
    if max_clips is not None:
        test_clips = test_clips[:max_clips]
    print(f"{len(train_clips)} train clips, {len(test_clips)} test clips", flush=True)

    # ---- Three-stream model ---------------------------------------------
    print(f"loading three-stream checkpoint {three_stream_ckpt.name}...", flush=True)
    ts_model, ts_pca, ts_cfg = _load_three_stream(three_stream_ckpt, device)
    print(f"  streams={ts_cfg.streams}", flush=True)
    print("building frozen backbones...", flush=True)
    frozen = FrozenStreams.build(device)
    ts_preds, truths = _predict_three_stream(
        test_clips, ts_model, ts_pca, frozen, device,
        batch_size=batch_size, num_workers=num_workers,
    )

    all_preds = {"three_stream": ts_preds}

    # ---- V1 (optional) ---------------------------------------------------
    if v1_ckpt is not None and v1_ckpt.exists():
        print(f"loading V1 checkpoint {v1_ckpt.name}...", flush=True)
        v1_model, v1_pca = _load_v1(v1_ckpt, device)
        v1_preds = _predict_v1(test_clips, v1_model, v1_pca, device)
        all_preds["v1"] = v1_preds
    else:
        print("skipping V1 (checkpoint not provided or not found)", flush=True)

    # ---- Baselines -------------------------------------------------------
    print("building baselines from training cochleagrams...", flush=True)
    pool = _gather_training_cochleagrams(train_clips)
    n = len(test_clips)
    all_preds["baseline_random"] = baseline_random_predictions(pool, n)
    all_preds["baseline_mean"] = baseline_mean_predictions(pool, n)

    # ---- Score -----------------------------------------------------------
    metrics = {name: aggregate_metrics(p, truths) for name, p in all_preds.items()}

    # ---- Print + save ----------------------------------------------------
    print("\n=== Greatest Hits test set ===")
    header = f"{'method':<18}{'loudness MAE':>14}{'cent. MSE':>14}{'cent. r':>10}{'n':>6}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(
            f"{name:<18}{m['loudness_mae']:>14.4f}{m['centroid_mse']:>14.4f}"
            f"{m['centroid_pearson_r']:>10.3f}{m['n_clips']:>6d}"
        )

    out_metrics = out_dir / "metrics.json"
    payload = {
        "checkpoints": {
            "three_stream": str(three_stream_ckpt),
            "v1": str(v1_ckpt) if v1_ckpt else None,
        },
        "model_cfg": {
            "three_stream_streams": list(ts_cfg.streams),
        },
        "n_test_clips": n,
        "metrics": metrics,
    }
    out_metrics.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_metrics}")

    # ---- Listening samples ----------------------------------------------
    if n_samples > 0:
        samples_dir = out_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=min(n_samples, n), replace=False)
        for k in idx:
            clip = test_clips[k]
            tag = f"{clip.entry.video_id}_t{clip.onset_time:.2f}_{clip.material or 'unk'}"
            real_wav = cochleagram_to_waveform(truths[k], sr=TARGET_SR, envelope_sr=ENVELOPE_SR)
            sf.write(str(samples_dir / f"{tag}_real.wav"), real_wav, TARGET_SR)
            for name, preds in all_preds.items():
                pred_wav = cochleagram_to_waveform(preds[k], sr=TARGET_SR, envelope_sr=ENVELOPE_SR)
                sf.write(str(samples_dir / f"{tag}_{name}.wav"), pred_wav, TARGET_SR)
        print(f"wrote {len(idx)} clip × {len(all_preds) + 1} wav files to {samples_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--three-stream-ckpt", type=Path,
        default=CHECKPOINT_DIR / "three_stream_app-m3d-flow.pt",
    )
    parser.add_argument(
        "--v1-ckpt", type=Path, default=CHECKPOINT_DIR / "v1.pt",
        help="Optional. Skipped if file not present.",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "eval_greatest_hits")
    args = parser.parse_args()
    v1_ckpt = args.v1_ckpt if args.v1_ckpt.exists() else None
    main(
        three_stream_ckpt=args.three_stream_ckpt,
        v1_ckpt=v1_ckpt,
        max_clips=args.max_clips,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
