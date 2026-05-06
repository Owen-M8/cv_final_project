"""Cross-dataset zero-shot eval for the three-stream V2A model.

Runs a Greatest-Hits-trained checkpoint on EPIC-Sounds clips, computes
loudness + spectral-centroid metrics against the real audio, and (optionally)
writes paired predicted/real .wav files for listening.

Two baselines are scored alongside the model so the result table tells you
whether the model is doing anything useful on the new domain:
  - random:  predict a random training-set cochleagram for each clip
  - mean:    predict the dataset's mean cochleagram for every clip

If the model doesn't beat both baselines, it isn't transferring and the
discussion section should say so plainly.

Usage:
    python src/eval_zero_shot.py \
        --checkpoint checkpoints/three_stream_app-m3d-flow.pt \
        --annotations-csv path/to/EPIC_Sounds_validation.csv \
        --videos-dir path/to/EPIC-KITCHENS-100 \
        --max-clips 200 \
        --n-samples 5
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

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
    ClipIndex,
    build_clip_index,
    cochleagram_for_video,
    discover_videos,
    load_clip_index,
    slice_cochleagram_for_clip,
)
from epic_sounds_dataset import build_epic_sounds_dataset
from model import PCAState
from streams import FrozenStreams, compute_frozen_features
from three_stream_model import ModelConfig, PCAHead, ThreeStreamV2A


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def loudness_error(pred: np.ndarray, true: np.ndarray) -> float:
    """|max(L2_norm(pred_subbands)) - max(L2_norm(true_subbands))|.

    L2 norm per subband collapses time; max picks the peak energy across
    subbands. Captures whether the predicted impact is the right loudness
    (independent of when within the window it lands).
    """
    pred_loud = np.max(np.linalg.norm(pred, axis=1))
    true_loud = np.max(np.linalg.norm(true, axis=1))
    return float(abs(pred_loud - true_loud))


def spectral_centroid(coch: np.ndarray) -> float:
    """Center of mass over frequency channels at the impact center.

    Uses a 1-frame slice centered on ONSET_FRAME_INDEX of the (channels, time)
    cochleagram. Returns the channel index (lower = darker / more bassy).
    """
    n_channels, n_time = coch.shape
    onset_audio_idx = int(round(ONSET_FRAME_INDEX / CLIP_FRAMES * n_time))
    onset_audio_idx = min(max(0, onset_audio_idx), n_time - 1)
    column = coch[:, onset_audio_idx]
    total = float(column.sum()) + 1e-8
    weights = np.arange(n_channels, dtype=np.float32)
    return float((weights * column).sum() / total)


def aggregate_metrics(
    pred_cochs: list[np.ndarray],
    true_cochs: list[np.ndarray],
) -> dict[str, float]:
    """Per-clip loudness MAE; cross-clip centroid Pearson r and MSE."""
    loud_errs = [loudness_error(p, t) for p, t in zip(pred_cochs, true_cochs)]
    pred_centroids = np.array([spectral_centroid(p) for p in pred_cochs])
    true_centroids = np.array([spectral_centroid(t) for t in true_cochs])
    centroid_mse = float(((pred_centroids - true_centroids) ** 2).mean())
    if len(pred_centroids) > 1 and pred_centroids.std() > 0 and true_centroids.std() > 0:
        centroid_r = float(np.corrcoef(pred_centroids, true_centroids)[0, 1])
    else:
        centroid_r = float("nan")
    return {
        "loudness_mae": float(np.mean(loud_errs)),
        "loudness_std": float(np.std(loud_errs)),
        "centroid_mse": centroid_mse,
        "centroid_pearson_r": centroid_r,
        "n_clips": len(loud_errs),
    }


# ---------------------------------------------------------------------------
# Baselines (need a pool of real training-set cochleagrams to draw from)
# ---------------------------------------------------------------------------

def _gather_training_cochleagrams(train_clips: list[ClipIndex]) -> np.ndarray:
    """Slice a 0.5s window around each training-clip onset. Used for the
    'random' and 'mean' baselines — both pick from the same pool the model
    saw at train time."""
    onset_frac = ONSET_FRAME_INDEX / CLIP_FRAMES
    cache: dict[str, np.ndarray] = {}
    out = []
    for c in train_clips:
        if c.entry.video_id not in cache:
            cache[c.entry.video_id] = cochleagram_for_video(c.entry)
        full = cache[c.entry.video_id]
        out.append(slice_cochleagram_for_clip(
            full, c.onset_time, ENVELOPE_SR, CLIP_DURATION_S, onset_frac,
        ))
    return np.stack(out, axis=0)


def baseline_random_predictions(pool: np.ndarray, n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(low=0, high=pool.shape[0], size=n)
    return [pool[i] for i in idx]


def baseline_mean_predictions(pool: np.ndarray, n: int) -> list[np.ndarray]:
    mean = pool.mean(axis=0)
    return [mean.copy() for _ in range(n)]


# ---------------------------------------------------------------------------
# Model loading + per-clip prediction
# ---------------------------------------------------------------------------

def _load_three_stream(checkpoint: Path, device: torch.device):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_cfg"])
    streams = tuple(cfg.streams)
    head = PCAHead(d_model=cfg.d_model, out_dim=PCA_DIM)
    model = ThreeStreamV2A(head, streams=streams, cfg=cfg).to(device).eval()
    model.load_state_dict(blob["model_state"])
    p = blob["pca"]
    pca = PCAState(
        mean=p["mean"], components=p["components"],
        explained_variance=p["explained_variance"],
        target_shape=tuple(p["target_shape"]),
    )
    return model, pca, cfg


@torch.no_grad()
def _predict_coch_batch(
    frames_uint8: np.ndarray,    # (T, H, W, 3) uint8
    flow_fp16: np.ndarray,       # (T-1, 2, H_f, W_f) fp16
    model: ThreeStreamV2A,
    frozen: FrozenStreams,
    pca: PCAState,
    device: torch.device,
) -> np.ndarray:
    """Run one clip end-to-end. Returns the predicted cochleagram, clipped at 0."""
    frames_chw = (frames_uint8.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    frames_t = torch.from_numpy(frames_chw).unsqueeze(0).to(device)
    flow_t = torch.from_numpy(flow_fp16.astype(np.float32)).unsqueeze(0).to(device)
    app, m3d = compute_frozen_features(frames_t, frozen)
    z = model(appearance=app, motion3d=m3d, flow_field=flow_t).cpu().numpy()[0]
    coch_flat = pca.inverse_transform(z[None])[0]
    return np.clip(coch_flat.reshape(pca.target_shape).astype(np.float32), 0.0, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    checkpoint: Path,
    annotations_csv: Path,
    videos_dir: Path,
    classes: list[str] | None = None,
    max_clips: int | None = None,
    n_samples: int = 5,
    out_dir: Path = OUTPUT_DIR / "eval_zero_shot",
) -> None:
    from cache_visual_features import _pick_device
    device = _pick_device()  # CUDA -> MPS -> CPU
    print(f"device: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"

    # --- Load model ---------------------------------------------------------
    model, pca, cfg = _load_three_stream(checkpoint, device)
    print(f"loaded {checkpoint.name}  streams={cfg.streams}")
    frozen = FrozenStreams.build(device)

    # --- Build EPIC eval set ------------------------------------------------
    eval_ds = build_epic_sounds_dataset(
        annotations_csv, videos_dir, classes=classes, max_clips=max_clips,
    )
    if len(eval_ds) == 0:
        print("no eval clips matched filter; nothing to do")
        return

    # --- Build baseline pool from the SAME training clips the model saw -----
    from config import CACHE_DIR  # noqa: F401  (CLIP_INDEX_PATH lives under CACHE_DIR)
    from dataset import CLIP_INDEX_PATH

    if CLIP_INDEX_PATH.exists():
        train_clips, _ = load_clip_index()
    else:
        train_clips = build_clip_index(discover_videos())
    pool = _gather_training_cochleagrams(train_clips)
    print(f"baseline pool: {pool.shape[0]} training cochleagrams")

    # --- Predict + record per-clip cochleagrams -----------------------------
    pred_cochs: list[np.ndarray] = []
    true_cochs: list[np.ndarray] = []
    sample_metadata: list[dict] = []
    for i in range(len(eval_ds)):
        sample = eval_ds[i]
        pred = _predict_coch_batch(
            sample["frames"], sample["flow"], model, frozen, pca, device,
        )
        true = sample["cochleagram"]
        # Pad/clip the real cochleagram to the model's predicted shape if a
        # boundary clip came back short.
        if true.shape != pred.shape:
            tc, tt = true.shape
            pc, pt = pred.shape
            t = min(tt, pt)
            true = true[:, :t]
            pred = pred[:, :t]
        pred_cochs.append(pred)
        true_cochs.append(true)
        sample_metadata.append({
            "video_id": sample["video_id"],
            "onset_time": sample["onset_time"],
            "class": sample["material"],
        })
        if (i + 1) % 25 == 0 or i == len(eval_ds) - 1:
            print(f"  predicted {i + 1}/{len(eval_ds)}", flush=True)

    # --- Score model + baselines -------------------------------------------
    n = len(pred_cochs)
    metrics = {
        "model": aggregate_metrics(pred_cochs, true_cochs),
        "baseline_random": aggregate_metrics(
            baseline_random_predictions(pool, n), true_cochs,
        ),
        "baseline_mean": aggregate_metrics(
            baseline_mean_predictions(pool, n), true_cochs,
        ),
    }

    # --- Print summary ------------------------------------------------------
    print("\n=== zero-shot transfer to EPIC-Sounds ===")
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
        "checkpoint": str(checkpoint),
        "annotations_csv": str(annotations_csv),
        "videos_dir": str(videos_dir),
        "model_cfg": asdict(cfg),
        "metrics": metrics,
        "clips": sample_metadata,
    }
    out_metrics.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_metrics}")

    # --- Listening samples --------------------------------------------------
    if n_samples > 0:
        samples_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pred_cochs), size=min(n_samples, len(pred_cochs)), replace=False)
        for k in idx:
            meta = sample_metadata[k]
            tag = f"{meta['video_id']}_t{meta['onset_time']:.3f}".replace(".", "p")
            pred_wav = cochleagram_to_waveform(pred_cochs[k], sr=TARGET_SR, envelope_sr=ENVELOPE_SR)
            true_wav = cochleagram_to_waveform(true_cochs[k], sr=TARGET_SR, envelope_sr=ENVELOPE_SR)
            sf.write(str(samples_dir / f"{tag}_pred.wav"), pred_wav, TARGET_SR)
            sf.write(str(samples_dir / f"{tag}_real.wav"), true_wav, TARGET_SR)
        print(f"wrote {len(idx)} predicted/real wav pairs to {samples_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=CHECKPOINT_DIR / "three_stream_app-m3d-flow.pt",
    )
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Comma-separated subset of EPIC-Sounds class labels (default: "
        "substring filter to impact-style sounds).",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "eval_zero_shot")
    args = parser.parse_args()
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    main(
        checkpoint=args.checkpoint,
        annotations_csv=args.annotations_csv,
        videos_dir=args.videos_dir,
        classes=classes,
        max_clips=args.max_clips,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
    )
