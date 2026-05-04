"""Parallel visual feature pre-computation.

Optimization vs. visual_features.precompute_all: process one video at a time so
each .mp4 is opened once instead of once per clip. With ~30 onsets/video, this
is ~30x fewer video opens.

Per-clip output: cache/<video_id>_f<onset_frame:06d>_feat.npy of shape (15, 1024)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Limit BLAS threads per worker so N workers don't oversubscribe the CPU.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import CLIP_FRAMES, ONSET_FRAME_INDEX, PER_FRAME_FEAT_DIM
from dataset import ClipIndex, build_clip_index, discover_videos
from visual_features import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _build_resnet18,
    _feat_cache_path,
)


_NORM_MEAN: dict[torch.device, torch.Tensor] = {}
_NORM_STD: dict[torch.device, torch.Tensor] = {}


def _norm(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if device not in _NORM_MEAN:
        _NORM_MEAN[device] = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        _NORM_STD[device] = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return _NORM_MEAN[device], _NORM_STD[device]


def _gpu_preprocess(
    rgb_frames: np.ndarray,
    device: torch.device,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """rgb_frames: (N, H, W, 3) uint8. Returns (rgb_normed, spacetime_normed), both (N, 3, 224, 224).

    Resizes one chunk of frames at a time so we never hold the full-resolution
    float tensor on the GPU (a 73s 1920x1080 clip is ~50 GB at fp32).

    Spacetime image at frame t = stack of gray(t-1), gray(t), gray(t+1)
    (clamped at boundaries), matching visual_features._grayscale_stack.
    """
    n = rgb_frames.shape[0]
    rgb_resized = torch.empty((n, 3, 224, 224), device=device, dtype=torch.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        x = torch.from_numpy(rgb_frames[s:e]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=256, mode="bilinear", align_corners=False)
        h, w = x.shape[-2:]
        i, j = (h - 224) // 2, (w - 224) // 2
        rgb_resized[s:e] = x[:, :, i : i + 224, j : j + 224]

    # ITU-R BT.601 luma (cv2.COLOR_RGB2GRAY uses these weights).
    gray = 0.2989 * rgb_resized[:, 0:1] + 0.5870 * rgb_resized[:, 1:2] + 0.1140 * rgb_resized[:, 2:3]

    idx_prev = torch.clamp(torch.arange(n, device=device) - 1, min=0)
    idx_next = torch.clamp(torch.arange(n, device=device) + 1, max=n - 1)
    spacetime = torch.cat([gray[idx_prev], gray, gray[idx_next]], dim=1)  # (n, 3, 224, 224)

    mean, std = _norm(device)
    return (rgb_resized - mean) / std, (spacetime - mean) / std


def _read_full_video(path: Path) -> np.ndarray:
    """Read every frame as RGB uint8 in one pass. Returns (N, H, W, 3)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)


def _features_for_video(
    video_clips: list[ClipIndex],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> int:
    """Compute and cache features for every clip belonging to one video.

    Decode video once -> GPU preprocess all frames once -> slice clip windows ->
    batched ResNet18 inference. ~10x faster than per-frame PIL transforms.
    """
    pending = [c for c in video_clips if not _feat_cache_path(c).exists()]
    if not pending:
        return 0
    video_path = pending[0].entry.video_path
    all_frames = _read_full_video(video_path)
    n_video = all_frames.shape[0]
    if n_video == 0:
        return 0

    rgb_normed, st_normed = _gpu_preprocess(all_frames, device)  # (N, 3, 224, 224) each

    rgb_chunks: list[torch.Tensor] = []
    st_chunks: list[torch.Tensor] = []
    clip_ranges: list[tuple[int, int]] = []

    cursor = 0
    for clip in pending:
        start = clip.onset_frame - ONSET_FRAME_INDEX
        end = start + CLIP_FRAMES
        if start < 0 or end > n_video:
            clip_ranges.append((-1, -1))
            continue
        rgb_chunks.append(rgb_normed[start:end])
        st_chunks.append(st_normed[start:end])
        clip_ranges.append((cursor, cursor + CLIP_FRAMES))
        cursor += CLIP_FRAMES

    if not rgb_chunks:
        return 0

    rgb_all = torch.cat(rgb_chunks, dim=0)
    st_all = torch.cat(st_chunks, dim=0)

    def run_in_batches(x: torch.Tensor) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                outs.append(model(x[i : i + batch_size]).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    rgb_feat = run_in_batches(rgb_all)
    st_feat = run_in_batches(st_all)
    cat_feat = np.concatenate([rgb_feat, st_feat], axis=-1)  # (sum_T, 1024)

    written = 0
    for clip, (lo, hi) in zip(pending, clip_ranges):
        if lo < 0:
            continue
        feats = cat_feat[lo:hi]
        assert feats.shape == (CLIP_FRAMES, PER_FRAME_FEAT_DIM), feats.shape
        np.save(_feat_cache_path(clip), feats)
        written += 1
    return written


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _worker(args: tuple[Path, list[ClipIndex]]) -> tuple[str, int, str]:
    video_path, clips = args
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        device = _pick_device()
        model = _build_resnet18(device)
        n = _features_for_video(clips, model, device)
        return clips[0].entry.video_id, n, "ok"
    except Exception as e:  # noqa: BLE001
        return clips[0].entry.video_id if clips else str(video_path), 0, f"{type(e).__name__}: {e}"


def main(n_workers: int) -> None:
    entries = discover_videos()
    clips = build_clip_index(entries)
    print(f"discovered {len(entries)} videos, {len(clips)} valid onsets")

    by_video: dict[str, list[ClipIndex]] = defaultdict(list)
    for c in clips:
        by_video[c.entry.video_id].append(c)
    todo = []
    for vid, vid_clips in by_video.items():
        if any(not _feat_cache_path(c).exists() for c in vid_clips):
            todo.append((vid_clips[0].entry.video_path, vid_clips))
    print(f"{len(todo)} videos still need feature cache; {len(by_video) - len(todo)} already cached")
    if not todo:
        return

    print(f"workers: {n_workers}")
    t0 = time.time()
    done = 0
    written_total = 0
    failures: list[tuple[str, str]] = []

    with mp.Pool(n_workers) as pool:
        for vid, n_written, msg in pool.imap_unordered(_worker, todo, chunksize=1):
            done += 1
            written_total += n_written
            if msg != "ok":
                failures.append((vid, msg))
            if done % 10 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(
                    f"  {done}/{len(todo)} videos  {written_total} clips cached  "
                    f"elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m  fails {len(failures)}",
                    flush=True,
                )
    print(f"done in {(time.time()-t0)/60:.1f} min; {written_total} clips cached, {len(failures)} failures")
    for vid, msg in failures[:20]:
        print(f"  FAIL {vid}: {msg}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    main(args.workers)
