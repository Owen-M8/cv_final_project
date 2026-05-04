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
) -> tuple[torch.Tensor, torch.Tensor]:
    """rgb_frames: (N, 224, 224, 3) uint8 — already resized + center-cropped on CPU.

    GPU work here is just float conversion + ImageNet normalize + spacetime stack.
    Spacetime image at frame t = (gray(t-1), gray(t), gray(t+1)) clamped at edges.
    """
    n = rgb_frames.shape[0]
    x = torch.from_numpy(rgb_frames).to(device).permute(0, 3, 1, 2).float() / 255.0  # (N, 3, 224, 224)

    # ITU-R BT.601 luma (cv2.COLOR_RGB2GRAY weights).
    gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

    idx_prev = torch.clamp(torch.arange(n, device=device) - 1, min=0)
    idx_next = torch.clamp(torch.arange(n, device=device) + 1, max=n - 1)
    spacetime = torch.cat([gray[idx_prev], gray, gray[idx_next]], dim=1)  # (N, 3, 224, 224)

    mean, std = _norm(device)
    return (x - mean) / std, (spacetime - mean) / std


def _read_full_video(path: Path, out_size: int = 224, resize_short_to: int = 256) -> np.ndarray:
    """Read every frame as RGB uint8, resized + center-cropped to out_size x out_size.

    Resizing during decode is critical: 1920x1080 video at 30fps over ~70s decodes
    to ~14 GB of uint8 if we keep original resolution. Resize-then-crop drops it
    to ~330 MB. Same target geometry as the standard ImageNet pipeline (resize
    short side to 256, center-crop to 224).
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if h < w:
            new_h, new_w = resize_short_to, int(round(w * resize_short_to / h))
        else:
            new_h, new_w = int(round(h * resize_short_to / w)), resize_short_to
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        ch = (new_h - out_size) // 2
        cw = (new_w - out_size) // 2
        frame = frame[ch : ch + out_size, cw : cw + out_size]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return np.zeros((0, out_size, out_size, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


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


def main_single(batch_size: int = 128) -> None:
    """Single-process driver. MPS/CUDA can't share a GPU across processes
    productively, so we just go single-process and saturate the device."""
    entries = discover_videos()
    clips = build_clip_index(entries)
    print(f"discovered {len(entries)} videos, {len(clips)} valid onsets")

    by_video: dict[str, list[ClipIndex]] = defaultdict(list)
    for c in clips:
        by_video[c.entry.video_id].append(c)
    todo = []
    for vid, vid_clips in by_video.items():
        if any(not _feat_cache_path(c).exists() for c in vid_clips):
            todo.append(vid_clips)
    print(f"{len(todo)} videos still need feature cache; {len(by_video) - len(todo)} already cached")
    if not todo:
        return

    device = _pick_device()
    print(f"device: {device}")
    model = _build_resnet18(device)
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, 224, 224, device=device))  # warmup

    t0 = time.time()
    written_total = 0
    failures: list[tuple[str, str]] = []
    for done, vid_clips in enumerate(todo, 1):
        vid = vid_clips[0].entry.video_id
        try:
            n_written = _features_for_video(vid_clips, model, device, batch_size=batch_size)
            written_total += n_written
        except Exception as e:  # noqa: BLE001
            failures.append((vid, f"{type(e).__name__}: {e}"))
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


def main(n_workers: int) -> None:
    """Multi-process CPU driver. Use this only when GPU is unavailable."""
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
    parser.add_argument("--workers", type=int, default=1, help="CPU-only multi-process; use 1 to force GPU single-process")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.workers <= 1:
        main_single(batch_size=args.batch_size)
    else:
        mp.set_start_method("spawn", force=True)
        main(args.workers)
