"""Three-stream cache driver.

Mirrors the structure of cache_visual_features.py: decode each video once,
extract the 17-frame window around every onset, run RAFT once per window,
JPEG-encode the frames, and write (clip_npz, flow_npy) per clip.

Single-process GPU driver only — RAFT + a single CUDA context is the right
shape here; multi-process GPU sharing isn't worth the complexity.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Match cache_visual_features.py: limit BLAS threads to keep CPU sane.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from cache_visual_features import _pick_device, _read_full_video
from config import (
    CACHE_FRAME_SIZE,
    CACHE_FRAMES_STORED,
    CLIP_FRAMES,
    MAX_TEMPORAL_JITTER,
    ONSET_FRAME_INDEX,
)
from dataset import ClipIndex, build_clip_index, discover_videos
from streams import build_raft, compute_flow_cache, save_streams, streams_cached


def _extract_window(all_frames: np.ndarray, clip: ClipIndex) -> np.ndarray | None:
    """Return the CACHE_FRAMES_STORED window for a clip, edge-replicated at
    video boundaries. Returns None if the canonical 15-frame onset window
    itself is out of bounds — matches build_clip_index's existing convention,
    so the train/test clip set is identical to the V1 path."""
    n = all_frames.shape[0]
    canonical_start = clip.onset_frame - ONSET_FRAME_INDEX
    canonical_end = canonical_start + CLIP_FRAMES
    if canonical_start < 0 or canonical_end > n:
        return None

    wide_start = canonical_start - MAX_TEMPORAL_JITTER
    wide_end = canonical_end + MAX_TEMPORAL_JITTER  # exclusive
    # Edge-replicate when the wider window pokes past the video boundaries.
    indices = np.clip(np.arange(wide_start, wide_end), 0, n - 1)
    return all_frames[indices]  # (CACHE_FRAMES_STORED, H, W, 3) uint8 RGB


def _process_video(
    video_clips: list[ClipIndex],
    raft: torch.nn.Module,
    device: torch.device,
) -> tuple[int, int]:
    """Cache (frames_jpeg_npz, flow_fp16) for every clip in one video.

    Returns (n_cached, n_skipped) where skipped includes clips with onset
    windows that don't fit even at the canonical 15-frame width.
    """
    pending = [c for c in video_clips if not streams_cached(c)]
    if not pending:
        return 0, 0
    video_path = pending[0].entry.video_path
    all_frames = _read_full_video(video_path, out_size=CACHE_FRAME_SIZE)
    if all_frames.shape[0] == 0:
        return 0, len(pending)

    n_cached = 0
    n_skipped = 0
    for clip in pending:
        window = _extract_window(all_frames, clip)
        if window is None:
            n_skipped += 1
            continue
        flow = compute_flow_cache(window, raft, device)
        save_streams(clip, window, flow)
        n_cached += 1
    return n_cached, n_skipped


def main(max_videos: int | None = None) -> None:
    entries = discover_videos()
    clips = build_clip_index(entries)
    print(f"discovered {len(entries)} videos, {len(clips)} valid onsets")

    by_video: dict[str, list[ClipIndex]] = defaultdict(list)
    for c in clips:
        by_video[c.entry.video_id].append(c)
    todo = [
        vid_clips for vid_clips in by_video.values()
        if any(not streams_cached(c) for c in vid_clips)
    ]
    print(
        f"{len(todo)} videos still need three-stream cache; "
        f"{len(by_video) - len(todo)} already cached"
    )
    if max_videos is not None:
        todo = todo[:max_videos]
        print(f"--max-videos {max_videos}: limiting this run to {len(todo)} videos")
    if not todo:
        return

    device = _pick_device()
    print(f"device: {device}")
    raft = build_raft(device)
    # Warm up RAFT so the first iteration's compile cost doesn't pollute the ETA.
    with torch.no_grad():
        warm = torch.zeros(1, 3, CACHE_FRAME_SIZE, CACHE_FRAME_SIZE, device=device)
        _ = raft(warm, warm)

    t0 = time.time()
    cached_total = 0
    skipped_total = 0
    failures: list[tuple[str, str]] = []

    for done, vid_clips in enumerate(todo, 1):
        vid = vid_clips[0].entry.video_id
        try:
            n_cached, n_skipped = _process_video(vid_clips, raft, device)
            cached_total += n_cached
            skipped_total += n_skipped
        except Exception as e:  # noqa: BLE001
            failures.append((vid, f"{type(e).__name__}: {e}"))
        if done % 5 == 0 or done == len(todo):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (len(todo) - done) / max(rate, 1e-6)
            print(
                f"  {done}/{len(todo)} videos  {cached_total} clips cached  "
                f"{skipped_total} skipped  elapsed {elapsed/60:.1f}m  "
                f"eta {eta/60:.1f}m  fails {len(failures)}",
                flush=True,
            )

    print(
        f"done in {(time.time()-t0)/60:.1f} min; "
        f"{cached_total} clips cached, {skipped_total} skipped, "
        f"{len(failures)} failures"
    )
    for vid, msg in failures[:20]:
        print(f"  FAIL {vid}: {msg}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Cap the number of videos processed this run (smoke-test).",
    )
    args = parser.parse_args()
    main(max_videos=args.max_videos)
