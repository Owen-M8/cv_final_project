"""Parallel one-shot pre-computation of per-video cochleagrams.

Runs `cochleagram_for_video` over every discovered video in parallel processes,
writing each result to cache/<video_id>_coch.npz. Re-runs are cheap: existing
cache entries are skipped automatically inside cochleagram_for_video.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

# Avoid BLAS oversubscription when we fan out to N processes.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import VideoEntry, _cache_path, cochleagram_for_video, discover_videos


def _process_one(entry: VideoEntry) -> tuple[str, bool, str]:
    if _cache_path(entry.video_id).exists():
        return entry.video_id, True, "cached"
    try:
        cochleagram_for_video(entry)
        return entry.video_id, True, "ok"
    except Exception as e:  # noqa: BLE001
        return entry.video_id, False, f"{type(e).__name__}: {e}"


def main(n_workers: int) -> None:
    entries = discover_videos()
    todo = [e for e in entries if not _cache_path(e.video_id).exists()]
    print(f"discovered {len(entries)} videos; {len(todo)} need cochleagram cache")
    if not todo:
        return
    print(f"workers: {n_workers}")
    t0 = time.time()
    done = 0
    failures: list[tuple[str, str]] = []
    with mp.Pool(n_workers) as pool:
        for vid, ok, msg in pool.imap_unordered(_process_one, todo, chunksize=1):
            done += 1
            if not ok:
                failures.append((vid, msg))
            if done % 25 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(todo)}  elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m  fails {len(failures)}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min; {len(failures)} failures")
    for vid, msg in failures[:20]:
        print(f"  FAIL {vid}: {msg}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    try:
        main(args.workers)
    except KeyboardInterrupt:
        print("\ninterrupted; partial cache preserved", file=sys.stderr)
