"""Run locally once visual feature caching finishes. Produces:
  cache/clip_index.json   — train/test clip metadata (no video files needed downstream)
  bundle.tar.zst (or .tar.gz) — single archive of cache/ ready to upload to Drive

Then on Colab: download bundle, extract into project root, run train.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CACHE_DIR, PROJECT_ROOT
from dataset import (
    CLIP_INDEX_PATH,
    build_clip_index,
    discover_videos,
    save_clip_index,
    video_level_split,
)


def write_clip_index() -> None:
    if CLIP_INDEX_PATH.exists():
        print(f"clip index already exists at {CLIP_INDEX_PATH}; rewriting from current data/")
    entries = discover_videos()
    clips = build_clip_index(entries)
    train_clips, test_clips = video_level_split(clips)
    save_clip_index(train_clips, test_clips)


def report_cache_status() -> None:
    coch = list(CACHE_DIR.glob("*_coch.npz"))
    feat = list(CACHE_DIR.glob("*_feat.npy"))
    coch_mb = sum(p.stat().st_size for p in coch) / 1e6
    feat_mb = sum(p.stat().st_size for p in feat) / 1e6
    print(f"  {len(coch):>5} cochleagram files  ({coch_mb:.1f} MB)")
    print(f"  {len(feat):>5} visual feature files ({feat_mb:.1f} MB)")
    print(f"  total cache:  {(coch_mb + feat_mb) / 1024:.2f} GB")


def make_archive(out_path: Path) -> Path:
    """Tar+gzip the cache/ directory. Use system tar; faster + handles long file lists."""
    if out_path.exists():
        out_path.unlink()
    print(f"creating {out_path} (this may take a few minutes)...")
    subprocess.check_call(
        ["tar", "-czf", str(out_path), "-C", str(PROJECT_ROOT), "cache"],
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({size_mb:.1f} MB)")
    return out_path


def main() -> None:
    write_clip_index()
    print("\ncache status:")
    report_cache_status()
    out = PROJECT_ROOT / "cache_bundle.tar.gz"
    make_archive(out)
    print("\nNext steps:")
    print(f"  1. Upload {out.name} to Google Drive (e.g. MyDrive/cv_final_project/)")
    print(f"  2. Open notebooks/train_on_colab.ipynb in Colab and run all cells")


if __name__ == "__main__":
    main()
