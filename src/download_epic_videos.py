"""Download specific EPIC-KITCHENS-100 video files for the zero-shot eval.

Wraps the official `epic_downloader.py` (from
https://github.com/epic-kitchens/epic-kitchens-download-scripts) with a
per-video loop so:
  - We get exactly the files the eval needs (one .MP4 per video_id), nothing
    more — no flow frames, no rgb frames, no metadata blobs, no consent forms.
  - --specific-videos taking a single argument is fine; we just call the
    script once per video.
  - Already-downloaded files are skipped (resumable on interrupt).
  - Final layout matches what epic_sounds_dataset.py looks for:
        <output_path>/<participant_id>/videos/<video_id>.MP4

Prereq: the official downloader must be cloned somewhere reachable. On Colab:
    !git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts /tmp/epic-dl

Usage:
    python src/download_epic_videos.py \\
        --video-ids data/video_ids.txt \\
        --output-path data/epic
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MIN_VIDEO_SIZE_BYTES = 50_000_000  # 50 MB sanity floor — real EPIC videos are ≥ 200 MB


def _expected_path(output_path: Path, video_id: str) -> Path:
    """Where the official downloader places <video_id>.MP4 once it finishes."""
    participant = video_id.split("_")[0]  # "P01_11" -> "P01"
    return output_path / participant / "videos" / f"{video_id}.MP4"


def _is_already_downloaded(output_path: Path, video_id: str) -> bool:
    p = _expected_path(output_path, video_id)
    return p.exists() and p.stat().st_size >= MIN_VIDEO_SIZE_BYTES


def _download_one(downloader: Path, output_path: Path, video_id: str) -> int:
    """Returns the subprocess exit code from the official downloader."""
    cmd = [
        sys.executable, str(downloader),
        "--videos",
        "--specific-videos", video_id,
        "--output-path", str(output_path),
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-ids", type=Path, required=True,
        help="Text file with one video_id per line (output of select_epic_subset.py).",
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Where to put the videos. Final layout: <output_path>/<P_id>/videos/<video_id>.MP4.",
    )
    parser.add_argument(
        "--downloader-path", type=Path,
        default=Path("/tmp/epic-dl/epic_downloader.py"),
        help="Path to the official epic_downloader.py.",
    )
    args = parser.parse_args()

    if not args.downloader_path.exists():
        sys.exit(
            f"epic_downloader.py not found at {args.downloader_path}.\n"
            "Clone the official downloader first:\n"
            "  !git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts "
            "/tmp/epic-dl"
        )
    if not args.video_ids.exists():
        sys.exit(f"video-ids file not found: {args.video_ids}")

    args.output_path.mkdir(parents=True, exist_ok=True)
    video_ids = [
        line.strip()
        for line in args.video_ids.read_text().splitlines()
        if line.strip()
    ]
    if not video_ids:
        sys.exit(f"{args.video_ids} is empty.")

    print(f"target: {args.output_path}", flush=True)
    print(f"videos: {video_ids}", flush=True)

    failures: list[tuple[str, int]] = []
    skipped = 0
    downloaded = 0
    for i, vid in enumerate(video_ids, 1):
        if _is_already_downloaded(args.output_path, vid):
            print(f"[{i}/{len(video_ids)}] {vid}: already downloaded, skipping", flush=True)
            skipped += 1
            continue
        print(f"[{i}/{len(video_ids)}] {vid}: downloading...", flush=True)
        rc = _download_one(args.downloader_path, args.output_path, vid)
        if rc != 0:
            print(f"[{i}/{len(video_ids)}] {vid}: downloader exited {rc}", flush=True)
            failures.append((vid, rc))
            continue
        if not _is_already_downloaded(args.output_path, vid):
            print(
                f"[{i}/{len(video_ids)}] {vid}: downloader exited 0 but file is "
                f"missing or tiny at {_expected_path(args.output_path, vid)}",
                flush=True,
            )
            failures.append((vid, -1))
            continue
        downloaded += 1

    # Summary + layout sanity
    print(file=sys.stderr)
    print("=== summary ===", flush=True)
    print(f"  downloaded: {downloaded}", flush=True)
    print(f"  skipped:    {skipped}", flush=True)
    print(f"  failures:   {len(failures)}", flush=True)
    for vid, rc in failures:
        print(f"    FAIL {vid} (rc={rc})", flush=True)

    found = sorted(args.output_path.rglob("*.MP4"))
    print(f"\n{len(found)} .MP4 files under {args.output_path}:", flush=True)
    for f in found:
        size_gb = f.stat().st_size / 1e9
        print(f"  {f.relative_to(args.output_path)}  ({size_gb:.2f} GB)", flush=True)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
