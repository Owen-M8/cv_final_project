"""Pick the smallest EPIC-KITCHENS-100 video subset that covers an EPIC-Sounds
impact-clip eval of size N.

Reads the EPIC-Sounds annotations CSV, filters to impact-style classes, takes
the first --max-clips matching rows, and prints the unique video_ids needed.
Pipe the output into the EPIC download tool (or read it manually if you're
downloading by hand) to avoid pulling the full hundreds-of-GB dataset.

Usage:
    python src/select_epic_subset.py \\
        --annotations-csv path/to/EPIC_Sounds_validation.csv \\
        --max-clips 200 > video_ids.txt

video_ids print to stdout (one per line, ready to pipe). A breakdown of clips
per video and per class is printed to stderr.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the same impact-keyword filter the eval dataset uses, so the subset
# you download exactly matches what the eval will load.
from epic_sounds_dataset import DEFAULT_IMPACT_KEYWORDS, _read_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=200)
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Comma-separated subset of EPIC-Sounds class labels (default: "
        "substring filter to impact-style sounds).",
    )
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    rows = _read_annotations(args.annotations_csv, classes, DEFAULT_IMPACT_KEYWORDS)
    rows = rows[: args.max_clips]
    if not rows:
        sys.exit("no rows matched the filter; check --annotations-csv path or --classes value")

    by_video: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r["class"])

    # stdout: video_ids, sorted, one per line — ready to pipe into a download tool.
    for vid in sorted(by_video):
        print(vid)

    # stderr: human-readable summary so the user sees the spread.
    p = lambda *a, **kw: print(*a, file=sys.stderr, **kw)
    p()
    p(f"selected {len(rows)} clips spanning {len(by_video)} videos")
    p()
    p(f"{'video_id':<10} {'clips':>6}  classes")
    for vid in sorted(by_video, key=lambda v: -len(by_video[v])):
        cls_summary = ", ".join(f"{c}: {n}" for c, n in Counter(by_video[vid]).most_common(3))
        p(f"{vid:<10} {len(by_video[vid]):>6}  {cls_summary}")
    p()
    p("class breakdown across selected clips:")
    for cls, n in Counter(r["class"] for r in rows).most_common():
        p(f"  {n:>5}  {cls}")


if __name__ == "__main__":
    main()
