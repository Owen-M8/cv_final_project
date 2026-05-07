"""EPIC-Sounds zero-shot eval dataset for the three-stream V2A model.

Loads timestamped sound-event annotations from EPIC-Sounds and pairs them with
video clips from EPIC-KITCHENS-100. Conforms to the V2AClipDataset protocol so
the same eval loop works on Greatest Hits and EPIC-Sounds.

Data the user must supply (this module does *not* download them):
  1. EPIC-Sounds annotations CSV (~800 KB for validation; ~6 MB for train)
       https://github.com/epic-kitchens/epic-sounds-annotations
       Files: EPIC_Sounds_{train,validation}.csv
  2. EPIC-KITCHENS-100 video files
       https://epic-kitchens.github.io/2024
       Layout assumed:
         <videos_dir>/<participant_id>/videos/<video_id>.MP4
       or fallback:
         <videos_dir>/<video_id>.MP4

Real CSV columns (verified against the validation file):
  annotation_id, participant_id, video_id, start_timestamp, stop_timestamp,
  start_sample, stop_sample, description, class, class_id
Timestamps are HH:MM:SS.ms strings; samples are audio sample indices (24 kHz).

Cache layout (under cache/epic_sounds/):
  <video_id>_t<onset>_clip.npz    JPEG-encoded 17 frames at 224x224
  <video_id>_t<onset>_flow.npy    fp16 (16, 2, 56, 56)
  <video_id>_coch.npz             full-track cochleagram (computed once per video)

Frames are read at VIDEO_FPS (30) regardless of the source video's native fps,
so motion velocities match the model's training distribution. EPIC at 50/60
fps would otherwise systematically under-shoot loudness.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from cochleagram import waveform_to_cochleagram
from config import (
    CACHE_DIR,
    CACHE_FRAMES_STORED,
    CLIP_DURATION_S,
    CLIP_FRAMES,
    ENVELOPE_SR,
    MAX_TEMPORAL_JITTER,
    ONSET_FRAME_INDEX,
    TARGET_SR,
    VIDEO_FPS,
)
from dataset import read_video_clip_at_time, slice_cochleagram_for_clip

# Substring filters against the `class` column. Tuned to overlap with Greatest
# Hits (drumstick on objects, isolated impact onsets). Override via `classes=`.
DEFAULT_IMPACT_KEYWORDS = (
    "collision", "knock", "tap", "hit", "click", "thud",
)

EPIC_CACHE_DIR = CACHE_DIR / "epic_sounds"


@dataclass
class EpicClip:
    video_id: str
    video_path: Path
    onset_time: float       # start_seconds from the EPIC-Sounds CSV
    duration: float         # stop_seconds - start_seconds
    class_label: str        # raw class string from the CSV (used as "material")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _resolve_video_path(video_id: str, videos_dir: Path) -> Optional[Path]:
    """Try the common EPIC-KITCHENS layouts in order.

    The official epic_downloader.py nests files under an extra
    `EPIC-KITCHENS/` directory; we check that first.
    """
    participant = video_id.split("_")[0]  # "P01_01" -> "P01"
    candidates = [
        videos_dir / "EPIC-KITCHENS" / participant / "videos" / f"{video_id}.MP4",
        videos_dir / participant / "videos" / f"{video_id}.MP4",
        videos_dir / participant / f"{video_id}.MP4",
        videos_dir / f"{video_id}.MP4",
        videos_dir / f"{video_id}.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_annotations(
    csv_path: Path,
    classes: list[str] | None,
    keywords: tuple[str, ...] = DEFAULT_IMPACT_KEYWORDS,
) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row.get("class", "").lower()
            if classes:
                if cls not in {c.lower() for c in classes}:
                    continue
            else:
                if not any(kw in cls for kw in keywords):
                    continue
            rows.append(row)
    return rows


def _timestamp_to_seconds(ts: str) -> float:
    """EPIC-Sounds timestamps look like 'HH:MM:SS.ms'. Returns seconds (float)."""
    h, m, rest = ts.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def discover_epic_clips(
    annotations_csv: Path,
    videos_dir: Path,
    classes: list[str] | None = None,
    max_clips: int | None = None,
) -> list[EpicClip]:
    """Build the EPIC-Sounds clip list. Skips clips whose video file is missing.

    Raises FileNotFoundError with a helpful message if no videos match (so the
    user can fix the layout before running a full eval).
    """
    rows = _read_annotations(annotations_csv, classes)
    clips: list[EpicClip] = []
    missing_videos: set[str] = set()
    parse_failed = 0
    for row in rows:
        vid = row.get("video_id")
        if not vid:
            continue
        try:
            t_start = _timestamp_to_seconds(row["start_timestamp"])
            t_end = _timestamp_to_seconds(row["stop_timestamp"])
        except (KeyError, ValueError):
            parse_failed += 1
            continue
        path = _resolve_video_path(vid, videos_dir)
        if path is None:
            missing_videos.add(vid)
            continue
        clips.append(EpicClip(
            video_id=vid,
            video_path=path,
            onset_time=t_start,
            duration=max(0.05, t_end - t_start),
            class_label=row.get("class", ""),
        ))
        if max_clips is not None and len(clips) >= max_clips:
            break
    if not clips and missing_videos:
        raise FileNotFoundError(
            f"No EPIC-KITCHENS videos found under {videos_dir}.\n"
            f"Expected layouts: <videos_dir>/<P_id>/videos/<video_id>.MP4 or "
            f"<videos_dir>/<video_id>.MP4\n"
            f"Sample of missing video_ids: {sorted(missing_videos)[:5]}"
        )
    if missing_videos:
        print(f"[epic-sounds] skipped {len(missing_videos)} clips with missing videos")
    if parse_failed:
        print(f"[epic-sounds] skipped {parse_failed} rows with unparseable timestamps")
    return clips


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EpicSoundsClipDataset(Dataset):
    """Conforms to V2AClipDataset (see dataset.py).

    Per-clip work the first time a clip is accessed:
      - Decode 17 frames at VIDEO_FPS around the onset, resize+crop to 224x224.
      - Run RAFT once for the 16 flow fields, downsample to 56x56.
      - Compute the full-track cochleagram (cached per-video, not per-clip).
    All three are cached to disk under cache/epic_sounds/. Subsequent epochs
    are I/O bound.
    """

    def __init__(
        self,
        clips: list[EpicClip],
        device: Optional[torch.device] = None,
    ):
        EPIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.clips = clips
        if device is None:
            from cache_visual_features import _pick_device
            device = _pick_device()  # CUDA -> MPS -> CPU
        self.device = device
        self._raft = None  # built lazily; cached clips skip RAFT entirely

    def __len__(self) -> int:
        return len(self.clips)

    def _cache_paths(self, clip: EpicClip) -> dict[str, Path]:
        # f"{:.3f}" is enough (ms-level precision); replace '.' so the file
        # name has no extra dots that confuse path operations.
        onset_str = f"{clip.onset_time:.3f}".replace(".", "p")
        base = EPIC_CACHE_DIR / f"{clip.video_id}_t{onset_str}"
        return {
            "clip": base.with_name(base.name + "_clip.npz"),
            "flow": base.with_name(base.name + "_flow.npy"),
            "coch_full": EPIC_CACHE_DIR / f"{clip.video_id}_coch.npz",
        }

    def _load_audio_cochleagram(self, clip: EpicClip) -> np.ndarray:
        coch_path = self._cache_paths(clip)["coch_full"]
        if coch_path.exists():
            with np.load(coch_path) as f:
                return f["coch"]
        # librosa.load with sr=TARGET_SR resamples for us. Reads the audio
        # track from the .MP4 via audioread+ffmpeg — slow on first call (one
        # full-track decode per video) but cached afterwards.
        import librosa
        wav, _ = librosa.load(str(clip.video_path), sr=TARGET_SR, mono=True)
        coch = waveform_to_cochleagram(wav.astype(np.float32))
        np.savez_compressed(coch_path, coch=coch.astype(np.float32))
        return coch

    def _load_or_compute_streams(self, clip: EpicClip) -> tuple[np.ndarray, np.ndarray]:
        from streams import (
            build_raft,
            compute_flow_cache,
            load_streams_from_paths,
            resize_and_crop,
            save_streams_to_paths,
        )

        paths = self._cache_paths(clip)
        if paths["clip"].exists() and paths["flow"].exists():
            return load_streams_from_paths(paths["clip"], paths["flow"])

        # Read CACHE_FRAMES_STORED frames at VIDEO_FPS centered on the onset.
        # The canonical onset position within the 15-frame model window is at
        # ONSET_FRAME_INDEX; the wider 17-frame cache extends ±MAX_TEMPORAL_JITTER.
        wide_start_frame_offset = ONSET_FRAME_INDEX + MAX_TEMPORAL_JITTER
        start_time = clip.onset_time - wide_start_frame_offset / VIDEO_FPS
        frames_native = read_video_clip_at_time(
            clip.video_path, max(0.0, start_time), CACHE_FRAMES_STORED, VIDEO_FPS,
        )
        frames_224 = resize_and_crop(frames_native)

        if self._raft is None:
            self._raft = build_raft(self.device)
        flow = compute_flow_cache(frames_224, self._raft, self.device)

        save_streams_to_paths(paths["clip"], paths["flow"], frames_224, flow)
        return frames_224, flow

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        frames_full, flow_full = self._load_or_compute_streams(clip)

        # Eval is deterministic — pick the canonical (un-jittered) window.
        start = MAX_TEMPORAL_JITTER
        frames = frames_full[start : start + CLIP_FRAMES]
        flow = flow_full[start : start + CLIP_FRAMES - 1]

        full_coch = self._load_audio_cochleagram(clip)
        onset_frac = ONSET_FRAME_INDEX / CLIP_FRAMES
        coch = slice_cochleagram_for_clip(
            full_coch, clip.onset_time, ENVELOPE_SR, CLIP_DURATION_S, onset_frac,
        )

        return {
            "frames": frames,
            "flow": flow,
            "cochleagram": coch.astype(np.float32),
            "video_id": clip.video_id,
            "onset_time": clip.onset_time,
            "material": clip.class_label,  # repurpose the field for class label
        }


def build_epic_sounds_dataset(
    annotations_csv: Path,
    videos_dir: Path,
    classes: list[str] | None = None,
    max_clips: int | None = None,
) -> EpicSoundsClipDataset:
    clips = discover_epic_clips(annotations_csv, videos_dir, classes, max_clips)
    print(f"[epic-sounds] {len(clips)} clips matched the filter")
    return EpicSoundsClipDataset(clips)
