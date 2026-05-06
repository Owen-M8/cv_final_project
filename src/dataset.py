"""1.3 Dataset class.

Yields (video_clip, cochleagram) pairs centered on impact onsets.

Discovery convention (Greatest Hits): for each video file in DATA_DIR, look for
a sibling `_denoised.wav` and (optionally) `_times.txt`. If `_times.txt` is
absent, run onset detection on the audio. Material labels (if present) come
from `_times.txt` columns 2+.

Cochleagrams are computed once per video and cached to CACHE_DIR/<video_id>.npz.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import scipy.signal as sps
import soundfile as sf
import torch
from torch.utils.data import Dataset

from config import (
    CACHE_DIR,
    CLIP_FRAMES,
    DATA_DIR,
    ONSET_FRAME_INDEX,
    SPLIT_RATIO,
    SPLIT_SEED,
    TARGET_SR,
    VIDEO_FPS,
)
from cochleagram import waveform_to_cochleagram
from onset_detection import detect_onsets, load_times_file

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
CLIP_INDEX_PATH = CACHE_DIR / "clip_index.json"


@dataclass
class VideoEntry:
    video_path: Path
    audio_path: Path
    times_path: Optional[Path]
    video_id: str


@dataclass
class ClipIndex:
    entry: VideoEntry
    onset_time: float           # seconds
    onset_frame: int            # absolute video-frame index of the impact
    material: Optional[str]     # for the material-probe eval
    action: Optional[str]       # "hit" / "scratch"


# Greatest Hits dataset layout:
#   <id>_denoised.mp4   (canonical denoised video)
#   <id>_denoised.wav   (matching audio)
#   <id>_denoised_thumb.mp4, <id>_mic.mp4, <id>_mic.wav  (variants we skip)
#   <id>_times.txt      (onset times + material/action labels)
def discover_videos(data_dir: Path = DATA_DIR) -> list[VideoEntry]:
    entries = []
    for vp in sorted(data_dir.rglob("*_denoised.mp4")):
        if vp.name.endswith("_thumb.mp4"):
            continue
        wav = vp.with_suffix(".wav")           # <id>_denoised.wav
        if not wav.exists():
            continue
        base_id = vp.stem[: -len("_denoised")]  # strip trailing "_denoised"
        times = vp.with_name(f"{base_id}_times.txt")
        entries.append(
            VideoEntry(
                video_path=vp,
                audio_path=wav,
                times_path=times if times.exists() else None,
                video_id=base_id,
            )
        )
    return entries


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    wav, sr = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        wav = sps.resample(wav, int(round(len(wav) * target_sr / sr))).astype(np.float32)
    return wav


VALID_ACTIONS = {"hit", "scratch"}  # drop "None" (missed/no-contact frames)


def parse_times_with_materials(path: Path) -> list[tuple[float, Optional[str], Optional[str]]]:
    """Greatest Hits times.txt: `<time> <material> <action> <reaction>`. Skip `None` actions."""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                t = float(parts[0])
            except ValueError:
                continue
            mat, action = parts[1], parts[2]
            if action not in VALID_ACTIONS:
                continue
            out.append((t, mat, action))
    return out


def build_clip_index(entries: list[VideoEntry]) -> list[ClipIndex]:
    """One ClipIndex per impact onset. Drops onsets too close to video edges."""
    clips: list[ClipIndex] = []
    for entry in entries:
        if entry.times_path is not None:
            onset_records = parse_times_with_materials(entry.times_path)
        else:
            wav = load_audio(entry.audio_path)
            onset_records = [(t, None, None) for t in detect_onsets(wav)]

        cap = cv2.VideoCapture(str(entry.video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
        cap.release()

        for t, mat, action in onset_records:
            onset_frame = int(round(t * fps))
            start = onset_frame - ONSET_FRAME_INDEX
            end = start + CLIP_FRAMES
            if start < 0 or end > n_frames:
                continue
            clips.append(ClipIndex(
                entry=entry, onset_time=t, onset_frame=onset_frame,
                material=mat, action=action,
            ))
    return clips


def video_level_split(
    clips: list[ClipIndex],
    ratio: float = SPLIT_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[ClipIndex], list[ClipIndex]]:
    """Split by video_id so clips from the same video stay together."""
    video_ids = sorted({c.entry.video_id for c in clips})
    rng = np.random.default_rng(seed)
    rng.shuffle(video_ids)
    n_train = int(round(len(video_ids) * ratio))
    train_ids = set(video_ids[:n_train])
    train, test = [], []
    for c in clips:
        (train if c.entry.video_id in train_ids else test).append(c)
    return train, test


def _cache_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}_coch.npz"


def cochleagram_for_video(entry: VideoEntry) -> np.ndarray:
    """Compute or load cached cochleagram for the entire audio track."""
    cache = _cache_path(entry.video_id)
    if cache.exists():
        return np.load(cache)["coch"]
    wav = load_audio(entry.audio_path)
    coch = waveform_to_cochleagram(wav)
    np.savez_compressed(cache, coch=coch)
    return coch


def slice_cochleagram_for_clip(
    full_coch: np.ndarray,
    onset_time: float,
    envelope_sr: int,
    clip_duration_s: float,
    onset_frac: float,
) -> np.ndarray:
    """Take a clip-length window from the full cochleagram, centered per onset_frac."""
    total_samples = int(round(clip_duration_s * envelope_sr))
    pre = int(round(onset_frac * total_samples))
    onset_idx = int(round(onset_time * envelope_sr))
    start = onset_idx - pre
    end = start + total_samples
    n_channels, n_total = full_coch.shape
    if start < 0:
        pad_left = -start
        start = 0
    else:
        pad_left = 0
    if end > n_total:
        pad_right = end - n_total
        end = n_total
    else:
        pad_right = 0
    window = full_coch[:, start:end]
    if pad_left or pad_right:
        window = np.pad(window, ((0, 0), (pad_left, pad_right)))
    return window


def read_video_clip(video_path: Path, start_frame: int, n_frames: int) -> np.ndarray:
    """Returns (n_frames, H, W, 3) uint8 in RGB."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < n_frames:  # pad with last frame
        last = frames[-1] if frames else np.zeros((1, 1, 3), dtype=np.uint8)
        frames.extend([last] * (n_frames - len(frames)))
    return np.stack(frames, axis=0)


def read_video_clip_at_time(
    video_path: Path,
    start_time_s: float,
    n_frames: int,
    target_fps: int = VIDEO_FPS,
) -> np.ndarray:
    """Read n_frames sampled at target_fps starting at start_time_s.

    Per-frame MSEC seek so the result is independent of the source video's
    native fps — important when running a 30-fps-trained model on EPIC-KITCHENS
    video at 50/60 fps. Slower than `read_video_clip` (the seek isn't free)
    but accurate.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    dt_s = 1.0 / target_fps
    for i in range(n_frames):
        t_ms = (start_time_s + i * dt_s) * 1000.0
        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < n_frames:
        last = frames[-1] if frames else np.zeros((1, 1, 3), dtype=np.uint8)
        frames.extend([last] * (n_frames - len(frames)))
    return np.stack(frames, axis=0)


class GreatestHitsDataset(Dataset):
    """Yields per-clip dicts. Cochleagrams cached per-video on first epoch."""

    def __init__(
        self,
        clips: list[ClipIndex],
        envelope_sr: int,
        clip_duration_s: float,
        onset_frac: float,
        return_video: bool = True,
    ):
        self.clips = clips
        self.envelope_sr = envelope_sr
        self.clip_duration_s = clip_duration_s
        self.onset_frac = onset_frac
        self.return_video = return_video
        self._coch_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.clips)

    def _get_full_coch(self, entry: VideoEntry) -> np.ndarray:
        if entry.video_id not in self._coch_cache:
            self._coch_cache[entry.video_id] = cochleagram_for_video(entry)
        return self._coch_cache[entry.video_id]

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        full_coch = self._get_full_coch(clip.entry)
        coch = slice_cochleagram_for_clip(
            full_coch,
            clip.onset_time,
            self.envelope_sr,
            self.clip_duration_s,
            self.onset_frac,
        )
        sample = {
            "cochleagram": torch.from_numpy(coch.astype(np.float32)),
            "video_id": clip.entry.video_id,
            "onset_time": clip.onset_time,
            "material": clip.material or "",
        }
        if self.return_video:
            start = clip.onset_frame - ONSET_FRAME_INDEX
            frames = read_video_clip(clip.entry.video_path, start, CLIP_FRAMES)
            sample["frames"] = torch.from_numpy(frames)  # (T, H, W, 3) uint8
        return sample


def save_clip_index(
    train_clips: list[ClipIndex],
    test_clips: list[ClipIndex],
    path: Path = CLIP_INDEX_PATH,
) -> None:
    """Persist the discovered clip index + train/test split to JSON.

    Lets a downstream box (e.g. Colab) reproduce the dataset without seeing the
    raw video files — paths are stored as strings but never opened during
    training, since the dataset reads only from cached cochleagrams + features.
    """
    import json

    def _ser(c: ClipIndex) -> dict:
        return {
            "video_id": c.entry.video_id,
            "video_path": str(c.entry.video_path),
            "audio_path": str(c.entry.audio_path),
            "times_path": str(c.entry.times_path) if c.entry.times_path else None,
            "onset_time": c.onset_time,
            "onset_frame": c.onset_frame,
            "material": c.material,
            "action": c.action,
        }

    payload = {
        "train": [_ser(c) for c in train_clips],
        "test": [_ser(c) for c in test_clips],
    }
    path.write_text(json.dumps(payload))
    print(f"wrote clip index ({len(train_clips)} train, {len(test_clips)} test) -> {path}")


def load_clip_index(path: Path = CLIP_INDEX_PATH) -> tuple[list[ClipIndex], list[ClipIndex]]:
    import json

    with open(path) as f:
        payload = json.load(f)
    _entry_cache: dict[str, VideoEntry] = {}

    def _de_entry(d: dict) -> VideoEntry:
        if d["video_id"] not in _entry_cache:
            _entry_cache[d["video_id"]] = VideoEntry(
                video_path=Path(d["video_path"]),
                audio_path=Path(d["audio_path"]),
                times_path=Path(d["times_path"]) if d["times_path"] else None,
                video_id=d["video_id"],
            )
        return _entry_cache[d["video_id"]]

    def _de(d: dict) -> ClipIndex:
        return ClipIndex(
            entry=_de_entry(d),
            onset_time=d["onset_time"],
            onset_frame=d["onset_frame"],
            material=d["material"],
            action=d["action"],
        )

    return [_de(d) for d in payload["train"]], [_de(d) for d in payload["test"]]


def build_datasets(
    use_cached_index: bool = True,
) -> tuple[GreatestHitsDataset, GreatestHitsDataset, list[ClipIndex], list[ClipIndex]]:
    """Build train/test datasets.

    If a serialized clip index exists at CLIP_INDEX_PATH, prefer it (lets us
    skip video discovery entirely on a box that doesn't have the .mp4 files).
    Otherwise discover from the local DATA_DIR and persist the index for next time.
    """
    from config import CLIP_DURATION_S, ENVELOPE_SR

    if use_cached_index and CLIP_INDEX_PATH.exists():
        train_clips, test_clips = load_clip_index()
    else:
        entries = discover_videos()
        if not entries:
            raise FileNotFoundError(
                f"No (video, audio) pairs found under {DATA_DIR}. "
                "Drop the Greatest Hits files in there once the download finishes."
            )
        clips = build_clip_index(entries)
        train_clips, test_clips = video_level_split(clips)
        save_clip_index(train_clips, test_clips)

    onset_frac = ONSET_FRAME_INDEX / CLIP_FRAMES
    train_ds = GreatestHitsDataset(train_clips, ENVELOPE_SR, CLIP_DURATION_S, onset_frac)
    test_ds = GreatestHitsDataset(test_clips, ENVELOPE_SR, CLIP_DURATION_S, onset_frac)
    return train_ds, test_ds, train_clips, test_clips


# ---------------------------------------------------------------------------
# Three-stream V2A dataset protocol + Greatest Hits implementation.
#
# Defines the interface every V2A dataset must implement so the train/eval
# loops can be written once and reused across Greatest Hits, EPIC-Sounds, and
# any future weak-supervision dataset. See `ThreeStreamGHDataset` below for
# the concrete Greatest Hits implementation.
# ---------------------------------------------------------------------------

from typing import Protocol


class V2AClipDataset(Protocol):
    """Per-clip dataset for V2A training.

    `__getitem__` returns a dict with keys:
        frames:      uint8  (CLIP_FRAMES, H, W, 3)        RGB
        flow:        fp16   (CLIP_FRAMES - 1, 2, H_f, W_f) raw RAFT flow
        cochleagram: float32 (n_channels, n_audio_steps)   audio target
        video_id:    str
        onset_time:  float
        material:    str (empty if unknown)
    """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...


class ThreeStreamGHDataset(Dataset):
    """Greatest Hits dataset for the three-stream model.

    Reads from the three-stream cache (frames + flow). Optionally applies
    temporal jitter by selecting a random window from the 17-frame cache;
    the audio target window is shifted by the same offset so audio-visual
    sync is preserved.

    Yields the V2AClipDataset dict described above.
    """

    def __init__(
        self,
        clips: list[ClipIndex],
        envelope_sr: int,
        clip_duration_s: float,
        onset_frac: float,
        temporal_jitter: int = 0,
    ):
        from config import CACHE_FRAMES_STORED, CLIP_FRAMES as _CF, MAX_TEMPORAL_JITTER

        if temporal_jitter < 0 or temporal_jitter > MAX_TEMPORAL_JITTER:
            raise ValueError(
                f"temporal_jitter must be in [0, {MAX_TEMPORAL_JITTER}]; got {temporal_jitter}"
            )
        # Window math: cache holds CACHE_FRAMES_STORED frames spanning ±M
        # around the canonical 15-frame window (M = MAX_TEMPORAL_JITTER).
        # Default window starts at index M (so cache[M : M+CLIP_FRAMES] is the
        # canonical 15-frame model input). Jitter offset is added to that.
        self._cache_frames = CACHE_FRAMES_STORED
        self._clip_frames = _CF
        self._max_jitter = MAX_TEMPORAL_JITTER
        self.clips = clips
        self.envelope_sr = envelope_sr
        self.clip_duration_s = clip_duration_s
        self.onset_frac = onset_frac
        self.temporal_jitter = temporal_jitter
        self._coch_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.clips)

    def _get_full_coch(self, entry: VideoEntry) -> np.ndarray:
        if entry.video_id not in self._coch_cache:
            self._coch_cache[entry.video_id] = cochleagram_for_video(entry)
        return self._coch_cache[entry.video_id]

    def __getitem__(self, idx: int) -> dict:
        # Imported here to break the streams.py <-> dataset.py cycle at module
        # load. streams.py only references ClipIndex via TYPE_CHECKING.
        from streams import load_streams

        clip = self.clips[idx]
        frames_full, flow_full = load_streams(clip)  # (17, H, W, 3) uint8, (16, 2, H_f, W_f) fp16

        if self.temporal_jitter > 0:
            off = int(np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1))
        else:
            off = 0
        start = self._max_jitter + off  # canonical center is at index _max_jitter

        frames = frames_full[start : start + self._clip_frames]                    # (T, H, W, 3)
        # Flow at index i is motion frame_i -> frame_{i+1}. For a frame window
        # [start .. start+T), aligned flow is [start .. start+T-1) -> (T-1) fields.
        flow = flow_full[start : start + self._clip_frames - 1]                    # (T-1, 2, H_f, W_f)

        # Audio window shifts with the visual window so the impact stays
        # synchronized; the impact's *position within the window* changes by
        # `off` frames, which trains the model to be robust to small onset
        # localization errors (the realistic case for cross-dataset transfer).
        full_coch = self._get_full_coch(clip.entry)
        shifted_onset_frac = (ONSET_FRAME_INDEX - off) / self._clip_frames
        coch = slice_cochleagram_for_clip(
            full_coch,
            clip.onset_time,
            self.envelope_sr,
            self.clip_duration_s,
            shifted_onset_frac,
        )

        return {
            "frames": frames,
            "flow": flow,
            "cochleagram": coch.astype(np.float32),
            "video_id": clip.entry.video_id,
            "onset_time": clip.onset_time,
            "material": clip.material or "",
        }


def build_three_stream_datasets(
    use_cached_index: bool = True,
    train_temporal_jitter: int = 1,
) -> tuple[ThreeStreamGHDataset, ThreeStreamGHDataset, list[ClipIndex], list[ClipIndex]]:
    """Three-stream sibling of `build_datasets`. Train set has temporal jitter
    enabled by default; test set has it off for deterministic eval."""
    from config import CLIP_DURATION_S, ENVELOPE_SR

    if use_cached_index and CLIP_INDEX_PATH.exists():
        train_clips, test_clips = load_clip_index()
    else:
        entries = discover_videos()
        if not entries:
            raise FileNotFoundError(
                f"No (video, audio) pairs found under {DATA_DIR}. "
                "Drop the Greatest Hits files in there once the download finishes."
            )
        clips = build_clip_index(entries)
        train_clips, test_clips = video_level_split(clips)
        save_clip_index(train_clips, test_clips)

    onset_frac = ONSET_FRAME_INDEX / CLIP_FRAMES
    train_ds = ThreeStreamGHDataset(
        train_clips, ENVELOPE_SR, CLIP_DURATION_S, onset_frac,
        temporal_jitter=train_temporal_jitter,
    )
    test_ds = ThreeStreamGHDataset(
        test_clips, ENVELOPE_SR, CLIP_DURATION_S, onset_frac,
        temporal_jitter=0,
    )
    return train_ds, test_ds, train_clips, test_clips
