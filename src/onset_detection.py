"""1.1 Onset detection.

Simplification of Owens et al.'s pipeline (amplitude-gradient threshold +
mean-shift + NMS). v1 uses scipy.signal.find_peaks on the audio envelope.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf

from config import ONSET_MIN_SEPARATION_S, TARGET_SR


def envelope(waveform: np.ndarray, sr: int, smooth_ms: float = 10.0) -> np.ndarray:
    """Rectified, low-pass-smoothed amplitude envelope."""
    rect = np.abs(waveform)
    win = max(1, int(sr * smooth_ms / 1000))
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(rect, kernel, mode="same")


def detect_onsets(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    height_quantile: float = 0.95,
    min_separation_s: float = ONSET_MIN_SEPARATION_S,
) -> np.ndarray:
    """Return onset timestamps (seconds).

    height_quantile picks a threshold from the envelope distribution so we don't
    have to retune per-clip. Tighten with eyeballed peaks if results look bad.
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    env = envelope(waveform, sr)
    height = np.quantile(env, height_quantile)
    distance = int(min_separation_s * sr)
    peaks, _ = sps.find_peaks(env, height=height, distance=distance)
    return peaks.astype(np.float64) / sr


def load_times_file(path: Path) -> np.ndarray:
    """Greatest Hits ships per-video `_times.txt` files: lines of `time material reaction`.

    If the dataset directory has these, prefer them over re-detecting.
    """
    times = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                times.append(float(parts[0]))
            except (ValueError, IndexError):
                continue
    return np.asarray(times, dtype=np.float64)


def _plot_check(wav_path: Path) -> None:
    import matplotlib.pyplot as plt

    waveform, sr = sf.read(str(wav_path))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    onsets = detect_onsets(waveform, sr)
    t = np.arange(len(waveform)) / sr
    plt.figure(figsize=(12, 3))
    plt.plot(t, waveform, lw=0.4)
    for o in onsets:
        plt.axvline(o, color="red", alpha=0.6, lw=0.8)
    plt.xlabel("time (s)")
    plt.title(f"{wav_path.name}: {len(onsets)} onsets")
    out = wav_path.with_suffix(".onsets.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path, help="audio file to visualize")
    args = parser.parse_args()
    _plot_check(args.wav)
