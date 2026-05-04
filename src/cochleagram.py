"""1.2 Cochleagram extraction + inversion.

Forward: waveform -> 40-channel ERB cochleagram, downsampled to 90 Hz, ^0.3.
Inversion: pycochleagram's iterative reconstruction (Griffin-Lim style).

Parametric fallback: impose subband envelopes on white noise (one pass). Used
if pycochleagram's inversion is unavailable or unstable.

CRITICAL: round-trip a real clip and listen before training anything.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf

from config import (
    COCH_HIGH_HZ,
    COCH_LOW_HZ,
    COMPRESSION_EXP,
    ENVELOPE_SR,
    N_COCH_FILTERS,
    TARGET_SR,
)

try:
    from pycochleagram import cochleagram as _pyc
except ImportError:
    _pyc = None


def _require_pyc():
    if _pyc is None:
        raise ImportError(
            "pycochleagram not installed. Run:\n"
            "  pip install git+https://github.com/mcdermottLab/pycochleagram.git"
        )


def waveform_to_cochleagram(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    n: int = N_COCH_FILTERS,
    envelope_sr: int = ENVELOPE_SR,
    compression: float = COMPRESSION_EXP,
) -> np.ndarray:
    """Forward pipeline. Returns cochleagram of shape (n_channels, n_envelope_samples).

    n_channels = n + 2 (low-pass + high-pass tails added by the library).
    """
    _require_pyc()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)

    # human_cochleagram with sample_factor=1 returns Hilbert envelopes per subband.
    coch = _pyc.human_cochleagram(
        waveform,
        sr,
        n=n,
        low_lim=COCH_LOW_HZ,
        hi_lim=min(COCH_HIGH_HZ, sr // 2 - 1),
        sample_factor=1,
        downsample=None,
        nonlinearity=None,
        ret_mode="envs",
        strict=False,
    )
    coch = np.asarray(coch, dtype=np.float32)

    # Downsample envelopes from sr to envelope_sr.
    n_out = max(1, int(round(coch.shape[1] * envelope_sr / sr)))
    coch_ds = sps.resample(coch, n_out, axis=1).astype(np.float32)
    coch_ds = np.clip(coch_ds, 0.0, None)

    # Compressive nonlinearity (paper: 0.3).
    return coch_ds ** compression


def cochleagram_to_waveform(
    coch: np.ndarray,
    sr: int = TARGET_SR,
    n: int = N_COCH_FILTERS,
    envelope_sr: int = ENVELOPE_SR,
    compression: float = COMPRESSION_EXP,
    n_iter: int = 20,
) -> np.ndarray:
    """Inverse pipeline. Falls back to parametric synthesis if iterative inversion fails."""
    _require_pyc()
    coch = np.asarray(coch, dtype=np.float32)

    # Undo compression and upsample envelopes back to audio rate.
    env = np.clip(coch, 0.0, None) ** (1.0 / compression)
    n_audio = int(round(env.shape[1] * sr / envelope_sr))
    env_full = sps.resample(env, n_audio, axis=1).astype(np.float32)
    env_full = np.clip(env_full, 0.0, None)

    try:
        # invert_cochleagram returns (inv_signal, inv_coch). We want the waveform.
        result = _pyc.invert_cochleagram(
            env_full,
            sr,
            n=n,
            low_lim=COCH_LOW_HZ,
            hi_lim=min(COCH_HIGH_HZ, sr // 2 - 1),
            sample_factor=1,
            n_iter=n_iter,
            strict=False,
        )
        wav = result[0] if isinstance(result, tuple) else result
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        peak = np.max(np.abs(wav)) + 1e-8
        return (wav / peak).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"[cochleagram] iterative inversion failed ({e}); using parametric fallback")
        return parametric_synthesis(env_full, sr)


def parametric_synthesis(envelopes: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """One-pass envelope-modulated white noise. Worse than iterative but always runs."""
    n_channels, n_samples = envelopes.shape
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(n_samples).astype(np.float32)

    # ERB-spaced band edges from low to high.
    edges = np.linspace(COCH_LOW_HZ, min(COCH_HIGH_HZ, sr // 2 - 1), n_channels + 1)
    out = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_channels):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        b, a = sps.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
        sub = sps.filtfilt(b, a, noise).astype(np.float32)
        out += sub * envelopes[i]
    peak = np.max(np.abs(out)) + 1e-8
    return (out / peak).astype(np.float32)


def round_trip_check(wav_path: Path, out_dir: Path) -> None:
    """Forward -> invert -> save .wav. Listen to verify the rep is usable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    waveform, sr = sf.read(str(wav_path))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    waveform = waveform.astype(np.float32)
    if sr != TARGET_SR:
        waveform = sps.resample(waveform, int(len(waveform) * TARGET_SR / sr))
        sr = TARGET_SR

    coch = waveform_to_cochleagram(waveform, sr)
    print(f"cochleagram shape: {coch.shape}")
    recon = cochleagram_to_waveform(coch, sr)

    sf.write(str(out_dir / f"{wav_path.stem}_orig.wav"), waveform, sr)
    sf.write(str(out_dir / f"{wav_path.stem}_recon.wav"), recon, sr)
    print(f"wrote round-trip outputs to {out_dir}")


if __name__ == "__main__":
    from config import OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path, help="audio file to round-trip")
    args = parser.parse_args()
    round_trip_check(args.wav, OUTPUT_DIR / "round_trip")
