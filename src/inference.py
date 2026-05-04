"""End-of-Day-1 deliverable: silent video clip -> predicted .wav.

Usage:
  python src/inference.py path/to/video.mp4 --onset-time 1.23 \
      --out outputs/predicted.wav

If --onset-time is omitted, runs onset detection on the audio track of the
input (so the same script works for sanity checks during development).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from cochleagram import cochleagram_to_waveform
from config import (
    CHECKPOINT_DIR,
    CLIP_DURATION_S,
    CLIP_FRAMES,
    ENVELOPE_SR,
    ONSET_FRAME_INDEX,
    OUTPUT_DIR,
    PCA_DIM,
    PER_FRAME_FEAT_DIM,
    TARGET_SR,
    VIDEO_FPS,
)
from dataset import read_video_clip
from model import PCAState, V1MLP
from visual_features import _build_resnet18, extract_clip_features


def load_checkpoint(path: Path, device: torch.device) -> tuple[V1MLP, PCAState]:
    blob = torch.load(path, map_location=device)
    cfg = blob["config"]
    model = V1MLP(
        n_frames=cfg["n_frames"],
        feat_dim=cfg["feat_dim"],
        out_dim=cfg["pca_dim"],
    ).to(device).eval()
    model.load_state_dict(blob["model_state"])
    p = blob["pca"]
    pca = PCAState(
        mean=p["mean"],
        components=p["components"],
        explained_variance=p["explained_variance"],
        target_shape=tuple(p["target_shape"]),
    )
    return model, pca


def predict_wav(
    video_path: Path,
    onset_time: float,
    checkpoint: Path,
    out_wav: Path,
    device: torch.device | None = None,
) -> Path:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, pca = load_checkpoint(checkpoint, device)
    feature_model = _build_resnet18(device)

    onset_frame = int(round(onset_time * VIDEO_FPS))
    start = max(0, onset_frame - ONSET_FRAME_INDEX)
    frames = read_video_clip(video_path, start, CLIP_FRAMES)
    feats = extract_clip_features(frames, feature_model, device)

    feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model(feats_t).cpu().numpy()[0]

    coch_flat = pca.inverse_transform(z[None])[0]
    coch = coch_flat.reshape(pca.target_shape).astype(np.float32)
    coch = np.clip(coch, 0.0, None)

    wav = cochleagram_to_waveform(coch, sr=TARGET_SR, envelope_sr=ENVELOPE_SR)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), wav, TARGET_SR)
    print(f"wrote {out_wav}  ({len(wav)/TARGET_SR:.2f}s @ {TARGET_SR} Hz)")
    return out_wav


def _detect_first_onset(video_path: Path) -> float:
    """Fallback when --onset-time is not provided. Looks for a sibling _denoised.wav."""
    from onset_detection import detect_onsets
    from dataset import load_audio

    candidates = [
        video_path.with_name(f"{video_path.stem}_denoised.wav"),
        video_path.with_suffix(".wav"),
    ]
    audio_path = next((p for p in candidates if p.exists()), None)
    if audio_path is None:
        raise FileNotFoundError(
            "No --onset-time given and no sibling audio file found for onset detection."
        )
    onsets = detect_onsets(load_audio(audio_path))
    if len(onsets) == 0:
        raise RuntimeError("No onsets detected in companion audio; pass --onset-time manually.")
    return float(onsets[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--onset-time", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "v1.pt")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "predicted.wav")
    args = parser.parse_args()

    onset_time = args.onset_time
    if onset_time is None:
        onset_time = _detect_first_onset(args.video)
        print(f"auto-detected onset_time = {onset_time:.3f}s")
    predict_wav(args.video, onset_time, args.checkpoint, args.out)
