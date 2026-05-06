"""Silent video clip -> predicted .wav.

Variants:
  --variant v1            (default)  V1 MLP over ResNet18 features.
  --variant three_stream             ThreeStreamV2A (ResNet50 + R(2+1)D-18 +
                                     RAFT flow + Transformer fusion).

Usage:
  python src/inference.py path/to/video.mp4 --onset-time 1.23 \
      --out outputs/predicted.wav
  python src/inference.py path/to/video.mp4 --onset-time 1.23 \
      --variant three_stream --checkpoint checkpoints/three_stream_app-m3d-flow.pt

If --onset-time is omitted, runs onset detection on the audio track of the
input.
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
    CLIP_FRAMES,
    ENVELOPE_SR,
    ONSET_FRAME_INDEX,
    OUTPUT_DIR,
    PCA_DIM,
    TARGET_SR,
    TS_D_MODEL,
    VIDEO_FPS,
)
from dataset import read_video_clip
from model import PCAState, V1MLP


def _load_pca(blob: dict) -> PCAState:
    p = blob["pca"]
    return PCAState(
        mean=p["mean"],
        components=p["components"],
        explained_variance=p["explained_variance"],
        target_shape=tuple(p["target_shape"]),
    )


# ---------------------------------------------------------------------------
# V1 path (unchanged behaviour, just refactored to return a cochleagram).
# ---------------------------------------------------------------------------

def _v1_predict_coch(
    video_path: Path,
    onset_time: float,
    checkpoint: Path,
    device: torch.device,
) -> np.ndarray:
    from visual_features import _build_resnet18, extract_clip_features

    # weights_only=False: checkpoint includes a Python dict for the PCA payload,
    # not just tensors. Safe here because we control what wrote it (train.py).
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = V1MLP(
        n_frames=cfg["n_frames"],
        feat_dim=cfg["feat_dim"],
        out_dim=cfg["pca_dim"],
    ).to(device).eval()
    model.load_state_dict(blob["model_state"])
    pca = _load_pca(blob)

    feature_model = _build_resnet18(device)
    onset_frame = int(round(onset_time * VIDEO_FPS))
    start = max(0, onset_frame - ONSET_FRAME_INDEX)
    frames = read_video_clip(video_path, start, CLIP_FRAMES)
    feats = extract_clip_features(frames, feature_model, device)

    feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model(feats_t).cpu().numpy()[0]

    coch_flat = pca.inverse_transform(z[None])[0]
    return np.clip(coch_flat.reshape(pca.target_shape).astype(np.float32), 0.0, None)


# ---------------------------------------------------------------------------
# Three-stream path. Mirrors the train-time forward but for a single clip.
# ---------------------------------------------------------------------------

def _three_stream_predict_coch(
    video_path: Path,
    onset_time: float,
    checkpoint: Path,
    device: torch.device,
) -> np.ndarray:
    from streams import (
        FrozenStreams,
        build_raft,
        compute_flow,
        compute_frozen_features,
        resize_and_crop,
    )
    from three_stream_model import ModelConfig, PCAHead, ThreeStreamV2A

    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_cfg"])
    streams = tuple(cfg.streams)
    head = PCAHead(d_model=cfg.d_model, out_dim=PCA_DIM)
    model = ThreeStreamV2A(head, streams=streams, cfg=cfg).to(device).eval()
    model.load_state_dict(blob["model_state"])
    pca = _load_pca(blob)

    onset_frame = int(round(onset_time * VIDEO_FPS))
    start = max(0, onset_frame - ONSET_FRAME_INDEX)
    frames_native = read_video_clip(video_path, start, CLIP_FRAMES)
    frames_224 = resize_and_crop(frames_native)

    raft = build_raft(device)
    flow_fp16 = compute_flow(frames_224, raft, device)        # (T-1, 2, H_f, W_f)
    flow_t = torch.from_numpy(flow_fp16.astype(np.float32)).unsqueeze(0).to(device)

    frozen = FrozenStreams.build(device)
    frames_chw = (frames_224.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    frames_t = torch.from_numpy(frames_chw).unsqueeze(0).to(device)  # (1, T, 3, H, W)

    with torch.no_grad():
        app, m3d = compute_frozen_features(frames_t, frozen)
        pred = model(appearance=app, motion3d=m3d, flow_field=flow_t)
    z = pred.cpu().numpy()[0]
    coch_flat = pca.inverse_transform(z[None])[0]
    return np.clip(coch_flat.reshape(pca.target_shape).astype(np.float32), 0.0, None)


# ---------------------------------------------------------------------------
# Top-level: predict cochleagram -> synthesise wav -> write to disk.
# ---------------------------------------------------------------------------

def predict_wav(
    video_path: Path,
    onset_time: float,
    checkpoint: Path,
    out_wav: Path,
    variant: str = "v1",
    device: torch.device | None = None,
) -> Path:
    if device is None:
        from cache_visual_features import _pick_device
        device = _pick_device()  # CUDA -> MPS -> CPU
    if variant == "v1":
        coch = _v1_predict_coch(video_path, onset_time, checkpoint, device)
    elif variant == "three_stream":
        coch = _three_stream_predict_coch(video_path, onset_time, checkpoint, device)
    else:
        raise ValueError(f"unknown variant {variant!r}; expected 'v1' or 'three_stream'")

    wav = cochleagram_to_waveform(coch, sr=TARGET_SR, envelope_sr=ENVELOPE_SR)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), wav, TARGET_SR)
    print(f"wrote {out_wav}  ({len(wav)/TARGET_SR:.2f}s @ {TARGET_SR} Hz)  [{variant}]")
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


def _default_checkpoint(variant: str) -> Path:
    if variant == "v1":
        return CHECKPOINT_DIR / "v1.pt"
    if variant == "three_stream":
        return CHECKPOINT_DIR / "three_stream_app-m3d-flow.pt"
    raise ValueError(f"unknown variant {variant!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--onset-time", type=float, default=None)
    parser.add_argument("--variant", type=str, default="v1", choices=("v1", "three_stream"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "predicted.wav")
    args = parser.parse_args()

    onset_time = args.onset_time
    if onset_time is None:
        onset_time = _detect_first_onset(args.video)
        print(f"auto-detected onset_time = {onset_time:.3f}s")
    checkpoint = args.checkpoint or _default_checkpoint(args.variant)
    predict_wav(args.video, onset_time, checkpoint, args.out, variant=args.variant)
