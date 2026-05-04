"""1.4 Visual feature extraction.

For each frame in a 15-frame clip:
  - RGB feature  : ResNet18 avgpool on the frame                     (512-d)
  - Spacetime    : ResNet18 avgpool on a 3-channel grayscale stack
                   (frames t-1, t, t+1)                              (512-d)
Concatenate -> (15, 1024). Per-clip features cached to disk.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from tqdm import tqdm

from config import CACHE_DIR, CLIP_FRAMES, ONSET_FRAME_INDEX, PER_FRAME_FEAT_DIM
from dataset import ClipIndex, read_video_clip

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_resnet18(device: torch.device) -> nn.Module:
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()  # avgpool output, 512-d
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _grayscale_stack(rgb_frames: np.ndarray, t: int) -> np.ndarray:
    """3-channel image where channels are gray frames (t-1, t, t+1)."""
    n = rgb_frames.shape[0]
    idxs = [max(0, t - 1), t, min(n - 1, t + 1)]
    grays = [cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2GRAY) for i in idxs]
    return np.stack(grays, axis=-1)  # (H, W, 3) uint8


def extract_clip_features(
    rgb_frames: np.ndarray,
    model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """rgb_frames: (T, H, W, 3) uint8. Returns (T, 1024) float32."""
    rgb_batch = torch.stack([_TRANSFORM(f) for f in rgb_frames]).to(device)
    spacetime_batch = torch.stack([
        _TRANSFORM(_grayscale_stack(rgb_frames, t)) for t in range(rgb_frames.shape[0])
    ]).to(device)
    with torch.no_grad():
        rgb_feat = model(rgb_batch).cpu().numpy()
        st_feat = model(spacetime_batch).cpu().numpy()
    return np.concatenate([rgb_feat, st_feat], axis=-1).astype(np.float32)


def _feat_cache_path(clip: ClipIndex) -> Path:
    return CACHE_DIR / f"{clip.entry.video_id}_f{clip.onset_frame:06d}_feat.npy"


def features_for_clip(
    clip: ClipIndex,
    model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    cache = _feat_cache_path(clip)
    if cache.exists():
        return np.load(cache)
    start = clip.onset_frame - ONSET_FRAME_INDEX
    frames = read_video_clip(clip.entry.video_path, start, CLIP_FRAMES)
    feats = extract_clip_features(frames, model, device)
    assert feats.shape == (CLIP_FRAMES, PER_FRAME_FEAT_DIM), feats.shape
    np.save(cache, feats)
    return feats


def precompute_all(clips: list[ClipIndex], device: torch.device | None = None) -> None:
    """One-shot caching pass. Run once; subsequent training reads from disk."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_resnet18(device)
    for clip in tqdm(clips, desc="visual features"):
        features_for_clip(clip, model, device)


if __name__ == "__main__":
    from dataset import build_datasets

    train_ds, test_ds, train_clips, test_clips = build_datasets()
    print(f"{len(train_clips)} train clips, {len(test_clips)} test clips")
    precompute_all(train_clips + test_clips)
