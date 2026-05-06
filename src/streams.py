"""Three-stream V2A: pretrained backbones + cache I/O.

Cache layout (option (c) — flow-only frozen; app/3D recomputed at train time
so we can augment in pixel space):

    <vid>_f<onset>_clip.npz   JPEG-encoded frames, CACHE_FRAMES_STORED of them
                              (~25 KB/frame at q=90, ~375 KB/clip at T=17)
    <vid>_f<onset>_flow.npy   fp16 (CACHE_FRAMES_STORED-1, 2,
                                   CACHE_FLOW_SIZE, CACHE_FLOW_SIZE)

The cache window is 17 frames (CLIP_FRAMES + 2*MAX_TEMPORAL_JITTER) to allow
±1-frame temporal jitter at train time without re-decoding the video.

At train time:
    1. Load (frames_uint8, flow_field) from cache, both 17 frames long.
    2. Pick a 15-frame window (random offset in [-J..J] for jitter, 0 for eval).
    3. Apply pixel-space augmentation in float space — hflip (also flips flow_x
       sign and spatially mirrors flow), color jitter, brightness, etc.
    4. Call `compute_frozen_features(clip_float, frozen)` -> (app, m3d).
    5. Pass (app, m3d, flow_field) to ThreeStreamV2A; the model contains a
       learnable flow encoder over the raw flow tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from config import (
    APP_INPUT_SIZE,
    CACHE_DIR,
    CACHE_FLOW_SIZE,
    CACHE_FRAME_SIZE,
    CACHE_FRAMES_STORED,
    JPEG_QUALITY,
    MOTION3D_INPUT_SIZE,
    RAFT_INPUT_SIZE,
)

if TYPE_CHECKING:
    # Type-only import to avoid a circular dependency: dataset.py imports from
    # streams.py at runtime when it builds ThreeStreamGHDataset.
    from dataset import ClipIndex

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


# ---------------------------------------------------------------------------
# Frozen backbones
# ---------------------------------------------------------------------------

def _freeze(m: nn.Module) -> nn.Module:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_appearance_backbone(device: torch.device) -> nn.Module:
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity()  # 2048-d avgpool output
    return _freeze(m).to(device)


def build_motion3d_backbone(device: torch.device) -> nn.Module:
    m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    m.fc = nn.Identity()  # 512-d avgpool output
    return _freeze(m).to(device)


def build_raft(device: torch.device) -> nn.Module:
    m = raft_small(weights=Raft_Small_Weights.DEFAULT)
    return _freeze(m).to(device)


@dataclass
class FrozenStreams:
    """Frozen backbones used at *train time*. RAFT is loaded separately at
    cache time via `build_raft` since it isn't needed once flow is cached."""
    appearance: nn.Module
    motion3d: nn.Module

    @classmethod
    def build(cls, device: torch.device) -> "FrozenStreams":
        return cls(
            appearance=build_appearance_backbone(device),
            motion3d=build_motion3d_backbone(device),
        )


# ---------------------------------------------------------------------------
# Train-time forward (frozen part). Run once per batch on the GPU.
# ---------------------------------------------------------------------------

def _normalize(x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def compute_frozen_features(
    clip_float: torch.Tensor,    # (B, T, 3, H, W) float in [0, 1]
    frozen: FrozenStreams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen ResNet50 + R(2+1)D-18 forward passes.

    Input is float in [0, 1] so the train script can apply pixel-space
    augmentation (color jitter, brightness, etc.) before this call.

    Returns:
        app: (B, T, APP_FEAT_DIM)
        m3d: (B, MOTION3D_FEAT_DIM)
    """
    assert clip_float.is_floating_point(), "expected float input in [0, 1]"
    B, T, C, H, W = clip_float.shape
    assert C == 3, f"expected 3-channel input, got {C}"

    # --- Stream A: per-frame appearance, ResNet50 @ APP_INPUT_SIZE -----------
    x_app = clip_float.reshape(B * T, 3, H, W)
    if H != APP_INPUT_SIZE:
        x_app = F.interpolate(x_app, size=APP_INPUT_SIZE, mode="bilinear", align_corners=False)
    x_app = _normalize(x_app, IMAGENET_MEAN, IMAGENET_STD)
    app = frozen.appearance(x_app).view(B, T, -1)        # (B, T, 2048)

    # --- Stream B: clip-level 3D motion, R(2+1)D-18 @ MOTION3D_INPUT_SIZE ----
    x_3d = clip_float.reshape(B * T, 3, H, W)
    if H != MOTION3D_INPUT_SIZE:
        x_3d = F.interpolate(x_3d, size=MOTION3D_INPUT_SIZE, mode="bilinear", align_corners=False)
    x_3d = _normalize(x_3d, KINETICS_MEAN, KINETICS_STD)
    x_3d = x_3d.view(B, T, 3, MOTION3D_INPUT_SIZE, MOTION3D_INPUT_SIZE).permute(0, 2, 1, 3, 4)
    m3d = frozen.motion3d(x_3d)                          # (B, 512)

    return app, m3d


# ---------------------------------------------------------------------------
# Cache-time RAFT pass + cache I/O
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_flow(
    frames_uint8: np.ndarray,  # (T, H, W, 3) uint8 RGB, square
    raft: nn.Module,
    device: torch.device,
    target_size: int = CACHE_FLOW_SIZE,
    raft_input_size: int = RAFT_INPUT_SIZE,
) -> np.ndarray:
    """RAFT on consecutive pairs at `raft_input_size`, then bilinear-down to
    `target_size` (rescaling magnitudes accordingly). Returns fp16
    (T-1, 2, target_size, target_size).

    Used both by the cache driver (with T=CACHE_FRAMES_STORED) and at inference
    (with T=CLIP_FRAMES). compute_flow_cache is a thin wrapper that asserts the
    cache-time frame count.
    """
    assert frames_uint8.shape[0] >= 2, f"need at least 2 frames, got {frames_uint8.shape[0]}"
    assert frames_uint8.shape[1] == frames_uint8.shape[2], (
        f"square frames required, got {frames_uint8.shape}"
    )
    x = torch.from_numpy(frames_uint8).to(device).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != raft_input_size:
        x = F.interpolate(x, size=raft_input_size, mode="bilinear", align_corners=False)
    raft_in = (x * 2.0) - 1.0
    flows = raft(raft_in[:-1], raft_in[1:])
    flow_field = flows[-1] if isinstance(flows, list) else flows  # (T-1, 2, H, W)
    if flow_field.shape[-1] != target_size:
        # Flow magnitudes scale with spatial resolution; rescale when resizing.
        scale = target_size / flow_field.shape[-1]
        flow_field = F.interpolate(flow_field, size=target_size, mode="bilinear", align_corners=False) * scale
    return flow_field.cpu().numpy().astype(np.float16)


@torch.no_grad()
def compute_flow_cache(
    clip_uint8_cached: np.ndarray,  # (CACHE_FRAMES_STORED, H, W, 3) uint8 RGB
    raft: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Cache-time wrapper: asserts CACHE_FRAMES_STORED frames and delegates to
    compute_flow."""
    assert clip_uint8_cached.shape[0] == CACHE_FRAMES_STORED, (
        f"cache write expects {CACHE_FRAMES_STORED} frames, got {clip_uint8_cached.shape[0]}"
    )
    return compute_flow(clip_uint8_cached, raft, device)


def resize_and_crop(
    frames: np.ndarray,
    target_size: int = CACHE_FRAME_SIZE,
    resize_short_to: int = 256,
) -> np.ndarray:
    """Resize short side to `resize_short_to`, center-crop to (target_size,
    target_size). Same algorithm as cache_visual_features._read_full_video, so
    inference-time framing matches the cached training-time framing."""
    if frames.size == 0:
        return frames
    h, w = frames.shape[1:3]
    if h == target_size and w == target_size:
        return frames
    if h < w:
        new_h, new_w = resize_short_to, int(round(w * resize_short_to / h))
    else:
        new_h, new_w = int(round(h * resize_short_to / w)), resize_short_to
    out = np.empty((frames.shape[0], target_size, target_size, 3), dtype=np.uint8)
    ch = (new_h - target_size) // 2
    cw = (new_w - target_size) // 2
    for i, f in enumerate(frames):
        resized = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        out[i] = resized[ch : ch + target_size, cw : cw + target_size]
    return out


def stream_cache_paths(clip: "ClipIndex") -> dict[str, Path]:
    base = CACHE_DIR / f"{clip.entry.video_id}_f{clip.onset_frame:06d}"
    return {
        "clip": base.with_name(base.name + "_clip.npz"),
        "flow": base.with_name(base.name + "_flow.npy"),
    }


def streams_cached(clip: "ClipIndex") -> bool:
    return all(p.exists() for p in stream_cache_paths(clip).values())


def encode_jpeg_frame(frame_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H, W, 3) -> 1D uint8 array of JPEG bytes (q=JPEG_QUALITY)."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed for JPEG cache write")
    return buf


def decode_jpeg_frame(buf: np.ndarray) -> np.ndarray:
    """1D uint8 array of JPEG bytes -> RGB uint8 (H, W, 3)."""
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed for JPEG cache read")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_streams_to_paths(
    clip_path: Path,
    flow_path: Path,
    frames_uint8: np.ndarray,
    flow_fp16: np.ndarray,
) -> None:
    """Path-based variant of save_streams. Used by both Greatest Hits and
    EPIC-Sounds; the dataset-specific layer just picks the paths."""
    assert frames_uint8.dtype == np.uint8, frames_uint8.dtype
    assert frames_uint8.shape == (
        CACHE_FRAMES_STORED, CACHE_FRAME_SIZE, CACHE_FRAME_SIZE, 3
    ), frames_uint8.shape
    assert flow_fp16.dtype == np.float16, flow_fp16.dtype
    assert flow_fp16.shape == (
        CACHE_FRAMES_STORED - 1, 2, CACHE_FLOW_SIZE, CACHE_FLOW_SIZE
    ), flow_fp16.shape
    payload = {f"f{i:02d}": encode_jpeg_frame(frames_uint8[i]) for i in range(CACHE_FRAMES_STORED)}
    # JPEG bytes are already compressed; np.savez (uncompressed) is faster than
    # np.savez_compressed and only marginally larger here.
    np.savez(clip_path, **payload)
    np.save(flow_path, flow_fp16)


def load_streams_from_paths(clip_path: Path, flow_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Path-based variant of load_streams."""
    with np.load(clip_path) as f:
        frames = np.stack(
            [decode_jpeg_frame(f[f"f{i:02d}"]) for i in range(CACHE_FRAMES_STORED)],
            axis=0,
        )
    flow = np.load(flow_path)
    return frames, flow


def save_streams(clip: "ClipIndex", frames_uint8: np.ndarray, flow_fp16: np.ndarray) -> None:
    paths = stream_cache_paths(clip)
    save_streams_to_paths(paths["clip"], paths["flow"], frames_uint8, flow_fp16)


def load_streams(clip: "ClipIndex") -> tuple[np.ndarray, np.ndarray]:
    """Returns (frames_rgb_uint8, flow_fp16) at full cache size — no jitter
    applied. The dataset class does jitter window selection."""
    paths = stream_cache_paths(clip)
    return load_streams_from_paths(paths["clip"], paths["flow"])
