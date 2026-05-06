"""Three-stream V2A model: 2D appearance + 3D motion + learnable optical flow.

Inputs (per clip):
    appearance:  (B, T, APP_FEAT_DIM)        frozen ResNet50, run by streams.py
    motion3d:    (B, MOTION3D_FEAT_DIM)      frozen R(2+1)D-18, run by streams.py
    flow_field:  (B, T-1, 2, H_f, W_f)       raw RAFT flow, encoded *here*

Architecture:
    LearnableFlowEncoder (small CNN, trained from scratch) → (B, T-1, F_flow)
    Replicate flow[0] as the t=0 padding so flow has T tokens aligned with
    appearance — keeps the t=0 token semantically meaningful (especially in
    the flow-only ablation, where zero-pad would make t=0 a learned constant).

    Per-frame token = Linear(concat(app_t, flow_t)) → d_model
    CLS-like motion token = Linear(motion3d) → d_model
    Onset embedding: a single learnable vector added to the onset-frame token.
    Learned positional embedding over (1 + T) tokens.

    Transformer encoder (2 layers) over [motion_cls, frame_0..frame_{T-1}].

    Readout: windowed mean-pool over frame tokens at [onset±ONSET_WINDOW_HALF],
    *not* the CLS token. The CLS token still feeds attention into the impact
    window via the encoder; we just don't read out from it. This reflects that
    the audio target is concentrated near the impact, not summarised globally.

Audio heads:
    PCAHead: Linear(d_model → PCA_DIM). Drop-in replacement for the V1 target.
    MelHead: STUB. Per-frame readout is not yet implemented; this head is wrong
        if used as-is. Replace with a per-token decoder before training mel.

Stream selection (real ablation, not input zeroing):
    Pass `streams=("app", "m3d", "flow")` or any subset to the constructor.
    Only the included projections are built; ablation rows = separate trainings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from config import (
    APP_FEAT_DIM,
    CLIP_FRAMES,
    FLOW_ENCODER_DIM,
    MOTION3D_FEAT_DIM,
    ONSET_FRAME_INDEX,
    ONSET_WINDOW_HALF,
    PCA_DIM,
    TS_D_MODEL,
    TS_DROPOUT,
    TS_N_HEADS,
    TS_N_LAYERS,
)

VALID_STREAMS = ("app", "m3d", "flow")


@dataclass
class ModelConfig:
    """Persisted alongside the checkpoint so ablations are reproducible."""
    streams: tuple[str, ...]
    n_frames: int = CLIP_FRAMES
    d_model: int = TS_D_MODEL
    n_heads: int = TS_N_HEADS
    n_layers: int = TS_N_LAYERS
    dropout: float = TS_DROPOUT
    flow_dim: int = FLOW_ENCODER_DIM
    onset_idx: int = ONSET_FRAME_INDEX
    onset_window_half: int = ONSET_WINDOW_HALF


# ---------------------------------------------------------------------------
# Learnable flow encoder
# ---------------------------------------------------------------------------

class LearnableFlowEncoder(nn.Module):
    """Small from-scratch CNN over raw 2-channel flow fields.

    Input:  (B, T-1, 2, H, W)
    Output: (B, T-1, FLOW_ENCODER_DIM)
    """

    def __init__(self, out_dim: int = FLOW_ENCODER_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        B, Tm1, C, H, W = flow.shape
        z = self.net(flow.reshape(B * Tm1, C, H, W))  # (B*(T-1), out_dim)
        return z.view(B, Tm1, -1)


# ---------------------------------------------------------------------------
# Audio heads
# ---------------------------------------------------------------------------

class PCAHead(nn.Module):
    """Pool readout already done in the model; head just projects to PCA dim."""
    def __init__(self, d_model: int = TS_D_MODEL, out_dim: int = PCA_DIM):
        super().__init__()
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc(pooled)


class MelHead(nn.Module):
    """STUB. Per-frame mel decoding requires per-token readout — not yet wired.

    Use only after replacing the windowed-pool readout in ThreeStreamV2A with a
    per-frame readout (or adding a separate token-sequence forward path).
    """
    def __init__(self, d_model: int = TS_D_MODEL, n_mels: int = 80, n_time: int = 32):
        super().__init__()
        raise NotImplementedError(
            "MelHead requires per-frame readout. Implement before training mel; "
            "the current ThreeStreamV2A.forward returns a single pooled vector."
        )


# ---------------------------------------------------------------------------
# Three-stream fusion model
# ---------------------------------------------------------------------------

class ThreeStreamV2A(nn.Module):
    def __init__(
        self,
        head: nn.Module,
        streams: Sequence[str] = VALID_STREAMS,
        cfg: ModelConfig | None = None,
    ):
        super().__init__()
        streams = tuple(streams)
        for s in streams:
            if s not in VALID_STREAMS:
                raise ValueError(f"unknown stream {s!r}; expected one of {VALID_STREAMS}")
        if not streams:
            raise ValueError("at least one stream must be enabled")
        self.streams = streams
        self.cfg = cfg or ModelConfig(streams=streams)
        d = self.cfg.d_model
        T = self.cfg.n_frames

        # Frame token = projection of whichever per-frame streams are enabled.
        per_frame_in = 0
        if "app" in streams:
            per_frame_in += APP_FEAT_DIM
        if "flow" in streams:
            self.flow_encoder = LearnableFlowEncoder(out_dim=self.cfg.flow_dim)
            per_frame_in += self.cfg.flow_dim
        if per_frame_in == 0:
            # m3d-only model: synthesize per-frame tokens from a learned embedding.
            self.frame_proj = None
            self.frame_token = nn.Parameter(torch.zeros(1, T, d))
            nn.init.trunc_normal_(self.frame_token, std=0.02)
        else:
            self.frame_proj = nn.Linear(per_frame_in, d)

        # CLS-like motion3d token (only built if the m3d stream is enabled).
        if "m3d" in streams:
            self.cls_proj = nn.Linear(MOTION3D_FEAT_DIM, d)
        else:
            # Neutral learned CLS token when motion3d is ablated.
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Onset embedding — added only at the impact-frame token.
        self.onset_embed = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.onset_embed, std=0.02)

        # Positional embedding over (cls + T) tokens.
        self.pos = nn.Parameter(torch.zeros(1, T + 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.cfg.n_heads,
            dim_feedforward=d * 4,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.n_layers)
        self.norm = nn.LayerNorm(d)
        self.head = head

    def _frame_tokens(
        self,
        appearance: torch.Tensor | None,
        flow_field: torch.Tensor | None,
    ) -> torch.Tensor:
        # B and T have to come from one of the enabled inputs; m3d-only path uses
        # the learned `frame_token` parameter.
        if self.frame_proj is None:
            B = appearance.size(0) if appearance is not None else flow_field.size(0)
            return self.frame_token.expand(B, -1, -1)

        parts: list[torch.Tensor] = []
        if "app" in self.streams:
            assert appearance is not None, "app stream is enabled; appearance must be provided"
            parts.append(appearance)
        if "flow" in self.streams:
            assert flow_field is not None, "flow stream is enabled; flow_field must be provided"
            flow_per_frame = self.flow_encoder(flow_field)        # (B, T-1, F_flow)
            # t=0 has no preceding frame, so RAFT didn't produce flow there.
            # Replicate the first observed flow embedding rather than zero-pad
            # so t=0 carries a meaningful (if approximate) motion signal — and
            # the flow-only ablation doesn't degenerate to a learned constant.
            pad0 = flow_per_frame[:, :1]
            parts.append(torch.cat([pad0, flow_per_frame], dim=1))  # (B, T, F_flow)

        return self.frame_proj(torch.cat(parts, dim=-1))           # (B, T, d_model)

    def _cls_token(self, motion3d: torch.Tensor | None, B: int, device, dtype) -> torch.Tensor:
        if "m3d" in self.streams:
            assert motion3d is not None, "m3d stream is enabled; motion3d must be provided"
            return self.cls_proj(motion3d).unsqueeze(1)             # (B, 1, d_model)
        return self.cls_token.expand(B, -1, -1).to(device=device, dtype=dtype)

    def forward(
        self,
        appearance: torch.Tensor | None = None,   # (B, T, APP_FEAT_DIM)
        motion3d: torch.Tensor | None = None,     # (B, MOTION3D_FEAT_DIM)
        flow_field: torch.Tensor | None = None,   # (B, T-1, 2, H, W)
    ) -> torch.Tensor:
        frame_tokens = self._frame_tokens(appearance, flow_field)   # (B, T, d)
        B, T, d = frame_tokens.shape

        # Onset embedding: add to the impact-frame token only.
        onset_idx = self.cfg.onset_idx
        onset_mask = torch.zeros(1, T, 1, device=frame_tokens.device, dtype=frame_tokens.dtype)
        onset_mask[0, onset_idx, 0] = 1.0
        frame_tokens = frame_tokens + onset_mask * self.onset_embed

        cls = self._cls_token(motion3d, B, frame_tokens.device, frame_tokens.dtype)
        tokens = torch.cat([cls, frame_tokens], dim=1)              # (B, T+1, d)
        tokens = tokens + self.pos[:, : tokens.size(1)]

        z = self.encoder(tokens)                                    # (B, T+1, d)
        # Frame tokens are at indices 1..T; impact frame is at 1+onset_idx.
        impact = 1 + onset_idx
        half = self.cfg.onset_window_half
        lo = max(1, impact - half)
        hi = min(T + 1, impact + half + 1)
        pooled = self.norm(z[:, lo:hi].mean(dim=1))                 # (B, d)
        return self.head(pooled)
