"""Generate writeup figures from trained checkpoints and eval JSONs.

Three figures, each independently runnable. All write to figures/ (gitignored,
copy out the ones you use into your report's images/ dir).

Usage:
    # all three (skips any whose inputs are missing)
    python src/make_figures.py

    # individually
    python src/make_figures.py --loss-curve
    python src/make_figures.py --cochleagrams --n-examples 4
    python src/make_figures.py --metrics-bar --metrics outputs/eval_greatest_hits/metrics.json
    python src/make_figures.py --metrics-bar --metrics outputs/eval_zero_shot/metrics.json

The figures:
  - loss_curve.png:           train/val MSE per epoch from history JSON.
  - cochleagram_comparison.png: 2x4 grid, real (top) vs predicted (bottom)
                                cochleagrams for 4 held-out test clips.
                                Mimics Owens et al. 2016 Figure 8.
  - metrics_bar.png:          grouped bar chart of loudness MAE + centroid
                                MSE for model + baselines, from a metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import CHECKPOINT_DIR, FIGURE_DIR, OUTPUT_DIR


# ---------------------------------------------------------------------------
# 1. Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(history_path: Path, out_path: Path) -> None:
    history = json.loads(history_path.read_text())
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h.get("val_loss") for h in history]

    val_mask = [v is not None and not (isinstance(v, float) and np.isnan(v)) for v in val]
    val_epochs = [e for e, m in zip(epochs, val_mask) if m]
    val_loss = [v for v, m in zip(val, val_mask) if m]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(epochs, train, "o-", label="train", color="C0", lw=1.5, ms=4)
    if val_loss:
        ax.plot(val_epochs, val_loss, "s-", label="val", color="C3", lw=1.5, ms=5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (PCA-cochleagram)")
    ax.set_title(f"Training history — {history_path.stem}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# 2. Cochleagram comparison
# ---------------------------------------------------------------------------

def _load_three_stream_for_inference(checkpoint: Path, device):
    import torch
    from model import PCAState
    from three_stream_model import ModelConfig, PCAHead, ThreeStreamV2A

    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_cfg"])
    head = PCAHead(d_model=cfg.d_model, out_dim=blob["pca"]["components"].shape[0])
    model = ThreeStreamV2A(head, streams=tuple(cfg.streams), cfg=cfg).to(device).eval()
    model.load_state_dict(blob["model_state"])
    pca = PCAState(
        mean=blob["pca"]["mean"],
        components=blob["pca"]["components"],
        explained_variance=blob["pca"]["explained_variance"],
        target_shape=tuple(blob["pca"]["target_shape"]),
    )
    return model, pca


def plot_cochleagram_comparison(
    checkpoint: Path,
    out_path: Path,
    n_examples: int = 4,
    seed: int = 0,
) -> None:
    import torch
    from cache_visual_features import _pick_device
    from config import CLIP_DURATION_S, CLIP_FRAMES, ENVELOPE_SR, ONSET_FRAME_INDEX
    from dataset import ThreeStreamGHDataset, load_clip_index
    from streams import FrozenStreams, compute_frozen_features

    device = _pick_device()
    print(f"device: {device}", flush=True)

    model, pca = _load_three_stream_for_inference(checkpoint, device)
    frozen = FrozenStreams.build(device)

    _, test_clips = load_clip_index()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test_clips), size=n_examples, replace=False)
    selected = [test_clips[i] for i in idx]

    onset_frac = ONSET_FRAME_INDEX / CLIP_FRAMES
    ds = ThreeStreamGHDataset(
        selected, ENVELOPE_SR, CLIP_DURATION_S, onset_frac, temporal_jitter=0,
    )

    real_cochs: list[np.ndarray] = []
    pred_cochs: list[np.ndarray] = []
    titles: list[str] = []
    use_amp = device.type == "cuda"
    autocast_ctx = (
        torch.amp.autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)
    )

    for i in range(len(ds)):
        sample = ds[i]
        clip = selected[i]
        frames = sample["frames"]            # uint8 (T, H, W, 3)
        flow = sample["flow"]                # fp16 (T-1, 2, H, W)
        true = sample["cochleagram"]
        frames_t = (
            torch.from_numpy((frames.astype(np.float32) / 255.0).transpose(0, 3, 1, 2))
            .unsqueeze(0).to(device)
        )
        flow_t = torch.from_numpy(flow.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad(), autocast_ctx:
            app, m3d = compute_frozen_features(frames_t, frozen)
            z = model(appearance=app, motion3d=m3d, flow_field=flow_t).float().cpu().numpy()[0]
        coch_flat = pca.inverse_transform(z[None])[0]
        pred = np.clip(coch_flat.reshape(pca.target_shape), 0.0, None).astype(np.float32)
        real_cochs.append(true)
        pred_cochs.append(pred)
        titles.append(f"{clip.entry.video_id[:14]}\n{clip.material or ''}")

    # Shared colour scale across all panels (real + predicted) so visual
    # intensity differences mean what they look like.
    vmin, vmax = 0.0, max(c.max() for c in real_cochs + pred_cochs)

    fig, axes = plt.subplots(2, n_examples, figsize=(3.0 * n_examples, 4.5))
    if n_examples == 1:
        axes = axes[:, None]
    for j, (real, pred, title) in enumerate(zip(real_cochs, pred_cochs, titles)):
        for row, (ax, coch, label) in enumerate(zip(
            axes[:, j], (real, pred), ("real", "predicted")
        )):
            im = ax.imshow(
                coch, aspect="auto", origin="lower",
                vmin=vmin, vmax=vmax, cmap="magma",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{label}\nfreq channel", fontsize=9)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.set_label("compressed envelope (^0.3)", fontsize=8)
    fig.suptitle("Real vs predicted cochleagrams (held-out Greatest Hits clips)", fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# 3. Metrics bar chart
# ---------------------------------------------------------------------------

def plot_metrics_bar(metrics_path: Path, out_path: Path) -> None:
    payload = json.loads(metrics_path.read_text())
    metrics = payload["metrics"]

    method_order = [
        ("model", "three-stream"),
        ("three_stream", "three-stream"),  # eval_greatest_hits uses this key
        ("v1", "V1"),
        ("baseline_random", "random"),
        ("baseline_mean", "mean"),
    ]
    methods: list[str] = []
    labels: list[str] = []
    for key, lab in method_order:
        if key in metrics:
            methods.append(key)
            labels.append(lab)

    loud = [metrics[m]["loudness_mae"] for m in methods]
    cmse = [metrics[m]["centroid_mse"] for m in methods]

    fig, (ax_l, ax_c) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    colors = ["#1f77b4", "#ff7f0e", "#7f7f7f", "#bcbcbc"][: len(methods)]
    xs = np.arange(len(methods))

    ax_l.bar(xs, loud, color=colors, edgecolor="black", lw=0.5)
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_l.set_ylabel("loudness MAE  (lower is better)")
    ax_l.set_title("Peak loudness error")
    ax_l.grid(axis="y", alpha=0.3)
    for x, y in zip(xs, loud):
        ax_l.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    ax_c.bar(xs, cmse, color=colors, edgecolor="black", lw=0.5)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_c.set_ylabel("centroid MSE  (lower is better)")
    ax_c.set_title("Spectral centroid error")
    ax_c.grid(axis="y", alpha=0.3)
    for x, y in zip(xs, cmse):
        ax_c.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    title_src = metrics_path.parent.name  # eval_greatest_hits or eval_zero_shot
    fig.suptitle(f"Metrics — {title_src} (N={list(metrics.values())[0]['n_clips']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-curve", action="store_true")
    parser.add_argument("--cochleagrams", action="store_true")
    parser.add_argument("--metrics-bar", action="store_true")

    parser.add_argument(
        "--history", type=Path,
        default=CHECKPOINT_DIR / "three_stream_app-m3d-flow_history.json",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=CHECKPOINT_DIR / "three_stream_app-m3d-flow.pt",
    )
    parser.add_argument(
        "--metrics", type=Path,
        default=OUTPUT_DIR / "eval_greatest_hits" / "metrics.json",
    )
    parser.add_argument("--n-examples", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # If no individual flag was passed, run all three (skipping whichever inputs are missing).
    run_all = not (args.loss_curve or args.cochleagrams or args.metrics_bar)

    if (run_all or args.loss_curve):
        if args.history.exists():
            plot_loss_curve(args.history, args.out_dir / "loss_curve.png")
        else:
            print(f"skipping loss curve: {args.history} not found")

    if (run_all or args.cochleagrams):
        if args.checkpoint.exists():
            plot_cochleagram_comparison(
                args.checkpoint,
                args.out_dir / "cochleagram_comparison.png",
                n_examples=args.n_examples,
            )
        else:
            print(f"skipping cochleagram comparison: {args.checkpoint} not found")

    if (run_all or args.metrics_bar):
        if args.metrics.exists():
            tag = args.metrics.parent.name  # eval_greatest_hits or eval_zero_shot
            plot_metrics_bar(args.metrics, args.out_dir / f"metrics_bar_{tag}.png")
        else:
            print(f"skipping metrics bar: {args.metrics} not found")


if __name__ == "__main__":
    main()
