# Visually Indicated Sounds → real-world V2A

Predicting plausible impact audio from silent video. Two model paths in one
repo:

- **V1** — a faithful reproduction of [Owens et al. 2016, *Visually Indicated
  Sounds*](https://arxiv.org/abs/1512.08512): onset detection → 40-channel
  ERB cochleagram → ResNet18 features over a 15-frame window → MLP → PCA-10
  regression target → cochleagram inversion to waveform.
- **Three-stream V2A** — an extension aimed at moving beyond Greatest Hits:
  ResNet50 (ImageNet1K-V2, frozen) + R(2+1)D-18 (Kinetics-400, frozen) +
  RAFT-small flow with a learnable flow encoder, fused by a small Transformer
  with a windowed mean-pool readout around the impact frame, and a PCA head
  on top. Same 15-frame, ±1 temporal-jitter setup; same cochleagram target;
  pixel-space augmentation (hflip + color jitter) at train time.

Both models train from the same clip index off the
[Greatest Hits](https://andrewowens.com/vis/) dataset and are scored against
the same held-out video-level split. The three-stream path is also designed
for cross-dataset zero-shot eval against
[EPIC-Sounds](https://github.com/epic-kitchens/epic-sounds-annotations) impact
classes.

## Why two paths

V1 is the controlled paper-reproduction baseline — small, fast to train,
matches the 2016 paper's regression target so we have a known reference
point. It assumes the world looks like Greatest Hits: a drumstick striking
objects in a tightly framed scene with isolated impact onsets.

Three-stream is the research extension. The frozen pretrained backbones (a
strong 2D appearance encoder + a 3D motion encoder + an explicit optical-flow
stream) provide visual representations that *should* generalize beyond
Greatest Hits. The honest constraint that remains: the visual→audio mapping
is still trained on a 9-hour dataset, so cross-dataset transfer to real-world
video is a hypothesis to be tested, not a guarantee. The
[`eval_zero_shot.py`](src/eval_zero_shot.py) script is built specifically to
make that test the central result.

## Project layout

```
src/
  config.py              # central hyperparams + cache paths
  cochleagram.py         # waveform <-> cochleagram (forward + inversion)
  onset_detection.py     # impact onset times from audio
  dataset.py             # Greatest Hits discovery, ClipIndex, V2AClipDataset protocol
  visual_features.py     # V1: ResNet18 per-frame + spacetime-image features
  cache_visual_features.py  # V1: parallel feature caching driver
  model.py               # V1: PCA fit + V1MLP regressor
  train.py               # V1: training loop
  inference.py           # both: load checkpoint -> predict cochleagram -> wav
                         #       --variant {v1, three_stream}

  streams.py             # three-stream: frozen backbones (ResNet50 / R(2+1)D-18 / RAFT)
                         # + JPEG-in-NPZ cache I/O + train-time frozen forward
  three_stream_model.py  # three-stream: LearnableFlowEncoder + ThreeStreamV2A
                         # (transformer fusion, windowed pool, PCA head)
  cache_streams.py       # three-stream: per-video cache driver (frames + flow)
  train_three_stream.py  # three-stream: AMP training loop, eval-every-N

  epic_sounds_dataset.py # EpicSoundsClipDataset conforming to V2AClipDataset
  select_epic_subset.py  # pick smallest video subset for an N-clip eval
  eval_zero_shot.py      # cross-dataset eval: model + random/mean baselines
                         # loudness MAE + spectral centroid MSE + pearson r

notebooks/
  train_on_colab.ipynb   # Colab path: V1 cells + three-stream cells

docs/
  pipeline.md            # original 2-day Owens repro plan (V1 reference)
  project_state.md       # current architecture + active goals
```

## Quickstart

### V1 path (Greatest Hits reproduction)

```bash
# Install
pip install -r requirements.txt
pip install git+https://github.com/mcdermottLab/pycochleagram.git

# Drop the dataset under data/, then:
python src/cache_visual_features.py     # one-time visual feature cache
python src/train.py --epochs 20         # ~5-10 min on T4
python src/inference.py path/to/video.mp4 --onset-time 1.23 --variant v1
```

### Three-stream path

```bash
# One-time three-stream cache (RAFT flow + JPEG-encoded frames). Resumable.
# ~2 hours on Mac MPS, ~30-60 min on Colab T4.
python src/cache_streams.py

# Train. ~1 hour on Colab A100, ~3-4 hours on T4.
python src/train_three_stream.py \
    --epochs 20 --batch-size 64 --num-workers 2 --eval-every 2

# Inference
python src/inference.py path/to/video.mp4 --onset-time 1.23 --variant three_stream

# Cross-dataset eval (after EPIC-Sounds data is downloaded — see below)
python src/eval_zero_shot.py \
    --checkpoint checkpoints/three_stream_app-m3d-flow.pt \
    --annotations-csv path/to/EPIC_Sounds_validation.csv \
    --videos-dir path/to/EPIC-KITCHENS-100 \
    --max-clips 200 --n-samples 5
```

### Colab

The repo includes [a Colab notebook](notebooks/train_on_colab.ipynb) that
clones the repo, mounts Drive for the cached training bundle, runs training
with checkpoints persisted to Drive, and produces listening samples inline.
The three-stream cells assume a separately-uploaded
`three_stream_bundle.tar` (~14 GB of cached frames + flow fields).

## Datasets

### Greatest Hits (training)

[Owens et al. 2016 dataset](https://andrewowens.com/vis/). ~977 videos of a
drumstick striking objects, with precise onset annotations and material
labels. Used at the video level for the train/test split (75/25). Total
~28,635 valid impact clips after onset filtering and 15-frame-window
boundary checks.

### EPIC-Sounds (zero-shot eval)

[EPIC-Sounds annotations](https://github.com/epic-kitchens/epic-sounds-annotations)
provides timestamped sound-event labels on
[EPIC-KITCHENS-100](https://epic-kitchens.github.io/2024) egocentric kitchen
video. The eval pipeline filters to impact-style classes (`metal-only
collision`, `metal / ceramic collision`, `click`, etc.) — 3,298 of 8,035
validation clips match this filter. For an eval of N clips, download only
the video files needed:

```bash
curl -O https://raw.githubusercontent.com/epic-kitchens/epic-sounds-annotations/main/EPIC_Sounds_validation.csv
python src/select_epic_subset.py --annotations-csv EPIC_Sounds_validation.csv --max-clips 200 > video_ids.txt
# Pipe video_ids.txt into the EPIC download tools; ~5 videos cover 200 impact clips
```

Place files under `<videos_dir>/<participant_id>/videos/<video_id>.MP4` (or
the simpler `<videos_dir>/<video_id>.MP4` fallback).

## Three-stream architecture detail

Per clip (B, 15 frames, 224×224 RGB):

```
appearance:  frozen ResNet50 per frame                  -> (B, 15, 2048)
motion3d:    frozen R(2+1)D-18 over the clip @ 112x112  -> (B, 512)
flow:        RAFT-small consecutive pairs @ 224x224     -> (B, 14, 2, 56, 56)
             then learnable flow CNN inside the model   -> (B, 14, 128)
             with flow[0] replicated for the t=0 token  -> (B, 15, 128)

frame_token = Linear(concat(app_t, flow_t)) -> d_model
cls_token   = Linear(motion3d) -> d_model
+ learned positional embedding over (1+T) tokens
+ a learned onset embedding added only at the impact-frame token

[cls_token, frame_0..frame_14]
  -> 2-layer Transformer encoder (d_model=384, 6 heads, dropout 0.1)
  -> windowed mean-pool over frame tokens at [onset±2]
  -> LayerNorm
  -> PCAHead -> 10-d cochleagram-PCA target
```

Trainable parameters: ~4.7M (just the flow encoder + frame/cls projections +
onset/positional embeddings + Transformer + PCA head). All three pretrained
backbones are frozen.

### Cache layout

```
cache/
  <vid>_coch.npz                  # full-track cochleagram (shared with V1)
  <vid>_f<onset:06d>_feat.npy     # V1 visual features (15, 1024)
  <vid>_f<onset:06d>_clip.npz     # JPEG-encoded 17 frames at 224x224 (q=90)
  <vid>_f<onset:06d>_flow.npy     # fp16 (16, 2, 56, 56) raw RAFT flow
  clip_index.json                 # train/test split, shared across paths
```

The 17-frame storage window is `CLIP_FRAMES + 2*MAX_TEMPORAL_JITTER`, so the
training-time loader can sample a 15-frame window with ±1 frame of temporal
jitter (and shifts the audio target window in lock-step).

## Honest constraints

- **Audio quality is capped by PCA-10 cochleagram + iterative inversion.**
  Both models will sound like degraded versions of real audio regardless of
  visual feature quality. The natural next step is replacing the PCA head
  with a mel-spectrogram head + a pretrained vocoder (HiFi-GAN / BigVGAN);
  [`MelHead`](src/three_stream_model.py) is stubbed but not wired (the
  current readout is single-vector and a per-frame readout is needed for mel).
- **Real-world generalization rests on the fusion head's transfer.** The
  frozen visual backbones generalize well across domains; the visual→audio
  mapping is still trained on 9 hours of Greatest Hits. The realistic
  upper bound on cross-dataset transfer is set by paired-data scale, not
  visual encoder strength.
- **EPIC-Sounds frame rate differs from Greatest Hits** (50/60 fps vs 30
  fps). The EPIC dataset class temporally resamples to 30 fps via per-frame
  MSEC seek (`read_video_clip_at_time`) so per-frame motion velocities
  match training distribution.

## References

- Owens et al., 2016. *Visually Indicated Sounds*. CVPR.
  [paper](https://arxiv.org/abs/1512.08512) · [dataset](https://andrewowens.com/vis/)
- Tran et al., 2018. *A Closer Look at Spatiotemporal Convolutions for
  Action Recognition* (R(2+1)D).
- Teed & Deng, 2020. *RAFT: Recurrent All-Pairs Field Transforms for
  Optical Flow*.
- He et al., 2016. *Deep Residual Learning for Image Recognition* (ResNet).
- Damen et al., 2022. *Rescaling Egocentric Vision* (EPIC-KITCHENS-100).
- Huh et al., 2023. *EPIC-SOUNDS: A Large-Scale Dataset of Actions That
  Sound*.

## Acknowledgments

- McDermott Lab's [pycochleagram](https://github.com/mcdermottLab/pycochleagram)
  for the cochleagram forward/inverse pipeline.
- The EPIC-KITCHENS team for the annotations and the impact-class taxonomy
  that maps cleanly onto Greatest Hits' domain.
