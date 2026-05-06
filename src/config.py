"""Central config. Edit paths/hyperparams here, import everywhere else."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"

for d in (CACHE_DIR, CHECKPOINT_DIR, OUTPUT_DIR, FIGURE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Audio
TARGET_SR = 22050              # resample target (paper captures at 96 kHz)
N_COCH_FILTERS = 40            # bandpass channels on ERB scale
COCH_LOW_HZ = 50
COCH_HIGH_HZ = 11000           # < TARGET_SR / 2
ENVELOPE_SR = 90               # downsampled envelope rate (Hz)
COMPRESSION_EXP = 0.3
ONSET_MIN_SEPARATION_S = 0.25  # NMS distance for onset detection

# Video / clipping
VIDEO_FPS = 30
CLIP_FRAMES = 15               # 0.5 s window
ONSET_FRAME_INDEX = 7          # impact lands here within the 15-frame window
CLIP_DURATION_S = CLIP_FRAMES / VIDEO_FPS

# Visual features
RGB_FEAT_DIM = 512             # ResNet18 avgpool
SPACETIME_FEAT_DIM = 512
PER_FRAME_FEAT_DIM = RGB_FEAT_DIM + SPACETIME_FEAT_DIM  # 1024

# Model
PCA_DIM = 10
HIDDEN1 = 512
HIDDEN2 = 256
LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 20

# Train/test
SPLIT_RATIO = 0.75             # video-level, not clip-level
SPLIT_SEED = 0

# ---------------------------------------------------------------------------
# Three-stream V2A (real-world generalization track). Additive to the V1 path:
# the V1 MLP/cache files are untouched so both models can be trained side by
# side off the same clip index.
#
# Cache layout (option (c) — flow-only frozen, app/3D recomputed at train
# time so we can augment in pixel space):
#   <vid>_f<onset>_clip.npz   JPEG-encoded uint8 frames, CACHE_FRAMES_STORED of
#                             them; one entry per frame keyed "f00".."f{N-1}"
#   <vid>_f<onset>_flow.npy   fp16 (CACHE_FRAMES_STORED-1, 2,
#                                  CACHE_FLOW_SIZE, CACHE_FLOW_SIZE)
#
# Cache stores 17 frames (15-frame model window + 1 frame on each side) so the
# train-time loader can apply temporal jitter ±MAX_TEMPORAL_JITTER frames.
# ---------------------------------------------------------------------------

# Pretrained backbone outputs (frozen at train time)
APP_FEAT_DIM = 2048            # ResNet50 avgpool (ImageNet1K_V2)
MOTION3D_FEAT_DIM = 512        # R(2+1)D-18 avgpool (Kinetics-400)

# Stream input sizes
APP_INPUT_SIZE = 224           # ResNet50 native input (bilinear-up from cache)
MOTION3D_INPUT_SIZE = 112      # R(2+1)D-18 native input (bilinear-down from cache)
CACHE_FRAME_SIZE = 224         # cached RGB resolution
CACHE_FLOW_SIZE = 56           # cached flow resolution (fp16)
RAFT_INPUT_SIZE = 224          # RAFT only runs at cache time; native res is fine

# Cache storage window — 15-frame model window + 1 frame on each side so the
# train-time loader can apply temporal jitter without re-decoding the video.
MAX_TEMPORAL_JITTER = 1        # max train-time offset in frames
CACHE_FRAMES_STORED = CLIP_FRAMES + 2 * MAX_TEMPORAL_JITTER  # 17
JPEG_QUALITY = 90              # cv2.imencode quality for cached frames

# Learnable flow encoder (replaces the frozen HSV→ResNet18 path).
# Operates on raw flow (2, H, W) cached at CACHE_FLOW_SIZE.
FLOW_ENCODER_DIM = 128

# Onset alignment for the fusion head.
ONSET_WINDOW_HALF = 2          # readout pools tokens at [onset-2 .. onset+2]

# Fusion head
TS_D_MODEL = 384
TS_N_HEADS = 6
TS_N_LAYERS = 2
TS_DROPOUT = 0.1
