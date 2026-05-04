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
