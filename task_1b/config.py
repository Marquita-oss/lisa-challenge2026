from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
TRAIN_DIR = DATA_DIR / "train"
VAL_SINGLE_DIR = DATA_DIR / "val" / "single_plane"
VAL_COMPLETE_DIR = DATA_DIR / "val" / "complete"

TASK_DIR = ROOT / "task_1b"
RESULTS_DIR = TASK_DIR / "results"
CHECKPOINTS_DIR = TASK_DIR / "checkpoints"
SUBMISSION_DIR = TASK_DIR / "submission"

# Partition CSVs (mutually exclusive, exhaustive split of lisa_task1a_2026.csv)
CSV_NONOISE_NOMOTION   = METADATA_DIR / "task_1b_nonoise_nomotion.csv"
CSV_NONOISE_WITHMOTION = METADATA_DIR / "task_1b_nonoise_withmotion.csv"
CSV_WITHNOISE_NOMOTION = METADATA_DIR / "task_1b_withnoise_nomotion.csv"
CSV_WITHNOISE_WITHMOTION = METADATA_DIR / "task_1b_withnoise_withmotion.csv"

# Task 1A artefacts of interest for quality feedback
TASK_1A_CHECKPOINT  = ROOT / "task_1a" / "checkpoints" / "best_1a.pth"
TASK_1A_THRESHOLDS  = ROOT / "task_1a" / "results" / "04_thresholds.json"

ARTIFACT_COLS = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
RANDOM_SEED = 42

# ── Image size ────────────────────────────────────────────────────────────────
# All slices are resized to this fixed square before batching.
# Must be a multiple of 2^N_LEVELS (16) to avoid U-Net padding issues.
IMG_SIZE = 256

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_CH   = 32    # first encoder channels; doubles at each level
N_LEVELS  = 4     # encoder depth  (→ divisor = 2^4 = 16 required)
BOTTLE_CH = 512   # bottleneck channels

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 8
NUM_WORKERS  = 0   # Windows-safe
EPOCHS       = 100
PATIENCE     = 20  # early stopping on val PSNR
LR           = 2e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.20

# ── Loss ──────────────────────────────────────────────────────────────────────
LAMBDA_L1   = 0.6
LAMBDA_SSIM = 0.4

# ── Synthetic degradation ─────────────────────────────────────────────────────
# Rician noise: sigma relative to image std
NOISE_SIGMA_MIN = 0.03
NOISE_SIGMA_MAX = 0.18
# Motion ghosting
MOTION_PROB       = 0.50   # fraction of training samples that get motion
MOTION_GHOSTS_MIN = 2
MOTION_GHOSTS_MAX = 6
MOTION_ALPHA_MIN  = 0.04
MOTION_ALPHA_MAX  = 0.25

# ── Evaluation criteria ───────────────────────────────────────────────────────
MIN_PSNR = 28.0   # dB — challenge criterion
MIN_SSIM = 0.80

# ── v2: Adversarial loss (Task 1A as frozen discriminator) ────────────────────
# Weight of the adversarial BCE term in CombinedLossWithAdversarial.
# Start at 0.05 so reconstruction quality still dominates; increase if
# the Task 1A noise/motion scores do not improve.
LAMBDA_ADV = 0.05

# ── v2: Warmup epochs before adversarial loss is activated ────────────────────
ADV_WARMUP_EPOCHS = 10   # first 10 epochs: pure L1+SSIM, then adv is added

# ── v2: Checkpoint / results names ───────────────────────────────────────────
CKPT_V2  = 'best_1b_v2.pth'
TRAIN_RESULTS_V2 = '02_training_v2.json'
