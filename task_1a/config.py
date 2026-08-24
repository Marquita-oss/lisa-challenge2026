"""
config.py — Configuración única del pipeline Task 1A (limpio, k-fold honesto).

Reemplaza la dispersión de hiperparámetros entre 02_train_v8/v9/v10. Toda la
configuración del pipeline canónico vive aquí.
"""
from pathlib import Path

# --- Rutas ------------------------------------------------------------------
ROOT           = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT / "data"
METADATA_DIR   = DATA_DIR / "metadata"
TRAIN_DIR      = DATA_DIR / "train"
VAL_SINGLE_DIR = DATA_DIR / "val" / "single_plane"
VAL_COMPLETE_DIR = DATA_DIR / "val" / "complete"

TASK_DIR        = Path(__file__).resolve().parent
RESULTS_DIR     = TASK_DIR / "results"
CHECKPOINTS_DIR = TASK_DIR / "checkpoints"

CSV_PATH = METADATA_DIR / "lisa_task1a_2026.csv"

# --- Etiquetas --------------------------------------------------------------
ARTIFACT_COLS = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']

# --- Backbones del ensemble -------------------------------------------------
# Cada (nombre_timm, etiqueta_corta). Los checkpoints se nombran best_{label}_fold{k}.pth
BACKBONES = [
    ('efficientnet_b4', 'effb4'),
    ('convnext_small',  'convnexts'),
]

# --- Hiperparámetros de entrenamiento --------------------------------------
IMG_SIZE      = 256
BATCH_SIZE    = 16
NUM_WORKERS   = 0          # 0 = seguro en Windows
EPOCHS        = 100
PATIENCE      = 20
WARMUP_EPOCHS = 5
LR            = 1e-4
WEIGHT_DECAY  = 5e-2       # subido de 1e-2 para reducir overfitting
DROPOUT_1     = 0.4
DROPOUT_2     = 0.3
HIDDEN_DIM    = 512
GRAD_CLIP     = 0.5
RANDOM_SEED   = 42

# --- Loss (focal ordinal) ---------------------------------------------------
FOCAL_GAMMA = 2.0
LOSS_W2     = 4.0          # peso de la cabeza score>=2
LOSS_W_MONO = 0.5         # penalización de monotonicidad P(>=2)<=P(>=1)

# --- Validación k-fold ------------------------------------------------------
N_FOLDS = 5

# --- Normalización ImageNet -------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- Calibración de umbrales (anti-fuga) -----------------------------------
# Grid conservador alrededor de 0.5 — umbrales extremos sobreajustan el val pequeño.
T1_GRID = [round(0.30 + 0.05 * i, 2) for i in range(8)]   # 0.30..0.65
T2_GRID = [round(0.20 + 0.05 * i, 2) for i in range(10)]  # 0.20..0.65
# Sólo aceptar umbral calibrado si mejora en >= este nº de folds (de N_FOLDS).
CALIB_MIN_FOLDS_IMPROVE = 4

# --- Submission -------------------------------------------------------------
SUBMISSION_FILE = 'LISA_LF_QC_predictions.csv'
N_SUBMISSION_ROWS = 114   # 72 single_plane + 14 complete x 3 planos
