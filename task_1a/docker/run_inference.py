"""run_inference.py — Entrypoint de inferencia Docker para LISA 2026 Task 1a (QC).

Lee todos los .nii.gz bajo $INPUT_DIR (recursivo, cualquier estructura de carpetas),
corre el ensemble EfficientNet-B4 + ConvNeXt-Small (5 folds c/u) con TTA x8, aplica
los umbrales calibrados y escribe LISA_LF_QC_predictions.csv en $OUTPUT_DIR.

No descarga nada en tiempo de ejecución: los pesos (pretrained=False) y los
checkpoints ya están copiados dentro de la imagen.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OrdinalClassifier

SOLUTION_DIR = Path(__file__).resolve().parent
INPUT_DIR = Path(os.environ.get('INPUT_DIR', '/input'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', '/output'))
CHECKPOINTS_DIR = SOLUTION_DIR / 'checkpoints'
THRESHOLDS_PATH = SOLUTION_DIR / 'thresholds.json'
OUT_CSV_NAME = 'LISA_LF_QC_predictions.csv'

ARTIFACT_COLS = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
BACKBONES = [('efficientnet_b4', 'effb4'), ('convnext_small', 'convnexts')]
N_FOLDS = 5
IMG_SIZE = 256
BATCH_SIZE = 16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TTA_OPS = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),
    lambda x: torch.flip(x, dims=[2]),
    lambda x: torch.flip(x, dims=[2, 3]),
    lambda x: torch.rot90(x, k=1, dims=[2, 3]),
    lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3]),
    lambda x: torch.rot90(x, k=3, dims=[2, 3]),
    lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[3]),
]


def val_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_lf_multichannel(nii_path: Path) -> Image.Image:
    """3 slices equidistantes (P25/P50/P75) del eje mas delgado como canales RGB."""
    import nibabel as nib
    nii = nib.load(str(nii_path))
    vol = nii.get_fdata(dtype=np.float32)

    if vol.ndim == 2:
        raw_slices = [vol, vol, vol]
    elif vol.ndim == 3:
        thin_axis = int(np.argmin(vol.shape))
        n = vol.shape[thin_axis]
        positions = [max(0, int(n * p)) for p in (0.25, 0.50, 0.75)]
        raw_slices = []
        for pos in positions:
            idx = [slice(None)] * 3
            idx[thin_axis] = pos
            raw_slices.append(vol[tuple(idx)])
    else:
        raise ValueError(f"Dimensiones NIfTI inesperadas: {vol.ndim} en {nii_path}")

    channels = []
    for s in raw_slices:
        p1, p99 = np.percentile(s, [1, 99])
        denom = p99 - p1
        s = np.clip(s, p1, p99)
        s = (s - p1) / denom if denom > 1e-8 else np.zeros_like(s)
        channels.append((s * 255).astype(np.uint8))

    return Image.fromarray(np.stack(channels, axis=-1))


def extract_patient_id(fname: str) -> str:
    """Extrae el ID numerico del caso sin asumir la palabra intermedia entre 'LISA_' y
    el ID (varia: 'TESTING', 'VALIDATION', o ninguna). Formato real confirmado por el
    equipo para la fase de testing: LISA_TESTING_0001_LF_axi.nii.gz -- el ID es el
    grupo de digitos que precede inmediatamente a '_LF_' (case-insensitive).
    """
    m = re.search(r'(?i)(\d+)_lf_', fname)
    if m:
        return f'LISA_LF_{m.group(1)}'
    m = re.search(r'(\d+)', fname)
    if m:
        return f'LISA_LF_{m.group(1)}'
    return re.sub(r'(?i)\.nii\.gz$', '', fname)


def collect_lf_files(directory: Path) -> list:
    if not directory.exists():
        return []
    return [f for f in sorted(directory.rglob('*.nii.gz')) if '_ciso' not in f.name.lower()]


class SubmissionDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_lf_multichannel(self.paths[idx])
        return self.transform(img), self.paths[idx].name


def load_model(backbone: str, ckpt_path: Path, device):
    model = OrdinalClassifier(backbone=backbone, hidden_dim=512, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_probs(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_p1, all_p2 = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        acc1, acc2 = [], []
        for op in TTA_OPS:
            lo1, lo2 = model(op(imgs))
            acc1.append(torch.sigmoid(lo1).cpu().numpy())
            acc2.append(torch.sigmoid(lo2).cpu().numpy())
        all_p1.append(np.mean(acc1, axis=0))
        all_p2.append(np.mean(acc2, axis=0))
    return np.concatenate(all_p1), np.concatenate(all_p2)


def main():
    t0 = time.time()
    print("===== LISA 2026 Task 1a — QC Inference =====")
    print(f"[INFO] INPUT_DIR={INPUT_DIR}")
    print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] Python: {sys.version.split()[0]}  Torch: {torch.__version__}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    files = collect_lf_files(INPUT_DIR)
    print(f"[INFO] Archivos .nii.gz encontrados en {INPUT_DIR}: {len(files)}")
    if not files:
        print(f"[ERROR] No se encontraron archivos .nii.gz en {INPUT_DIR}. Abortando.")
        sys.exit(1)

    # Pre-validar cada caso por separado: un archivo corrupto/atipico no debe tirar
    # abajo toda la corrida (perder las N filas). Los que fallan quedan con fila
    # por defecto (todo 0) mas abajo, y se loguean como error.
    good_files, bad_files = [], []
    for f in files:
        try:
            load_lf_multichannel(f)
            good_files.append(f)
        except Exception as e:
            print(f"  [ERROR] no se pudo leer {f.name}: {e}")
            bad_files.append(f)
    print(f"[INFO] Casos validos: {len(good_files)}  Casos con error: {len(bad_files)}")

    t1d, t2d = {c: 0.5 for c in ARTIFACT_COLS}, {c: 0.5 for c in ARTIFACT_COLS}
    if THRESHOLDS_PATH.exists():
        d = json.loads(THRESHOLDS_PATH.read_text())
        t1d, t2d = d['t1_per_class'], d['t2_per_class']
        print(f"[INFO] Umbrales calibrados cargados desde {THRESHOLDS_PATH.name}")
    else:
        print(f"[WARN] {THRESHOLDS_PATH} no encontrado, usando 0.5/0.5")

    predictions = {}
    if good_files:
        acc_p1, acc_p2, n_models = [], [], 0
        for backbone, label in BACKBONES:
            ds = SubmissionDataset(good_files, val_transform(IMG_SIZE))
            for k in range(N_FOLDS):
                ckpt = CHECKPOINTS_DIR / f'best_{label}_fold{k}.pth'
                if not ckpt.exists():
                    print(f"[WARN] checkpoint faltante: {ckpt}")
                    continue
                model = load_model(backbone, ckpt, device)
                p1, p2 = predict_probs(model, ds, device)
                acc_p1.append(p1); acc_p2.append(p2)
                n_models += 1
                print(f"  [OK] {backbone:18s} fold{k}")

        if n_models == 0:
            print("[ERROR] Ningun checkpoint del ensemble se pudo cargar. Abortando.")
            sys.exit(1)
        print(f"[INFO] Modelos en el ensemble: {n_models}")

        mean_p1 = np.mean(acc_p1, axis=0)
        mean_p2 = np.mean(acc_p2, axis=0)

        for i, f in enumerate(good_files):
            row = {'patient_id': extract_patient_id(f.name)}
            for k, col in enumerate(ARTIFACT_COLS):
                score = 2 if mean_p2[i, k] >= t2d.get(col, 0.5) else (1 if mean_p1[i, k] >= t1d.get(col, 0.5) else 0)
                row[col] = score
            predictions[f] = row
            print(f"  [{i+1}/{len(good_files)}] {f.name} -> {row['patient_id']}")

    for f in bad_files:
        predictions[f] = {'patient_id': extract_patient_id(f.name), **{c: 0 for c in ARTIFACT_COLS}}

    rows = [predictions[f] for f in files]

    df_sub = pd.DataFrame(rows)[['patient_id'] + ARTIFACT_COLS]
    out_path = OUTPUT_DIR / OUT_CSV_NAME
    df_sub.to_csv(out_path, index=False)

    print(f"\n[INFO] Submission escrita: {out_path}  ({len(df_sub)} filas)")
    for col in ARTIFACT_COLS:
        vc = df_sub[col].value_counts()
        print(f"  {col:15s}  s=0:{int(vc.get(0,0)):3d}  s=1:{int(vc.get(1,0)):3d}  s=2:{int(vc.get(2,0)):3d}")
    print(f"[INFO] Tiempo total: {time.time()-t0:.1f}s")
    print("===== DONE =====")


if __name__ == '__main__':
    main()
