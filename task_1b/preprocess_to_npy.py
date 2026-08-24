"""Pre-procesamiento de datos para Task 1b — convierte NIfTI a .npy.

Carga cada volumen NIfTI de entrenamiento, extrae slices, normaliza
y redimensiona a IMG_SIZE x IMG_SIZE, y guarda como arrays .npy.

Esto elimina el cuello de botella de I/O durante el entrenamiento:
  ANTES: cada __getitem__ llama nib.load + decompress gzip + percentiles + PIL resize
  DESPUES: cada __getitem__ es un np.load de un array binario precompilado

El dataset resultante ocupa mas espacio en disco (float32 raw) pero
la GPU deja de esperar al CPU/disco durante el entrenamiento.

Uso (desde la raiz del proyecto):
  conda activate lisa2026
  python task_1b/preprocess_to_npy.py
  python task_1b/preprocess_to_npy.py --split val    # para set de validacion
  python task_1b/preprocess_to_npy.py --split all    # train + val
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_1b.config import (
    TRAIN_DIR, VAL_SINGLE_DIR, TASK_DIR, IMG_SIZE,
    CSV_NONOISE_NOMOTION, CSV_NONOISE_WITHMOTION,
    CSV_WITHNOISE_NOMOTION, CSV_WITHNOISE_WITHMOTION,
)
from task_1b.utils.io import load_volume, extract_slices, normalize_slice, find_image_path

# Directorio donde se guardaran los .npy
NPY_DIR = TASK_DIR / 'npy_cache'


def resize_slice(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize a 2D float32 [0,1] slice to (size, size) via PIL bilinear."""
    pil = PILImage.fromarray((img * 255).astype(np.uint8)).resize(
        (size, size), PILImage.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def process_nifti(nii_path: Path, out_dir: Path) -> int:
    """Extrae, normaliza y guarda todas las slices de un NIfTI como .npy.

    Cada slice se guarda como: out_dir/<stem>_s{idx:03d}.npy
    Shape: (IMG_SIZE, IMG_SIZE) float32

    Devuelve el numero de slices guardadas.
    """
    stem = nii_path.stem.replace('.nii', '')  # quitar doble extension
    try:
        vol, _, thin_axis = load_volume(nii_path)
    except Exception as e:
        print(f"  [ERROR] {nii_path.name}: {e}")
        return 0

    raw_slices = extract_slices(vol, thin_axis)
    saved = 0
    for i, s in enumerate(raw_slices):
        norm = normalize_slice(s)
        resized = resize_slice(norm, IMG_SIZE)
        out_path = out_dir / f'{stem}_s{i:03d}.npy'
        np.save(out_path, resized)
        saved += 1
    return saved


def process_csv(csv_path: Path, train_dir: Path, out_dir: Path, label: str) -> dict:
    """Procesa todos los volumenes de una particion CSV."""
    if not csv_path.exists():
        print(f"  [SKIP] CSV no encontrado: {csv_path}")
        return {'volumes': 0, 'slices': 0}

    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_volumes, total_slices = 0, 0
    t0 = time.time()

    print(f"\n  Procesando {label} ({len(df)} volumenes)...")
    for fn in df['filename']:
        nii_path = find_image_path(fn, train_dir)
        if nii_path is None:
            continue
        n = process_nifti(nii_path, out_dir)
        if n > 0:
            total_volumes += 1
            total_slices += n

    elapsed = time.time() - t0
    print(f"  -> {total_volumes} volumenes / {total_slices} slices en {elapsed:.1f}s")
    return {'volumes': total_volumes, 'slices': total_slices}


def process_val_dir(val_dir: Path, out_dir: Path) -> dict:
    """Procesa val/single_plane/ (imagenes de test, sin ground truth)."""
    if not val_dir.exists():
        print(f"  [SKIP] {val_dir} no encontrado")
        return {'volumes': 0, 'slices': 0}

    nii_files = sorted(val_dir.rglob('*.nii.gz'))
    out_dir.mkdir(parents=True, exist_ok=True)

    total_volumes, total_slices = 0, 0
    t0 = time.time()

    print(f"\n  Procesando val/single_plane/ ({len(nii_files)} volumenes)...")
    for nii_path in nii_files:
        n = process_nifti(nii_path, out_dir)
        if n > 0:
            total_volumes += 1
            total_slices += n

    elapsed = time.time() - t0
    print(f"  -> {total_volumes} volumenes / {total_slices} slices en {elapsed:.1f}s")
    return {'volumes': total_volumes, 'slices': total_slices}


def check_existing(out_dir: Path, label: str) -> bool:
    """Avisa si el cache ya existe para evitar re-procesamiento innecesario."""
    existing = list(out_dir.glob('*.npy')) if out_dir.exists() else []
    if existing:
        print(f"  [INFO] Cache {label} ya existe ({len(existing)} archivos). "
              f"Usar --force para re-procesar.")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Pre-procesa NIfTI a .npy para acelerar el entrenamiento de Task 1b')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'all'],
                        help='Que conjunto pre-procesar (default: train)')
    parser.add_argument('--force', action='store_true',
                        help='Re-procesar aunque el cache ya exista')
    args = parser.parse_args()

    print("=" * 60)
    print("Task 1b — Pre-procesamiento NIfTI -> .npy")
    print("=" * 60)
    print(f"  IMG_SIZE:  {IMG_SIZE}x{IMG_SIZE}")
    print(f"  NPY_DIR:   {NPY_DIR}")

    t_total = time.time()
    results = {}

    if args.split in ('train', 'all'):
        partitions = [
            (CSV_NONOISE_NOMOTION,   NPY_DIR / 'train_nonoise_nomotion',   'nonoise_nomotion'),
            (CSV_WITHNOISE_NOMOTION, NPY_DIR / 'train_withnoise_nomotion', 'withnoise_nomotion'),
            (CSV_NONOISE_WITHMOTION, NPY_DIR / 'train_nonoise_withmotion', 'nonoise_withmotion'),
            (CSV_WITHNOISE_WITHMOTION, NPY_DIR / 'train_withnoise_withmotion', 'withnoise_withmotion'),
        ]
        print("\n--- Particiones de entrenamiento ---")
        for csv_path, out_dir, label in partitions:
            if not args.force and check_existing(out_dir, label):
                continue
            r = process_csv(csv_path, TRAIN_DIR, out_dir, label)
            results[label] = r

    if args.split in ('val', 'all'):
        print("\n--- Validacion ---")
        val_out = NPY_DIR / 'val_single_plane'
        if args.force or not check_existing(val_out, 'val_single_plane'):
            r = process_val_dir(VAL_SINGLE_DIR, val_out)
            results['val_single_plane'] = r

    elapsed_total = time.time() - t_total
    print(f"\nPre-procesamiento completado en {elapsed_total/60:.1f} min")
    print(f"Cache guardado en: {NPY_DIR}")

    total_slices = sum(r.get('slices', 0) for r in results.values())
    if total_slices:
        # Estimar tamano del cache
        bytes_per_slice = IMG_SIZE * IMG_SIZE * 4  # float32
        total_mb = total_slices * bytes_per_slice / 1024 / 1024
        print(f"Total slices: {total_slices}  (~{total_mb:.0f} MB en disco)")


if __name__ == '__main__':
    main()
