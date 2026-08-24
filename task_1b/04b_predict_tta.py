"""Task 1b — Inferencia mejorada (Fase 1) sobre el checkpoint v2 existente.

Dos cambios respecto a 04_predict_submission.py, ambos IN-DISTRIBUTION (sin reentrenar):
  1. TTA dihedral (8 vistas: rotaciones + flips) promediadas. El modelo se
     entreno con flips/rot90, asi que promediar sobre ese grupo es coherente.
  2. Salida en float32 de punta a punta: se elimina la cuantizacion a uint8
     y el resize de salida se hace en float (skimage), evitando blockiness.

La entrada se mantiene IDENTICA al entrenamiento (resize uint8 a 256) para no
salir de distribucion. Genera el mismo set de 114 archivos y el ZIP.

Uso:
  python task_1b/04b_predict_tta.py --ckpt best_1b_v2.pth
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from skimage.transform import resize as sk_resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_1b.config import VAL_SINGLE_DIR, VAL_COMPLETE_DIR, SUBMISSION_DIR, IMG_SIZE
from task_1b.dataset import SyntheticDenoiseDataset
import nibabel as nib

# Reutiliza helpers del script original (nombre empieza con digito -> import dinamico)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'predict_orig', str(Path(__file__).resolve().parent / '04_predict_submission.py'))
predict_orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_orig)

build_output_filename = predict_orig.build_output_filename
load_model = predict_orig.load_model
create_zip = predict_orig.create_zip

SUBMISSION_ZIP = 'LISA_enhanced_predictions.zip'

# ── TTA dihedral (grupo de 8) ──────────────────────────────────────────────────
# Cada entrada: (forward, inverse) sobre arrays 2D cuadrados (IMG_SIZE x IMG_SIZE)
TTA = [
    (lambda a: a,                         lambda a: a),
    (lambda a: np.rot90(a, 1),            lambda a: np.rot90(a, 3)),
    (lambda a: np.rot90(a, 2),            lambda a: np.rot90(a, 2)),
    (lambda a: np.rot90(a, 3),            lambda a: np.rot90(a, 1)),
    (lambda a: np.fliplr(a),              lambda a: np.fliplr(a)),
    (lambda a: np.flipud(a),              lambda a: np.flipud(a)),
    (lambda a: np.fliplr(np.rot90(a, 1)), lambda a: np.rot90(np.fliplr(a), 3)),
    (lambda a: np.fliplr(np.rot90(a, 3)), lambda a: np.rot90(np.fliplr(a), 1)),
]


@torch.no_grad()
def enhance_volume_tta(model, nii_path: Path, device: torch.device):
    """Enhance cada slice con TTA dihedral y salida float (sin uint8)."""
    img_obj = nib.load(str(nii_path))
    vol = img_obj.get_fdata(dtype=np.float32)

    if vol.ndim == 2:
        vol = vol[:, :, np.newaxis]
        thin_axis = 2
    else:
        thin_axis = int(np.argmin(vol.shape))

    enhanced_slices = []
    n = vol.shape[thin_axis]

    for i in range(n):
        idx = [slice(None)] * 3
        idx[thin_axis] = i
        s_raw = vol[tuple(idx)].copy()

        p1 = float(np.percentile(s_raw, 1))
        p99 = float(np.percentile(s_raw, 99))
        denom = p99 - p1
        if denom < 1e-8:
            enhanced_slices.append(s_raw.astype(np.float32))
            continue

        s_norm = np.clip((s_raw - p1) / denom, 0, 1).astype(np.float32)
        H, W = s_norm.shape

        # Entrada identica al entrenamiento: resize uint8 a 256
        resized = SyntheticDenoiseDataset._resize(s_norm)  # (256,256) float [0,1]

        # Construir batch con las 8 vistas TTA
        variants = [np.ascontiguousarray(f(resized)) for f, _ in TTA]
        batch = torch.from_numpy(np.stack(variants)[:, None]).to(device)  # (8,1,256,256)
        outs = model(batch).squeeze(1).cpu().numpy()                      # (8,256,256)

        # Invertir cada transformacion y promediar
        acc = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
        for k, (_, inv) in enumerate(TTA):
            acc += np.ascontiguousarray(inv(outs[k]))
        out256 = acc / len(TTA)

        # Resize de vuelta a nativo en FLOAT (sin uint8), con anti-aliasing
        out_native = sk_resize(out256, (H, W), order=1, anti_aliasing=True,
                               preserve_range=True).astype(np.float32)
        out_native = np.clip(out_native, 0.0, 1.0)

        enhanced_raw = out_native * denom + p1
        enhanced_slices.append(enhanced_raw.astype(np.float32))

    enhanced_vol = np.stack(enhanced_slices, axis=thin_axis)
    return enhanced_vol, img_obj


def process_directory(model, source_dir: Path, device, staging_dir: Path,
                      skip_ciso: bool = False):
    all_nii = sorted(source_dir.rglob('*.nii.gz'))
    nii_files = [f for f in all_nii if (not skip_ciso) or '_ciso' not in f.name.lower()]
    if not nii_files:
        print(f"  No NIfTI files in {source_dir}")
        return []
    print(f"\n── Enhancing {len(nii_files)} files (TTA x{len(TTA)}) ──")
    staging_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for k, nii_path in enumerate(nii_files, 1):
        out_name = build_output_filename(nii_path.name)
        out_path = staging_dir / out_name
        enhanced_vol, img_obj = enhance_volume_tta(model, nii_path, device)
        out_img = nib.Nifti1Image(enhanced_vol, img_obj.affine, img_obj.header)
        out_img.header.set_data_dtype(np.float32)
        nib.save(out_img, str(out_path))
        saved.append(out_path)
        if k % 10 == 0 or k == len(nii_files):
            print(f"  [{k}/{len(nii_files)}] {nii_path.name} -> {out_name}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='best_1b_v2.pth')
    args = ap.parse_args()

    staging_dir = SUBMISSION_DIR / 'submission_ready_tta'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Checkpoint: {args.ckpt}")
    model = load_model(device, ckpt_name=args.ckpt)

    t0 = time.time()
    saved = []
    if VAL_SINGLE_DIR.exists():
        saved += process_directory(model, VAL_SINGLE_DIR, device, staging_dir, skip_ciso=False)
    if VAL_COMPLETE_DIR.exists():
        saved += process_directory(model, VAL_COMPLETE_DIR, device, staging_dir, skip_ciso=True)

    print(f"\nTotal archivos: {len(saved)}")
    root = Path(__file__).resolve().parent.parent
    zip_path = root / 'LISA_enhanced_predictions_tta.zip'   # nombre separado; renombrar al subir
    if zip_path.exists():
        zip_path.unlink()
    create_zip(saved, zip_path)
    print(f"\nTiempo: {time.time()-t0:.1f}s")
    print(f"Staging: {staging_dir}")
    print(f"ZIP (renombrar a {SUBMISSION_ZIP} para subir): {zip_path}")


if __name__ == '__main__':
    main()
