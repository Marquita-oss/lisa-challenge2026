"""Task 1b — Submission prediction.

Aplica el denoiser ResUNet a TODOS los archivos LF de validacion:
  1. val/single_plane/  — 72 casos ciegos (1001-1072)
  2. val/complete/      — 14 casos con referencia (0001-0014), solo planos LF

Formato de salida (especificacion oficial del challenge):
  Nombre de cada NIfTI: LISA_VALIDATION_1234_axi_enhanced.nii.gz
                        LISA_VALIDATION_0001_cor_enhanced.nii.gz
  ZIP final:            LISA_enhanced_predictions.zip  (raiz del proyecto)

Uso:
  python task_1b/04_predict_submission.py                         # v2 (default)
  python task_1b/04_predict_submission.py --ckpt best_1b_v3.pth  # v3
  python task_1b/04_predict_submission.py --ckpt best_1b.pth     # v1
"""
import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_1b.config import (
    VAL_SINGLE_DIR, VAL_COMPLETE_DIR, SUBMISSION_DIR,
    CHECKPOINTS_DIR, RESULTS_DIR, RANDOM_SEED,
)
from task_1b.model import build_model
from task_1b.dataset import SyntheticDenoiseDataset

# Nombre exacto del ZIP requerido por el sistema de evaluacion del challenge
SUBMISSION_ZIP = 'LISA_enhanced_predictions.zip'

SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_output_filename(input_filename: str) -> str:
    """Convierte el nombre de entrada al formato requerido por el challenge.

    Reglas:
      - Eliminar el segmento '_lf' del nombre
      - Agregar '_enhanced' antes de la extension
      - Convertir a mayusculas el prefijo LISA_VALIDATION

    Ejemplos:
      lisa_validation_1001_lf_axi.nii.gz  -> LISA_VALIDATION_1001_axi_enhanced.nii.gz
      LISA_LF_1234_sag.nii.gz             -> LISA_VALIDATION_1234_sag_enhanced.nii.gz
    """
    stem = input_filename.replace('.nii.gz', '')

    # Extraer ID numerico y plano de cualquier variante del nombre
    m = re.search(r'(?i)lisa(?:_(?:lf|validation))?_(\d+)(?:_lf)?_(axi|cor|sag)', stem)
    if not m:
        # Fallback: agregar solo _enhanced si no se reconoce el patron
        return stem + '_enhanced.nii.gz'

    case_id = m.group(1)
    plane   = m.group(2).lower()
    return f'LISA_VALIDATION_{case_id}_{plane}_enhanced.nii.gz'


def load_model(device, ckpt_name: str = 'best_1b.pth'):
    ckpt_path = CHECKPOINTS_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    model = build_model(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val PSNR={ckpt['val_psnr']:.2f} dB)")
    return model


@torch.no_grad()
def enhance_volume(model, nii_path: Path, device: torch.device):
    """Load a NIfTI, enhance each slice, return enhanced float32 array in original range."""
    import nibabel as nib
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

        # Normalize to [0, 1]
        p1  = float(np.percentile(s_raw, 1))
        p99 = float(np.percentile(s_raw, 99))
        denom = p99 - p1
        if denom < 1e-8:
            enhanced_slices.append(s_raw)
            continue

        s_norm = np.clip((s_raw - p1) / denom, 0, 1).astype(np.float32)
        H, W = s_norm.shape

        # Resize to IMG_SIZE for the model, then resize output back to native resolution
        resized = SyntheticDenoiseDataset._resize(s_norm)
        inp = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device)

        out256 = model(inp).squeeze(0).squeeze(0).cpu().numpy()

        out_pil = PILImage.fromarray((out256 * 255).astype(np.uint8)).resize(
            (W, H), PILImage.BILINEAR)
        out_native = np.array(out_pil, dtype=np.float32) / 255.0

        enhanced_raw = out_native * denom + p1
        enhanced_slices.append(enhanced_raw.astype(np.float32))

    enhanced_vol = np.stack(enhanced_slices, axis=thin_axis)
    return enhanced_vol, img_obj


def process_directory(model, source_dir: Path, device, staging_dir: Path,
                      skip_ciso: bool = False) -> dict:
    """Enhance all NIfTI files under source_dir and save to staging_dir with correct names."""
    all_nii = sorted(source_dir.rglob('*.nii.gz'))
    if skip_ciso:
        nii_files = [f for f in all_nii if '_ciso' not in f.name.lower()]
    else:
        nii_files = all_nii

    if not nii_files:
        print(f"  No NIfTI files found in {source_dir}")
        return {'processed': 0, 'errors': 0, 'files': []}

    print(f"\n── Enhancing {len(nii_files)} files ──────────────────────────────")
    staging_dir.mkdir(parents=True, exist_ok=True)
    processed, errors = 0, 0
    saved_files = []

    for nii_path in nii_files:
        out_name = build_output_filename(nii_path.name)
        out_path = staging_dir / out_name

        try:
            enhanced_vol, img_obj = enhance_volume(model, nii_path, device)

            import nibabel as nib
            out_img = nib.Nifti1Image(enhanced_vol, img_obj.affine, img_obj.header)
            out_img.header.set_data_dtype(np.float32)
            nib.save(out_img, str(out_path))

            saved_files.append(out_path)
            processed += 1
            if processed % 10 == 0 or processed == len(nii_files):
                print(f"  [{processed}/{len(nii_files)}] {nii_path.name} -> {out_name}")
        except Exception as e:
            print(f"  [ERROR] {nii_path.name}: {e}")
            errors += 1

    print(f"  Done: {processed} enhanced, {errors} errors")
    return {'processed': processed, 'errors': errors, 'files': saved_files}


def create_zip(files: list, zip_path: Path) -> None:
    """Empaqueta todos los NIfTI en un ZIP plano (sin subcarpetas)."""
    print(f"\nCreando {zip_path.name} con {len(files)} archivos...")
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)   # ZIP plano: solo el nombre de archivo
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  ZIP creado: {zip_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='best_1b_v2.pth',
                        help='Checkpoint filename inside checkpoints/ (default: best_1b_v2.pth)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Numero de workers DataLoader')
    args = parser.parse_args()

    # Directorio de staging: archivos con nombre correcto, antes de zipear
    suffix      = args.ckpt.replace('best_1b', '').replace('.pth', '')
    staging_dir = SUBMISSION_DIR / f'submission_ready{suffix}'
    result_name = f'04_submission{suffix}.json'

    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Staging:    {staging_dir}")

    model = load_model(device, ckpt_name=args.ckpt)
    t0 = time.time()

    all_saved_files = []

    # val/single_plane/ — 72 casos (1001-1072)
    if VAL_SINGLE_DIR.exists():
        result_sp = process_directory(model, VAL_SINGLE_DIR, device, staging_dir,
                                      skip_ciso=False)
        all_saved_files.extend(result_sp['files'])
    else:
        print(f"[WARN] {VAL_SINGLE_DIR} no encontrado.")
        result_sp = {'processed': 0, 'errors': 0}

    # val/complete/ — 14 casos (0001-0014), solo planos LF (excluye CISO)
    if VAL_COMPLETE_DIR.exists():
        result_co = process_directory(model, VAL_COMPLETE_DIR, device, staging_dir,
                                      skip_ciso=True)
        all_saved_files.extend(result_co['files'])
    else:
        print(f"[WARN] {VAL_COMPLETE_DIR} no encontrado.")
        result_co = {'processed': 0, 'errors': 0}

    n = len(all_saved_files)
    print(f"\nTotal archivos generados: {n}")
    if n != 114:
        print(f"  [AVISO] Se esperaban 114 archivos (72 single_plane + 42 complete LF), "
              f"se generaron {n}.")


    # Crear ZIP con nombre exacto requerido por el challenge
    root = Path(__file__).resolve().parent.parent
    zip_path = root / SUBMISSION_ZIP
    if zip_path.exists():
        zip_path.unlink()
        print(f"  ZIP anterior eliminado.")
    create_zip(all_saved_files, zip_path)

    # Verificar estructura del ZIP
    print("\nVerificacion del ZIP — primeros 5 archivos:")
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        for name in names[:5]:
            print(f"  {name}")
        if len(names) > 5:
            print(f"  ... ({len(names)} archivos en total)")

    elapsed = time.time() - t0
    print(f"\nTiempo total: {elapsed:.1f} s")
    print(f"\nArchivo listo para subir a Synapse:")
    print(f"  {zip_path}")

    summary = {
        'checkpoint':      args.ckpt,
        'staging_dir':     str(staging_dir),
        'zip_path':        str(zip_path),
        'n_files':         n,
        'elapsed_seconds': elapsed,
        'single_plane':    {k: v for k, v in result_sp.items() if k != 'files'},
        'complete':        {k: v for k, v in result_co.items() if k != 'files'},
    }
    out = RESULTS_DIR / result_name
    out.write_text(json.dumps(summary, indent=2))
    print(f"  Resumen guardado en: {out}")


if __name__ == '__main__':
    main()
