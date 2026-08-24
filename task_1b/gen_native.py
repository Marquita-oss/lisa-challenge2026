"""Genera predicciones a resolucion NATIVA, limpias (sin resize-a-256, sin uint8).

Aplica una operacion suave por slice (identity o gamma) preservando dimensiones.
Objetivo: deshacer el dano auto-infligido del baseline (resize/uint8) que sube
BRISQUE +8.6. Salida float32; empaquetar con pack_uint16.py para subir.

Uso:
  python task_1b/gen_native.py --op identity --label idn
  python task_1b/gen_native.py --op gamma085 --label g085
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from skimage.exposure import adjust_gamma

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'po', str(Path(__file__).resolve().parent / '04_predict_submission.py'))
po = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(po)
build_output_filename = po.build_output_filename

from task_1b.config import VAL_SINGLE_DIR, VAL_COMPLETE_DIR, SUBMISSION_DIR

OPS = {
    'identity': lambda s: s,
    'gamma085': lambda s: adjust_gamma(np.clip(s, 0, 1), 0.85).astype(np.float32),
}


def process(op, src, staging, skip_ciso):
    files = [f for f in sorted(src.rglob('*.nii.gz'))
             if (not skip_ciso) or '_ciso' not in f.name.lower()]
    staging.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        img = nib.load(str(f)); vol = img.get_fdata(dtype=np.float32)
        thin = int(np.argmin(vol.shape)) if vol.ndim == 3 else 2
        if vol.ndim == 2:
            vol = vol[:, :, None]
        out_slices = []
        for i in range(vol.shape[thin]):
            idx = [slice(None)] * 3; idx[thin] = i
            s_raw = vol[tuple(idx)].copy()
            p1 = float(np.percentile(s_raw, 1)); p99 = float(np.percentile(s_raw, 99))
            d = p99 - p1
            if d < 1e-8:
                out_slices.append(s_raw.astype(np.float32)); continue
            sn = np.clip((s_raw - p1) / d, 0, 1).astype(np.float32)
            o = np.clip(op(sn), 0, 1).astype(np.float32)
            out_slices.append((o * d + p1).astype(np.float32))
        enh = np.stack(out_slices, axis=thin)
        out = nib.Nifti1Image(enh, img.affine, img.header)
        out.header.set_data_dtype(np.float32)
        name = build_output_filename(f.name)
        nib.save(out, str(staging / name))
        saved.append(staging / name)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--op', required=True, choices=list(OPS))
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    staging = SUBMISSION_DIR / f'submission_ready_{args.label}'
    op = OPS[args.op]
    saved = []
    saved += process(op, VAL_SINGLE_DIR, staging, skip_ciso=False)
    saved += process(op, VAL_COMPLETE_DIR, staging, skip_ciso=True)
    print(f"Total: {len(saved)}  staging: {staging}")


if __name__ == '__main__':
    main()
