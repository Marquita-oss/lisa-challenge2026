"""Validador local sobre los 14 casos `complete` (que tienen CISO de referencia).

Calcula PSNR y SSIM entre cada plano LF mejorado y el CISO remuestreado a la
grilla del plano (via affines). No reproduce exactamente el scorer del challenge
(normalizacion/remuestreo internos desconocidos), pero da una comparacion
RELATIVA consistente entre variantes del pipeline para elegir la mejor sin
prueba-y-error en el leaderboard.

Uso:
  python task_1b/eval_local.py --pred <dir_o_zip_con_enhanced> [--label baseline]
"""
import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import nibabel as nib
from skimage.transform import resize as sk_resize
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task_1b.config import VAL_COMPLETE_DIR


def norm01(a: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(a, [1, 99])
    d = p99 - p1
    if d < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - p1) / d, 0, 1).astype(np.float32)


def collect_pred_files(pred: Path) -> Path:
    """Devuelve un directorio con los .nii.gz (extrae si es zip)."""
    if pred.is_dir():
        return pred
    tmp = Path(tempfile.mkdtemp(prefix='eval_pred_'))
    with zipfile.ZipFile(pred) as zf:
        zf.extractall(tmp)
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='dir o zip con *_enhanced.nii.gz')
    ap.add_argument('--label', default='pred')
    args = ap.parse_args()

    pred_dir = collect_pred_files(Path(args.pred))
    pred_files = {p.name: p for p in pred_dir.rglob('*_enhanced.nii.gz')}

    rows = []
    for case_dir in sorted(VAL_COMPLETE_DIR.iterdir()):
        if not case_dir.is_dir():
            continue
        cid = case_dir.name  # e.g. 0001
        ciso = list(case_dir.glob('*_ciso.nii.gz'))
        if not ciso:
            continue
        ciso_img = nib.load(str(ciso[0]))

        for plane in ['axi', 'cor', 'sag']:
            pname = f'LISA_VALIDATION_{cid}_{plane}_enhanced.nii.gz'
            if pname not in pred_files:
                continue
            enh_img = nib.load(str(pred_files[pname]))
            enh = norm01(np.asarray(enh_img.get_fdata(dtype=np.float32)))
            # Alinear CISO a la grilla del plano via resize de array (mas robusto
            # que el affine, que esta mal en algunos casos). Solo comparacion relativa.
            ciso_n = norm01(np.asarray(ciso_img.get_fdata(dtype=np.float32)))
            ref = sk_resize(ciso_n, enh.shape, order=1, preserve_range=True,
                            anti_aliasing=True).astype(np.float32)

            p = psnr_fn(ref, enh, data_range=1.0)
            s = ssim_fn(ref, enh, data_range=1.0)
            rows.append((cid, plane, p, s))

    if not rows:
        print('No se encontraron pares enhanced/CISO. Revisa --pred.')
        return

    arr_p = np.array([r[2] for r in rows])
    arr_s = np.array([r[3] for r in rows])
    print(f"\n=== {args.label}  ({len(rows)} planos de {VAL_COMPLETE_DIR.name}) ===")
    print(f"PSNR_mean {arr_p.mean():.4f}  min {arr_p.min():.3f}  max {arr_p.max():.3f}")
    print(f"SSIM_mean {arr_s.mean():.4f}  min {arr_s.min():.3f}  max {arr_s.max():.3f}")


if __name__ == '__main__':
    main()
