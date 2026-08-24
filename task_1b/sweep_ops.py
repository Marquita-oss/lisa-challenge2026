"""Barrido de operaciones de enhancement SUAVES a resolucion nativa.

Mide, por operacion: BRISQUE_delta vs input (negativo = mejora calidad) y la
cercania al input (L1 medio; bajo = poco drift = bajo riesgo de FID/FRD).
Objetivo: encontrar enhancement con BRISQUE_delta NEGATIVO y drift bajo, lo
contrario de lo que hicieron baseline (+8.6) y LF2CISO (+27).

Uso:
  python task_1b/sweep_ops.py --n 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import piq
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import unsharp_mask
from skimage.exposure import equalize_adapthist, adjust_gamma
from scipy.ndimage import gaussian_filter, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task_1b.config import VAL_SINGLE_DIR, VAL_COMPLETE_DIR

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def npct(s):
    p1, p99 = np.percentile(s, [1, 99]); d = p99 - p1
    return np.clip((s - p1) / (d if d > 1e-8 else 1), 0, 1).astype(np.float32)


OPS = {
    'identity':     lambda s: s,
    'unsharp0.5':   lambda s: unsharp_mask(s, radius=1, amount=0.5),
    'unsharp1.0':   lambda s: unsharp_mask(s, radius=1, amount=1.0),
    'unsharp1.5r2': lambda s: unsharp_mask(s, radius=2, amount=1.5),
    'unsharp2.0r2': lambda s: unsharp_mask(s, radius=2, amount=2.0),
    'clahe':        lambda s: equalize_adapthist(np.clip(s, 0, 1), clip_limit=0.01).astype(np.float32),
    'gamma0.85':    lambda s: adjust_gamma(np.clip(s, 0, 1), 0.85),
    'tvthenunsharp': lambda s: unsharp_mask(denoise_tv_chambolle(s, weight=0.02), radius=1, amount=1.0),
}


def slices(vol):
    thin = int(np.argmin(vol.shape))
    out = []
    for i in range(vol.shape[thin]):
        idx = [slice(None)] * 3; idx[thin] = i
        out.append(vol[tuple(idx)])
    return out


@torch.no_grad()
def brisque_mean(norm_slices):
    ten = [torch.from_numpy(s)[None] for s in norm_slices if s.max() > 1e-6]
    if not ten:
        return np.nan
    x = torch.stack(ten).to(DEV)
    return float(piq.brisque(x, data_range=1.0, reduction='mean').item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    args = ap.parse_args()

    files = sorted(VAL_SINGLE_DIR.rglob('*_lf_*.nii.gz'))[:args.n]
    print(f"Archivos: {len(files)}")

    agg = {k: {'dbq': [], 'l1': []} for k in OPS}
    base_bq = []
    for f in files:
        vol = nib.load(str(f)).get_fdata(dtype=np.float32)
        sl = slices(vol)
        in_norm = [npct(s) for s in sl]
        bq_in = brisque_mean(in_norm)
        base_bq.append(bq_in)
        for name, op in OPS.items():
            out_norm = []
            l1s = []
            for sn in in_norm:
                o = np.clip(op(sn).astype(np.float32), 0, 1)
                out_norm.append(o)
                l1s.append(float(np.mean(np.abs(o - sn))))
            bq_out = brisque_mean(out_norm)
            agg[name]['dbq'].append(bq_out - bq_in)
            agg[name]['l1'].append(np.mean(l1s))

    print(f"\nInput BRISQUE medio: {np.nanmean(base_bq):.2f}\n")
    print(f"{'op':<12}{'BRISQUE_delta':>15}{'L1_drift':>12}")
    print('-' * 39)
    rows = sorted(OPS, key=lambda k: np.nanmean(agg[k]['dbq']))
    for k in rows:
        print(f"{k:<12}{np.nanmean(agg[k]['dbq']):>+15.3f}{np.mean(agg[k]['l1']):>12.4f}")


if __name__ == '__main__':
    main()
