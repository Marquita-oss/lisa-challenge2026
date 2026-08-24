"""Evaluador NO-REFERENCE fiel al scorer (BRISQUE/CLIPIQA via piq).

BRISQUE y CLIPIQA son no-reference: se pueden calcular localmente igual que el
scorer del challenge (que usa piq, segun el log). Permite comparar candidatos
de forma confiable ANTES de subir, y ademas el delta vs el LF de entrada.

Uso:
  python task_1b/eval_noref.py --pred <dir_o_zip> [--norm slice_pct|slice_minmax|vol_max] [--label X]
"""
import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import piq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task_1b.config import VAL_SINGLE_DIR, VAL_COMPLETE_DIR

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def norm_slice(s, mode):
    if mode == 'slice_pct':
        p1, p99 = np.percentile(s, [1, 99]); d = p99 - p1
        return np.clip((s - p1) / (d if d > 1e-8 else 1), 0, 1)
    if mode == 'slice_minmax':
        mn, mx = s.min(), s.max(); d = mx - mn
        return np.clip((s - mn) / (d if d > 1e-8 else 1), 0, 1)
    raise ValueError(mode)


def slices_of(vol):
    thin = int(np.argmin(vol.shape)) if vol.ndim == 3 else 2
    if vol.ndim == 2:
        vol = vol[:, :, None]; thin = 2
    out = []
    for i in range(vol.shape[thin]):
        idx = [slice(None)] * 3; idx[thin] = i
        out.append(vol[tuple(idx)])
    return out


@torch.no_grad()
def metrics_for_volume(vol, mode):
    """BRISQUE y CLIPIQA promediados sobre slices (vol-level 'case')."""
    sl = slices_of(vol)
    ten = []
    for s in sl:
        sn = norm_slice(s.astype(np.float32), mode)
        if sn.max() <= 1e-6:
            continue
        ten.append(torch.from_numpy(sn.astype(np.float32))[None])
    if not ten:
        return None, None
    x = torch.stack(ten).to(DEV)  # (N,1,H,W)
    # BRISQUE necesita batch homogeneo (mismas dims) -> ok dentro de un volumen
    bq = float(piq.brisque(x, data_range=1.0, reduction='mean').item())
    # CLIPIQA requiere 3 canales
    x3 = x.repeat(1, 3, 1, 1)
    try:
        cq = float(piq.clip_iqa(x3, data_range=1.0).mean().item())
    except Exception:
        cq = float('nan')
    return bq, cq


def collect(pred):
    p = Path(pred)
    if p.is_dir():
        return p
    tmp = Path(tempfile.mkdtemp(prefix='nr_'))
    with zipfile.ZipFile(p) as zf:
        zf.extractall(tmp)
    return tmp


def find_input(cid, plane):
    for base in [VAL_SINGLE_DIR, VAL_COMPLETE_DIR]:
        f = base / cid / f'lisa_validation_{cid}_lf_{plane}.nii.gz'
        if f.exists():
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--norm', default='slice_pct')
    ap.add_argument('--label', default='pred')
    ap.add_argument('--delta', action='store_true', help='calcular delta vs input LF')
    args = ap.parse_args()

    d = collect(args.pred)
    files = sorted(d.rglob('*_enhanced.nii.gz'))
    bqs, cqs, dbqs, dcqs = [], [], [], []
    for f in files:
        m = re.search(r'LISA_VALIDATION_(\d+)_(axi|cor|sag)_enhanced', f.name)
        if not m:
            continue
        cid, plane = m.group(1), m.group(2)
        vol = nib.load(str(f)).get_fdata(dtype=np.float32)
        bq, cq = metrics_for_volume(vol, args.norm)
        if bq is None:
            continue
        bqs.append(bq); cqs.append(cq)
        if args.delta:
            inp = find_input(cid, plane)
            if inp:
                ivol = nib.load(str(inp)).get_fdata(dtype=np.float32)
                ibq, icq = metrics_for_volume(ivol, args.norm)
                if ibq is not None:
                    dbqs.append(bq - ibq); dcqs.append(cq - icq)

    print(f"\n=== {args.label}  (norm={args.norm}, {len(bqs)} vols) ===")
    print(f"BRISQUE_enhanced_mean : {np.mean(bqs):.3f}")
    print(f"CLIPIQA_enhanced_mean : {np.nanmean(cqs):.3f}")
    if args.delta and dbqs:
        print(f"BRISQUE_delta_mean    : {np.mean(dbqs):+.3f}")
        print(f"CLIPIQA_delta_mean    : {np.nanmean(dcqs):+.3f}")


if __name__ == '__main__':
    main()
