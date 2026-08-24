"""Barrido de operaciones fijas midiendo FID por-caso vs CISO (no BRISQUE).

Para cada caso `complete`: aplica la op a los slices de sus 3 planos LF, calcula
FID vs los slices CISO del mismo caso. Reporta media/min/max sobre los 12 casos.
Objetivo: hallar una op FIJA (no entrenada => sin Goodhart) que baje el FID medio
por debajo de identity, acercando la distribucion al alto campo.

Uso:
  python task_1b/sweep_fid_ops.py
"""
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from skimage.transform import resize as skr
from skimage.filters import unsharp_mask
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.restoration import denoise_tv_chambolle
from piq import FID
from piq.feature_extractors import InceptionV3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task_1b.config import VAL_COMPLETE_DIR

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_INC = InceptionV3().to(DEV).eval()


def npct(s):
    p1, p99 = np.percentile(s, [1, 99]); d = p99 - p1
    return np.clip((s - p1) / (d if d > 1e-8 else 1), 0, 1).astype(np.float32)


OPS = {
    'identity':      lambda s: s,
    'unsharp0.3':    lambda s: unsharp_mask(s, radius=1, amount=0.3),
    'unsharp0.5':    lambda s: unsharp_mask(s, radius=1, amount=0.5),
    'unsharp1.0':    lambda s: unsharp_mask(s, radius=1, amount=1.0),
    'unsharp1.5r2':  lambda s: unsharp_mask(s, radius=2, amount=1.5),
    'clahe':         lambda s: equalize_adapthist(np.clip(s, 0, 1), clip_limit=0.01).astype(np.float32),
    'gamma0.85':     lambda s: adjust_gamma(np.clip(s, 0, 1), 0.85),
    'gamma1.2':      lambda s: adjust_gamma(np.clip(s, 0, 1), 1.2),
    'tv+unsharp':    lambda s: unsharp_mask(denoise_tv_chambolle(s, weight=0.02), radius=1, amount=0.8),
}


def norm_slices(vol):
    thin = int(np.argmin(vol.shape))
    out = []
    for i in range(vol.shape[thin]):
        idx = [slice(None)] * 3; idx[thin] = i
        s = vol[tuple(idx)]
        if s.max() > 1e-6:
            out.append(npct(s))
    return out


@torch.no_grad()
def feats(slice_list, batch=64):
    fs = []
    for i in range(0, len(slice_list), batch):
        chunk = slice_list[i:i + batch]
        t = torch.stack([torch.from_numpy(skr(np.clip(s, 0, 1), (128, 128), order=1,
                         preserve_range=True).astype(np.float32)) for s in chunk])
        t = t[:, None].repeat(1, 3, 1, 1).to(DEV)
        f = _INC(t)[0].squeeze(-1).squeeze(-1)
        fs.append(f.cpu())
    return torch.cat(fs)


def main():
    cases = []
    for cdir in sorted(VAL_COMPLETE_DIR.iterdir()):
        if not cdir.is_dir():
            continue
        ciso = list(cdir.glob('*_ciso.nii.gz'))
        lf = sorted(cdir.glob('*_lf_*.nii.gz'))
        if not ciso or not lf:
            continue
        cases.append((cdir.name, ciso[0], lf))
    print(f"Casos complete con CISO+LF: {len(cases)}")

    # precompute reference feats and LF normalized slices per case
    per_case = {}
    for cid, ciso, lfs in cases:
        rf = feats(norm_slices(nib.load(str(ciso)).get_fdata(dtype=np.float32)))
        lf_sl = []
        for f in lfs:
            lf_sl += norm_slices(nib.load(str(f)).get_fdata(dtype=np.float32))
        per_case[cid] = (rf, lf_sl)

    results = {}
    for name, op in OPS.items():
        fids = []
        for cid, (rf, lf_sl) in per_case.items():
            out = [np.clip(op(s).astype(np.float32), 0, 1) for s in lf_sl]
            pf = feats(out)
            fids.append(float(FID()(pf, rf).item()))
        results[name] = fids
        print(f"{name:<14} FID mean {np.mean(fids):7.2f}  min {np.min(fids):7.2f}  max {np.max(fids):7.2f}")

    print("\n=== ranked by mean FID (lower=better) ===")
    for name in sorted(results, key=lambda k: np.mean(results[k])):
        f = results[name]
        d = np.mean(f) - np.mean(results['identity'])
        print(f"  {name:<14} {np.mean(f):7.2f}  (delta vs identity {d:+.2f})")


if __name__ == '__main__':
    main()
