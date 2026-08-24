"""Opcion 2 (fase 3): seleccion HONESTA de checkpoint por FID held-out + BRISQUE.

Para cada checkpoint: mejora en memoria los 14 casos val/complete (nunca vistos en
train), calcula FID por-caso vs su CISO y BRISQUE_delta vs el LF de entrada.
Compara contra identity (native_clean = sub 9768920). Gate: solo vale la pena si
baja el FID medio SIN empeorar BRISQUE_delta. Held-out + VGG!=Inception => evita el
Goodhart que hundio al GAN (que selecciono por FID en la misma distribucion).

Uso:
  python task_1b/12_select_checkpoint.py --ckpts best_1b_perc.pth perc/ep020.pth ...
  python task_1b/12_select_checkpoint.py --glob 'perc/ep*.pth'
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import piq
from skimage.transform import resize as skr
from piq import FID
from piq.feature_extractors import InceptionV3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task_1b.config import VAL_COMPLETE_DIR, CHECKPOINTS_DIR
from task_1b.model import build_model

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_INC = InceptionV3().to(DEV).eval()
CANVAS = 160


def npct(s):
    p1, p99 = np.percentile(s, [1, 99]); d = p99 - p1
    return np.clip((s - p1) / (d if d > 1e-8 else 1), 0, 1).astype(np.float32)


def pad_to(a, size=CANVAS):
    H, W = a.shape
    out = np.zeros((size, size), np.float32); out[:H, :W] = a
    return out


def slices(vol):
    thin = int(np.argmin(vol.shape))
    out = []
    for i in range(vol.shape[thin]):
        idx = [slice(None)] * 3; idx[thin] = i
        s = vol[tuple(idx)]
        if s.max() > 1e-6:
            out.append(npct(s))
    return out


@torch.no_grad()
def feats(sl, batch=64):
    fs = []
    for i in range(0, len(sl), batch):
        chunk = sl[i:i + batch]
        t = torch.stack([torch.from_numpy(skr(np.clip(s, 0, 1), (128, 128), order=1,
                         preserve_range=True).astype(np.float32)) for s in chunk])
        t = t[:, None].repeat(1, 3, 1, 1).to(DEV)
        fs.append(_INC(t)[0].squeeze(-1).squeeze(-1).cpu())
    return torch.cat(fs)


@torch.no_grad()
def brisque_mean(norm_sl):
    # slices across planes have heterogeneous shapes -> compute per-slice and average
    vals = []
    for s in norm_sl:
        t = torch.from_numpy(np.clip(s, 0, 1).astype(np.float32))[None, None].to(DEV)
        try:
            vals.append(float(piq.brisque(t, data_range=1.0, reduction='mean').item()))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else np.nan


@torch.no_grad()
def enhance_plane(model, vol):
    """Return list of enhanced normalized slices (model applied per slice)."""
    thin = int(np.argmin(vol.shape))
    out = []
    for i in range(vol.shape[thin]):
        idx = [slice(None)] * 3; idx[thin] = i
        s = vol[tuple(idx)]
        if s.max() <= 1e-6:
            continue
        sn = npct(s); H, W = sn.shape
        t = torch.from_numpy(pad_to(np.ascontiguousarray(sn)))[None, None].to(DEV)
        o = model(t).squeeze().cpu().numpy()[:H, :W]
        out.append(np.clip(o, 0, 1).astype(np.float32))
    return out


def load_cases():
    cases = []
    for cdir in sorted(VAL_COMPLETE_DIR.iterdir()):
        if not cdir.is_dir():
            continue
        cid = cdir.name
        ciso = list(cdir.glob('*_ciso.nii.gz'))
        lfs = sorted(cdir.glob('*_lf_*.nii.gz'))
        if not ciso or not lfs:
            continue
        cases.append((cid, ciso[0], lfs))
    return cases


def evaluate(model, cases, ref_feats, in_slices_by_case):
    """Returns (fid_mean, fid_min, fid_max, brisque_mean, brisque_delta)."""
    fids, bq_enh, bq_in = [], [], []
    for cid, _, lfs in cases:
        enh = []
        for f in lfs:
            vol = nib.load(str(f)).get_fdata(dtype=np.float32)
            enh += (enhance_plane(model, vol) if model is not None else slices(vol))
        pf = feats(enh)
        fids.append(float(FID()(pf, ref_feats[cid]).item()))
        bq_enh.append(brisque_mean(enh))
        bq_in.append(in_slices_by_case[cid])
    dbq = float(np.nanmean(bq_enh) - np.nanmean(bq_in))
    return (float(np.mean(fids)), float(np.min(fids)), float(np.max(fids)),
            float(np.nanmean(bq_enh)), dbq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='*', default=[])
    ap.add_argument('--glob', default=None)
    args = ap.parse_args()

    ck_paths = [CHECKPOINTS_DIR / c for c in args.ckpts]
    if args.glob:
        ck_paths += sorted(CHECKPOINTS_DIR.glob(args.glob))
    ck_paths = [p for p in ck_paths if p.exists()]

    cases = load_cases()
    print(f"Casos held-out (val/complete): {len(cases)}")
    ref_feats = {cid: feats(slices(nib.load(str(ciso)).get_fdata(dtype=np.float32)))
                 for cid, ciso, _ in cases}
    # input LF brisque per case (for delta)
    in_bq = {}
    for cid, _, lfs in cases:
        insl = []
        for f in lfs:
            insl += slices(nib.load(str(f)).get_fdata(dtype=np.float32))
        in_bq[cid] = brisque_mean(insl)

    print(f"\n{'model':<22}{'FID_mean':>9}{'FID_min':>9}{'FID_max':>9}{'BRISQUE':>9}{'dBRISQUE':>10}")
    print('-' * 68)
    # identity baseline (native_clean)
    fm = evaluate(None, cases, ref_feats, in_bq)
    print(f"{'identity(native_clean)':<22}{fm[0]:>9.2f}{fm[1]:>9.2f}{fm[2]:>9.2f}{fm[3]:>9.2f}{fm[4]:>+10.2f}")
    base_fid = fm[0]

    for p in ck_paths:
        model = build_model(DEV)
        ck = torch.load(p, map_location=DEV, weights_only=False)
        model.load_state_dict(ck['model_state']); model.eval()
        r = evaluate(model, cases, ref_feats, in_bq)
        tag = f"{'WIN' if r[0] < base_fid and r[4] <= fm[4] + 0.5 else ''}"
        name = f"{p.parent.name}/{p.name}" if p.parent.name == 'perc' else p.name
        print(f"{name:<22}{r[0]:>9.2f}{r[1]:>9.2f}{r[2]:>9.2f}{r[3]:>9.2f}{r[4]:>+10.2f}  {tag}")


if __name__ == '__main__':
    main()
