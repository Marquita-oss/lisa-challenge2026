"""FID por-caso (piq + InceptionV3) para localizar casos catastroficos.

El scorer reporta FID_min/mean/max => evalua por sujeto. Aqui reproducimos eso
localmente sobre los 14 casos `complete` (unicos con CISO disponible): para cada
caso, FID entre sus slices enhanced (axi+cor+sag) y sus slices CISO.

Uso:
  python task_1b/eval_fid_percase.py --pred <dir_o_zip> [--label X]
"""
import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from skimage.transform import resize as skr
from piq import FID
from piq.feature_extractors import InceptionV3

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task_1b.config import VAL_COMPLETE_DIR

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_INC = InceptionV3().to(DEV).eval()


def npct(s):
    p1, p99 = np.percentile(s, [1, 99]); d = p99 - p1
    return np.clip((s - p1) / (d if d > 1e-8 else 1), 0, 1).astype(np.float32)


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
def feats(slice_list, batch=64):
    fs = []
    for i in range(0, len(slice_list), batch):
        chunk = slice_list[i:i + batch]
        t = torch.stack([torch.from_numpy(skr(s, (128, 128), order=1,
                         preserve_range=True).astype(np.float32)) for s in chunk])
        t = t[:, None].repeat(1, 3, 1, 1).to(DEV)
        f = _INC(t)[0].squeeze(-1).squeeze(-1)
        fs.append(f.cpu())
    return torch.cat(fs)


def collect(pred):
    p = Path(pred)
    if p.is_dir():
        return p
    tmp = Path(tempfile.mkdtemp(prefix='fid_'))
    with zipfile.ZipFile(p) as zf:
        zf.extractall(tmp)
    return tmp


def case_id(name):
    m = re.search(r'(\d{4})', name)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--label', default='pred')
    args = ap.parse_args()

    d = collect(args.pred)
    # group enhanced files by case
    enh_by_case = {}
    for f in sorted(d.rglob('*_enhanced.nii.gz')):
        cid = case_id(f.name)
        if cid is None:
            continue
        enh_by_case.setdefault(cid, []).append(f)

    rows = []
    for cdir in sorted(VAL_COMPLETE_DIR.iterdir()):
        if not cdir.is_dir():
            continue
        cid = cdir.name
        ciso = list(cdir.glob('*_ciso.nii.gz'))
        if not ciso or cid not in enh_by_case:
            continue
        ref_sl = slices(nib.load(str(ciso[0])).get_fdata(dtype=np.float32))
        pred_sl = []
        for f in enh_by_case[cid]:
            pred_sl += slices(nib.load(str(f)).get_fdata(dtype=np.float32))
        pf = feats(pred_sl); rf = feats(ref_sl)
        fid = float(FID()(pf, rf).item())
        rows.append((cid, fid, len(pred_sl), len(ref_sl)))
        print(f"  case {cid}: FID {fid:8.2f}  (enh {len(pred_sl):3d} sl, ciso {len(ref_sl):3d} sl)")

    fids = [r[1] for r in rows]
    print(f"\n{args.label}:  mean {np.mean(fids):.2f}  min {np.min(fids):.2f}  max {np.max(fids):.2f}  (n={len(rows)})")
    print("Worst cases:")
    for cid, fid, *_ in sorted(rows, key=lambda r: -r[1])[:5]:
        print(f"  {cid}: {fid:.2f}")


if __name__ == '__main__':
    main()
