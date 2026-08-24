"""FID local (piq + InceptionV3) entre slices enhanced y la distribucion CISO.

Proxy de la metrica dominante del scorer. Se valida comprobando que ordena
correctamente submissions conocidas (baseline < LF2CISO).

Uso:
  python task_1b/eval_fid.py --pred <dir_o_zip> [--label X] [--ref complete|train]
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
from task_1b.config import VAL_COMPLETE_DIR, TRAIN_DIR

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
        # resize a 128 para homogeneizar (Inception reescala a 299 internamente)
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


def ref_slices(which):
    base = VAL_COMPLETE_DIR if which == 'complete' else TRAIN_DIR
    sl = []
    for cdir in sorted(base.iterdir()):
        if not cdir.is_dir():
            continue
        ciso = list(cdir.glob('*_ciso.nii.gz'))
        if not ciso:
            continue
        sl += slices(nib.load(str(ciso[0])).get_fdata(dtype=np.float32))
    return sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True)
    ap.add_argument('--label', default='pred')
    ap.add_argument('--ref', default='complete', choices=['complete', 'train'])
    args = ap.parse_args()

    d = collect(args.pred)
    pred_sl = []
    for f in sorted(d.rglob('*_enhanced.nii.gz')):
        pred_sl += slices(nib.load(str(f)).get_fdata(dtype=np.float32))
    ref_sl = ref_slices(args.ref)
    print(f"{args.label}: pred slices {len(pred_sl)}, ref CISO slices {len(ref_sl)}")

    pf = feats(pred_sl)
    rf = feats(ref_sl)
    fid = float(FID()(pf, rf).item())
    print(f"  FID_local({args.ref}) = {fid:.2f}")


if __name__ == '__main__':
    main()
