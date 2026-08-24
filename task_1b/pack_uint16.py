"""Empaqueta un set de NIfTI enhanced a uint16 en un ZIP plano para submission.

Uso:
  python task_1b/pack_uint16.py --src <dir_o_zip> --dst <ruta_zip>
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import nibabel as nib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='dir o zip con *_enhanced.nii.gz')
    ap.add_argument('--dst', required=True, help='ruta del zip de salida')
    args = ap.parse_args()

    src = Path(args.src)
    if src.is_dir():
        in_dir = src
        tmp = None
    else:
        tmp = Path(tempfile.mkdtemp(prefix='pack_'))
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
        in_dir = tmp

    files = sorted(in_dir.rglob('*_enhanced.nii.gz'))
    print(f"Archivos: {len(files)}")
    out_dir = Path(tempfile.mkdtemp(prefix='pack_out_'))

    gmin, gmax, nclip = np.inf, -np.inf, 0
    for i, p in enumerate(files, 1):
        img = nib.load(str(p))
        data = np.asarray(img.get_fdata(dtype=np.float32))
        gmin, gmax = min(gmin, float(data.min())), max(gmax, float(data.max()))
        rounded = np.rint(data)
        clipped = np.clip(rounded, 0, 65535)
        if not np.array_equal(rounded, clipped):
            nclip += 1
        out = clipped.astype(np.uint16)
        new = nib.Nifti1Image(out, img.affine, img.header)
        new.header.set_data_dtype(np.uint16)
        new.header.set_slope_inter(1, 0)
        nib.save(new, str(out_dir / p.name))

    dst = Path(args.dst)
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.glob('*.nii.gz')):
            zf.write(f, arcname=f.name)

    print(f"Rango global [{gmin:.1f}, {gmax:.1f}]  clip={nclip}")
    print(f"ZIP: {dst}  ({dst.stat().st_size/1024/1024:.1f} MB)")
    shutil.rmtree(out_dir, ignore_errors=True)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
