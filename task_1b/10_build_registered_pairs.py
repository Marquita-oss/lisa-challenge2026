"""Opcion 2 (fase 1): construir pares LF<->CISO REGISTRADOS y cachearlos.

Para cada caso con CISO (train + val/complete), registra CISO -> cada plano LF
con rigido+Mattes MI (validado en probe_registration.py: corr 0.36 -> 0.63),
resamplea CISO a la grilla LF y guarda pares de slices alineados a .npz.

Salida:
  task_1b/reg_cache/<split>/<case>_<plane>.npz  (lf: [N,H,W] f16, ci: [N,H,W] f16)
  task_1b/reg_cache/registration_qc.csv         (corr por caso/plano)

Uso:
  python task_1b/10_build_registered_pairs.py --split train
  python task_1b/10_build_registered_pairs.py --split val
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task_1b.config import TRAIN_DIR, VAL_COMPLETE_DIR, TASK_DIR

CACHE = TASK_DIR / 'reg_cache'
MIN_FG = 0.03


def norm01(a):
    p1, p99 = np.percentile(a, [1, 99]); d = p99 - p1
    return np.clip((a - p1) / (d if d > 1e-8 else 1), 0, 1).astype(np.float32)


def corr_fg(a, b):
    a = a.ravel(); b = b.ravel()
    m = (a > 0.05) & (b > 0.05)
    if m.sum() < 100:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def to_sitk(nii):
    img = sitk.GetImageFromArray(np.ascontiguousarray(
        nii.get_fdata(dtype=np.float32).transpose(2, 1, 0)))
    zooms = nii.header.get_zooms()[:3]
    img.SetSpacing([float(z) for z in zooms])
    aff = nii.affine
    spacing = np.array(zooms, float)
    D = (aff[:3, :3] / spacing).T
    ras2lps = np.diag([-1, -1, 1]).astype(float)
    origin = ras2lps @ aff[:3, 3]
    D = (ras2lps @ D.T).T
    img.SetOrigin([float(x) for x in origin])
    img.SetDirection([float(x) for x in D.flatten()])
    return img


def rigid_register(ciso_img, lf_img):
    fixed = sitk.Cast(lf_img, sitk.sitkFloat32)
    moving = sitk.Cast(ciso_img, sitk.sitkFloat32)
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2, seed=42)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=120,
                                    convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(init, inPlace=False)
    return R.Execute(fixed, moving)


def resample(ciso_img, lf_img, tx):
    return sitk.Resample(ciso_img, lf_img, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def process_case(cdir, prefix, out_dir):
    cid = cdir.name
    ciso_p = cdir / f'{prefix}_{cid}_ciso.nii.gz'
    if not ciso_p.exists():
        return []
    ciso_nii = nib.load(str(ciso_p))
    ciso_img = to_sitk(ciso_nii)
    rows = []
    for plane in ['axi', 'cor', 'sag']:
        lf_p = cdir / f'{prefix}_{cid}_lf_{plane}.nii.gz'
        if not lf_p.exists():
            continue
        lf_nii = nib.load(str(lf_p))
        lf_vol = lf_nii.get_fdata(dtype=np.float32)
        if lf_vol.ndim != 3:
            continue
        lf_img = to_sitk(lf_nii)
        try:
            tx = rigid_register(ciso_img, lf_img)
            ci_reg = sitk.GetArrayFromImage(resample(ciso_img, lf_img, tx)).transpose(2, 1, 0)
        except Exception as e:
            print(f"    {cid} {plane}: registration FAILED ({e}); skipping")
            continue

        thin = int(np.argmin(lf_vol.shape))
        n = lf_vol.shape[thin]
        lf_slices, ci_slices, corrs = [], [], []
        for i in range(n):
            idx = [slice(None)] * 3; idx[thin] = i
            lf_s = lf_vol[tuple(idx)]
            ci_s = ci_reg[tuple(idx)]
            if lf_s.max() <= 1e-6:
                continue
            lf_n = norm01(lf_s)
            if float((lf_n > 0.1).mean()) < MIN_FG:
                continue
            ci_n = norm01(ci_s)
            lf_slices.append(lf_n.astype(np.float16))
            ci_slices.append(ci_n.astype(np.float16))
            corrs.append(corr_fg(ci_n, lf_n))
        if not lf_slices:
            continue
        np.savez_compressed(out_dir / f'{cid}_{plane}.npz',
                            lf=np.stack(lf_slices), ci=np.stack(ci_slices))
        mc = float(np.nanmean(corrs))
        rows.append((cid, plane, len(lf_slices), mc))
        print(f"    {cid} {plane}: {len(lf_slices):3d} slices  corr {mc:+.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'val'])
    args = ap.parse_args()

    if args.split == 'train':
        base, prefix = TRAIN_DIR, 'lisa'
    else:
        base, prefix = VAL_COMPLETE_DIR, 'lisa_validation'

    out_dir = CACHE / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(d for d in base.iterdir() if d.is_dir()
                   and list(d.glob('*_ciso.nii.gz')))
    print(f"Split={args.split}  casos con CISO: {len(cases)}  -> {out_dir}")

    all_rows, t0 = [], time.time()
    for k, cdir in enumerate(cases, 1):
        print(f"[{k}/{len(cases)}] {cdir.name}")
        all_rows += process_case(cdir, prefix, out_dir)

    qc = CACHE / 'registration_qc.csv'
    header = 'split,case,plane,n_slices,corr\n'
    lines = [f"{args.split},{c},{p},{n},{cc:.4f}" for c, p, n, cc in all_rows]
    mode = 'a' if qc.exists() else 'w'
    with open(qc, mode) as f:
        if mode == 'w':
            f.write(header)
        f.write('\n'.join(lines) + '\n')

    corrs = np.array([r[3] for r in all_rows], float)
    print(f"\nDONE {args.split}: {len(all_rows)} planes  mean corr {np.nanmean(corrs):+.3f}  "
          f"({time.time()-t0:.0f}s)  QC-> {qc}")


if __name__ == '__main__':
    main()
