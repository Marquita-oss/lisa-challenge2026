"""Probe: cuanto mejora la alineacion CISO->LF con registro real vs array-resize.

Compara 3 metodos de llevar CISO a la grilla de cada plano LF y mide la calidad
de alineacion (correlacion e informacion mutua sobre foreground):
  A. array-resize   : sk_resize del volumen CISO a la forma LF (lo que hizo lf2ciso; ~0.6 esperado)
  B. world-resample : resample SITK usando los affines de las cabeceras (espacio fisico)
  C. rigid+MI       : B seguido de registro rigido (Mattes MI, multi-res)

Solo casos complete (tienen CISO). Salida: tabla por caso/plano y medias.

Uso:
  python task_1b/probe_registration.py [--cases 0001 0002 ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize as sk_resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task_1b.config import VAL_COMPLETE_DIR


def norm01(a):
    p1, p99 = np.percentile(a, [1, 99]); d = p99 - p1
    return np.clip((a - p1) / (d if d > 1e-8 else 1), 0, 1).astype(np.float32)


def corr_mi(a, b, bins=32):
    """Pearson corr y MI sobre foreground (a>0.05) de dos arrays alineados."""
    a = a.ravel(); b = b.ravel()
    m = (a > 0.05) & (b > 0.05)
    if m.sum() < 100:
        return np.nan, np.nan
    av, bv = a[m], b[m]
    c = float(np.corrcoef(av, bv)[0, 1])
    hist, _, _ = np.histogram2d(av, bv, bins=bins)
    pxy = hist / hist.sum()
    px = pxy.sum(1); py = pxy.sum(0)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])).sum())
    return c, mi


def to_sitk(nii):
    img = sitk.GetImageFromArray(np.ascontiguousarray(
        nii.get_fdata(dtype=np.float32).transpose(2, 1, 0)))
    zooms = nii.header.get_zooms()[:3]
    img.SetSpacing([float(zooms[0]), float(zooms[1]), float(zooms[2])])
    # origin/direction from affine (nibabel RAS -> SITK LPS)
    aff = nii.affine
    R = aff[:3, :3].copy()
    spacing = np.array(zooms, float)
    dir_cos = (R / spacing).T.flatten()
    ras2lps = np.diag([-1, -1, 1]).astype(float)
    origin = ras2lps @ aff[:3, 3]
    D = np.array(dir_cos).reshape(3, 3)
    D = (ras2lps @ D.T).T
    img.SetOrigin([float(x) for x in origin])
    img.SetDirection([float(x) for x in D.flatten()])
    return img


def world_resample(ciso_img, lf_img, transform=None):
    tx = transform if transform is not None else sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(ciso_img, lf_img, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='*', default=None)
    args = ap.parse_args()

    dirs = sorted(d for d in VAL_COMPLETE_DIR.iterdir() if d.is_dir())
    if args.cases:
        dirs = [d for d in dirs if d.name in args.cases]

    agg = {'A': [], 'B': [], 'C': []}
    for d in dirs:
        cid = d.name
        cp = d / f'lisa_validation_{cid}_ciso.nii.gz'
        if not cp.exists():
            continue
        ciso_nii = nib.load(str(cp))
        ciso_arr = norm01(ciso_nii.get_fdata(dtype=np.float32))
        ciso_img = to_sitk(ciso_nii)
        for pl in ['axi', 'cor', 'sag']:
            lp = d / f'lisa_validation_{cid}_lf_{pl}.nii.gz'
            if not lp.exists():
                continue
            lf_nii = nib.load(str(lp))
            lf_arr = norm01(lf_nii.get_fdata(dtype=np.float32))
            lf_img = to_sitk(lf_nii)

            # A: array resize
            a_res = norm01(sk_resize(ciso_arr, lf_arr.shape, order=1,
                                     preserve_range=True, anti_aliasing=True))
            cA, _ = corr_mi(a_res, lf_arr)

            # B: world resample (affine)
            b_img = world_resample(ciso_img, lf_img)
            b_arr = norm01(sitk.GetArrayFromImage(b_img).transpose(2, 1, 0))
            cB, _ = corr_mi(b_arr, lf_arr)

            # C: rigid + MI
            try:
                tx = rigid_register(ciso_img, lf_img)
                c_img = world_resample(ciso_img, lf_img, tx)
                c_arr = norm01(sitk.GetArrayFromImage(c_img).transpose(2, 1, 0))
                cC, _ = corr_mi(c_arr, lf_arr)
            except Exception as e:
                cC = np.nan
            agg['A'].append(cA); agg['B'].append(cB); agg['C'].append(cC)
            print(f"{cid} {pl}:  A(resize) {cA:+.3f}   B(world) {cB:+.3f}   C(rigid) {cC:+.3f}")

    print("\n=== mean correlation over all case/planes ===")
    for k, lab in [('A', 'array-resize'), ('B', 'world-resample'), ('C', 'rigid+MI')]:
        v = np.array(agg[k], float)
        print(f"  {lab:16} {np.nanmean(v):+.3f}  (n={np.isfinite(v).sum()})")


if __name__ == '__main__':
    main()
