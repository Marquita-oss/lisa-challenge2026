"""
Barrido de postproceso medido contra las 5 métricas del ranking LISA Task 2.

Métricas: DSC, HD (max), HD95, ASSD (mm), RVE (%).
Se reportan SOLO sobre las 9 estructuras PUNTUADAS (los ventrículos 3,4 NO se
puntúan segun la wiki oficial) y promediando L/R por estructura, que es la
unidad de evaluación del challenge ("average value of each metric across both
sides").

Datos: predicciones OOF del ensemble 5-fold (crossval_results_folds_0_1_2_3_4)
vs GT de labelsTr (79 casos con ground truth).

Uso:
    python task_2/eval_postproc_sweep.py
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label, binary_fill_holes

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "nnunet_workspace/nnUNet_results/Dataset001_LISA_Task2/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
OOF = RES / "crossval_results_folds_0_1_2_3_4"
GT_DIR = REPO / "nnunet_workspace/nnUNet_raw/Dataset001_LISA_Task2/labelsTr"

ALL_LABELS = list(range(1, 12))
SCORED = [1, 2, 5, 6, 7, 8, 9, 10, 11]          # ventrículos 3,4 NO puntuan
PKL_LABELS = {3, 4, 7}                            # lo que limpia el pkl actual
# estructuras (para promediar L/R): nombre -> labels
STRUCTS = {"Hippocampus": [1, 2], "Caudate": [5, 6], "Lentiform": [7, 8],
           "Thalamus": [9, 10], "CorpusCallosum": [11]}
VARIANTS = ["raw", "pkl", "lcc_all", "frac30", "abs50", "pkl+abs50"]
METRICS = ("dice", "hd", "hd95", "assd", "rve")


def largest_cc(mask):
    if not mask.any():
        return mask
    lab, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()


def remove_minor_components(mask, frac=None, abs_vox=None):
    """Mantiene el mayor + cualquier componente grande; quita solo los pequeños.
    Un componente se ELIMINA si su tamaño < frac*mayor  Y/O  < abs_vox vóxeles."""
    if not mask.any():
        return mask
    lab, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    largest = sizes.max()
    keep = np.zeros_like(mask)
    for comp in range(1, n + 1):
        sz = sizes[comp]
        if sz == largest:
            keep |= lab == comp
            continue
        small = False
        if frac is not None and sz < frac * largest:
            small = True
        if abs_vox is not None and sz < abs_vox:
            small = True
        if not small:
            keep |= lab == comp
    return keep


def apply_pp(seg, variant):
    if variant == "raw":
        return seg
    out = seg.copy()
    if variant == "pkl":              # largest-CC solo en 3,4,7 (actual)
        for lbl in PKL_LABELS:
            m = seg == lbl
            if m.any():
                out[seg == lbl] = 0; out[largest_cc(m)] = lbl
        return out
    if variant == "lcc_all":          # largest-CC en los 11
        for lbl in ALL_LABELS:
            m = seg == lbl
            if m.any():
                out[seg == lbl] = 0; out[largest_cc(m)] = lbl
        return out
    # variantes "solo flotantes" en todos los labels
    cfg = {"frac30": dict(frac=0.30), "abs50": dict(abs_vox=50),
           "pkl+abs50": dict(abs_vox=50)}[variant]
    for lbl in ALL_LABELS:
        m = seg == lbl
        if not m.any():
            continue
        if variant == "pkl+abs50" and lbl in PKL_LABELS:
            m2 = largest_cc(m)
        else:
            m2 = remove_minor_components(m, **cfg)
        out[seg == lbl] = 0; out[m2] = lbl
    return out


def metrics_for_label(pred_m, gt_m, spacing):
    p, g = pred_m.sum(), gt_m.sum()
    inter = np.logical_and(pred_m, gt_m).sum()
    dice = 2 * inter / (p + g) if (p + g) > 0 else 1.0
    rve = abs(p - g) / g * 100 if g > 0 else (0.0 if p == 0 else 100.0)
    if p == 0 or g == 0:
        return dict(dice=dice, hd=np.nan, hd95=np.nan, assd=np.nan, rve=rve)
    pi = sitk.GetImageFromArray(pred_m.astype(np.uint8)); pi.SetSpacing(spacing)
    gi = sitk.GetImageFromArray(gt_m.astype(np.uint8));   gi.SetSpacing(spacing)
    ps = sitk.GetArrayFromImage(sitk.LabelContour(pi, False)).astype(bool)
    gs = sitk.GetArrayFromImage(sitk.LabelContour(gi, False)).astype(bool)
    pdm = sitk.GetArrayFromImage(sitk.Abs(sitk.SignedMaurerDistanceMap(pi, squaredDistance=False, useImageSpacing=True)))
    gdm = sitk.GetArrayFromImage(sitk.Abs(sitk.SignedMaurerDistanceMap(gi, squaredDistance=False, useImageSpacing=True)))
    alld = np.concatenate([gdm[ps], pdm[gs]])
    return dict(dice=dice, hd=float(alld.max()), hd95=float(np.percentile(alld, 95)),
                assd=float(alld.mean()), rve=rve)


def main():
    cases = sorted(p.name for p in GT_DIR.glob("lisa_*.nii.gz"))
    print(f"Casos OOF con GT: {len(cases)}")
    # store[variant][label] -> {metric: [valores por caso]}
    store = {v: {l: {m: [] for m in METRICS} for l in ALL_LABELS} for v in VARIANTS}

    for ci, name in enumerate(cases):
        gt_img = sitk.ReadImage(str(GT_DIR / name))
        spacing = gt_img.GetSpacing()
        gt = sitk.GetArrayFromImage(gt_img).astype(np.int16)
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(OOF / name))).astype(np.int16)
        for v in VARIANTS:
            segpp = apply_pp(pred, v)
            for lbl in ALL_LABELS:
                mm = metrics_for_label(segpp == lbl, gt == lbl, spacing)
                for k in METRICS:
                    store[v][lbl][k].append(mm[k])
        print(f"  [{ci+1}/{len(cases)}] {name}        ", end="\r", flush=True)
    print()

    def lbl_mean(v, lbl, m):
        return float(np.nanmean(store[v][lbl][m]))

    # --- tabla 1: media sobre las 9 estructuras puntuadas ---
    print(f"\n=== MEDIA sobre 9 estructuras PUNTUADAS (ventrículos excluidos) ===")
    print(f"{'variant':<10} {'DSC↑':>7} {'HD↓':>8} {'HD95↓':>8} {'ASSD↓':>8} {'RVE%↓':>8}")
    print("-" * 55)
    scored_rows = {}
    for v in VARIANTS:
        r = {m: float(np.mean([lbl_mean(v, l, m) for l in SCORED])) for m in METRICS}
        scored_rows[v] = r
        print(f"{v:<10} {r['dice']:>7.4f} {r['hd']:>8.3f} {r['hd95']:>8.3f} {r['assd']:>8.3f} {r['rve']:>8.2f}")

    base = scored_rows["pkl"]
    print(f"\nΔ vs pkl(actual) — negativo = mejor en HD/HD95/ASSD/RVE, positivo = mejor en DSC:")
    for v in VARIANTS:
        if v == "pkl":
            continue
        r = scored_rows[v]
        print(f"  {v:<10} DSC {r['dice']-base['dice']:+.4f}  HD {r['hd']-base['hd']:+.3f}  "
              f"HD95 {r['hd95']-base['hd95']:+.3f}  ASSD {r['assd']-base['assd']:+.3f}  "
              f"RVE {r['rve']-base['rve']:+.2f}")

    # --- tabla 2: por estructura (L/R promediado), pkl vs pkl+abs50 ---
    cmp_v = "pkl+abs50"
    print(f"\n=== POR ESTRUCTURA (L/R promediado) — DSC | HD95 | ASSD ===")
    print(f"{'estructura':<16} {'pkl':>22}   {cmp_v:>22}")
    for sname, lbls in STRUCTS.items():
        def s(v, m): return float(np.mean([lbl_mean(v, l, m) for l in lbls]))
        a = f"D{s('pkl','dice'):.3f} H{s('pkl','hd95'):.2f} A{s('pkl','assd'):.2f}"
        b = f"D{s(cmp_v,'dice'):.3f} H{s(cmp_v,'hd95'):.2f} A{s(cmp_v,'assd'):.2f}"
        print(f"{sname:<16} {a:>22}   {b:>22}")

    out = {"scored_mean": scored_rows,
           "per_label": {v: {l: {m: lbl_mean(v, l, m) for m in METRICS} for l in ALL_LABELS} for v in VARIANTS}}
    p = REPO / "task_2/qc_output/postproc_sweep.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nGuardado: {p}")


if __name__ == "__main__":
    main()
