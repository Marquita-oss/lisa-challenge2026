"""
evaluate_task2.py — Metricas completas Task 2: DSC, HD, HD95, ASSD, RVE
por estructura y promedio global.

Uso:
    python evaluate_task2.py                        # evalua ResEncL fold_0 (default)
    python evaluate_task2.py --trainer BoundaryLoss # evalua Nivel 1 fold_0
    python evaluate_task2.py --pred_dir PATH --gt_dir PATH
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

LABELS = {
    1: "HippocampusL", 2: "HippocampusR",
    3: "VentricleL",   4: "VentricleR",
    5: "CaudateL",     6: "CaudateR",
    7: "LentiformL",   8: "LentiformR",
    9: "ThalamusL",   10: "ThalamusR",
    11: "CorpusCallosum",
}

RESULTS_BASE = Path(r"C:\Users\rmarcar\Desktop\lisa-challenge2026\nnunet_workspace\nnUNet_results\Dataset001_LISA_Task2")
GT_DIR       = Path(r"C:\Users\rmarcar\Desktop\lisa-challenge2026\nnunet_workspace\nnUNet_raw\Dataset001_LISA_Task2\labelsTr")


# ---------------------------------------------------------------------------
# Metricas de superficie con scipy (sin medpy)
# ---------------------------------------------------------------------------

def surface_distances(pred_bin: np.ndarray, gt_bin: np.ndarray,
                      spacing: tuple) -> np.ndarray:
    """Distancias desde cada voxel de superficie de pred al GT y viceversa."""
    if not pred_bin.any() or not gt_bin.any():
        return None

    # Borde = voxeles dentro de la mascara pero adyacentes al exterior
    pred_border = pred_bin & ~_erode(pred_bin)
    gt_border   = gt_bin   & ~_erode(gt_bin)

    # Distance transform desde el borde del GT hacia todo el volumen
    dt_gt   = distance_transform_edt(~gt_border,   sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    # Distancias superficiales: pred->GT y GT->pred
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    return np.concatenate([d_pred_to_gt, d_gt_to_pred])


def _erode(mask: np.ndarray) -> np.ndarray:
    """Erosion 3x3x3 simple con scipy."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1)


def compute_metrics(pred: np.ndarray, gt: np.ndarray,
                    spacing: tuple) -> dict:
    """DSC, HD, HD95, ASSD, RVE para un par (pred, gt) binarios."""
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    n_pred = tp + fp
    n_gt   = tp + fn

    dsc = 2 * tp / (n_pred + n_gt) if (n_pred + n_gt) > 0 else 0.0
    rve = abs(n_pred - n_gt) / n_gt if n_gt > 0 else 0.0

    if tp == 0:
        # Prediccion vacia o GT vacio — penalizar con valores altos
        return {"DSC": 0.0, "HD": np.nan, "HD95": np.nan, "ASSD": np.nan, "RVE": rve}

    dists = surface_distances(pred, gt, spacing)
    if dists is None:
        return {"DSC": dsc, "HD": np.nan, "HD95": np.nan, "ASSD": np.nan, "RVE": rve}

    return {
        "DSC":  dsc,
        "HD":   float(np.max(dists)),
        "HD95": float(np.percentile(dists, 95)),
        "ASSD": float(np.mean(dists)),
        "RVE":  rve,
    }


# ---------------------------------------------------------------------------
# Evaluacion por caso
# ---------------------------------------------------------------------------

def evaluate_case(pred_path: Path, gt_path: Path) -> dict:
    pred_img = nib.load(str(pred_path))
    gt_img   = nib.load(str(gt_path))

    pred_arr = np.asarray(pred_img.get_fdata(), dtype=np.int32)
    gt_arr   = np.asarray(gt_img.get_fdata(),   dtype=np.int32)

    spacing = tuple(float(x) for x in pred_img.header.get_zooms()[:3])

    case_results = {}
    for lbl, name in LABELS.items():
        pred_bin = (pred_arr == lbl)
        gt_bin   = (gt_arr   == lbl)
        m = compute_metrics(pred_bin, gt_bin, spacing)
        case_results[name] = m

    return case_results


# ---------------------------------------------------------------------------
# Agregacion
# ---------------------------------------------------------------------------

def aggregate(all_results: dict) -> dict:
    """Promedio por estructura y global, ignorando NaN."""
    struct_means = {}
    for struct in LABELS.values():
        vals = {k: [] for k in ("DSC", "HD", "HD95", "ASSD", "RVE")}
        for case_data in all_results.values():
            if struct in case_data:
                for k in vals:
                    v = case_data[struct][k]
                    if not np.isnan(v):
                        vals[k].append(v)
        struct_means[struct] = {k: float(np.mean(v)) if v else float("nan")
                                for k, v in vals.items()}

    global_means = {k: float(np.nanmean([struct_means[s][k] for s in LABELS.values()]))
                    for k in ("DSC", "HD", "HD95", "ASSD", "RVE")}
    return struct_means, global_means


# ---------------------------------------------------------------------------
# Impresion
# ---------------------------------------------------------------------------

def print_results(struct_means: dict, global_means: dict, trainer: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  RESULTADOS: {trainer}")
    print(f"{'=' * 72}")
    print(f"  {'Estructura':<20} {'DSC':>7} {'HD':>8} {'HD95':>8} {'ASSD':>8} {'RVE':>7}")
    print(f"  {'-' * 62}")

    for struct, m in struct_means.items():
        print(f"  {struct:<20} {m['DSC']:>7.4f} {m['HD']:>8.2f} {m['HD95']:>8.2f} "
              f"{m['ASSD']:>8.3f} {m['RVE']:>7.3f}")

    print(f"  {'=' * 62}")
    g = global_means
    print(f"  {'GLOBAL':<20} {g['DSC']:>7.4f} {g['HD']:>8.2f} {g['HD95']:>8.2f} "
          f"{g['ASSD']:>8.3f} {g['RVE']:>7.3f}")

    # Score aproximado del challenge
    score = np.mean([1 - g['DSC'], g['HD95'], g['HD'], g['RVE'], g['ASSD']])
    print(f"\n  Score aprox. challenge (mean de 5 metricas sin normalizar): {score:.4f}")
    print(f"  Nota: el challenge puede normalizar HD/ASSD -- este score es referencial")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", default="nnUNetTrainerResEncL_BoundaryLoss",
                        help="Nombre del trainer (busca en nnUNet_results automaticamente)")
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--pred_dir", default=None,
                        help="Ruta directa a carpeta con predicciones .nii.gz")
    parser.add_argument("--gt_dir", default=str(GT_DIR))
    args = parser.parse_args()

    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
    else:
        pred_dir = (RESULTS_BASE /
                    f"{args.trainer}__nnUNetPlans__{args.config}" /
                    f"fold_{args.fold}" / "validation")

    gt_dir = Path(args.gt_dir)

    print(f"\nPredicciones: {pred_dir}")
    print(f"Ground truth: {gt_dir}")

    pred_files = sorted(pred_dir.glob("lisa_*.nii.gz"))
    if not pred_files:
        print("[ERROR] No se encontraron predicciones en:", pred_dir)
        return

    print(f"Casos a evaluar: {len(pred_files)}")

    all_results = {}
    for pred_path in pred_files:
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"  [SKIP] GT no encontrado: {gt_path.name}")
            continue
        print(f"  Evaluando {case_id}...", end="", flush=True)
        all_results[case_id] = evaluate_case(pred_path, gt_path)
        print(" ok")

    if not all_results:
        print("[ERROR] No se pudo evaluar ningun caso.")
        return

    struct_means, global_means = aggregate(all_results)
    print_results(struct_means, global_means, args.trainer)

    # Guardar JSON
    out_json = pred_dir / "metrics_full.json"
    with open(out_json, "w") as f:
        json.dump({"per_case": all_results, "per_structure": struct_means,
                   "global": global_means}, f, indent=2)
    print(f"\n  Resultados guardados en: {out_json}")


if __name__ == "__main__":
    main()
