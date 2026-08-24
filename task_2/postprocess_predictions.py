"""
Postprocesamiento de predicciones nnU-Net — LISA Challenge 2026, Task 2

Problema que resuelve:
    HD (Hausdorff Distance) tiene el mismo peso que DSC en el ranking.
    Una sola predicción flotante (componente desconectado) a 50mm de la
    estructura real puede destruir el HD de un caso aunque DSC sea 0.85.

Qué hace:
    Para cada caso predicho, y para cada label 1-11:
    - Mantiene solo el componente conectado más grande (elimina floaters)
    - Rellena agujeros internos pequeños (mejora HD interior)
    - Reporta cuántos vóxeles se eliminaron/añadieron

Uso:
    # Después de nnUNetv2_predict:
    python postprocess_predictions.py \\
        --input  PATH/TO/nnUNet_predictions \\
        --output PATH/TO/nnUNet_predictions_postproc \\
        --labels 1 2 3 4 5 6 7 8 9 10 11

    # Con ground truth para estimar impacto en métricas:
    python postprocess_predictions.py \\
        --input  PATH/TO/predictions \\
        --output PATH/TO/predictions_postproc \\
        --gt     PATH/TO/labelsTr

Estructuras LISA 2026 (labels 1-11):
    1:HippocampusL  2:HippocampusR  3:VentricleL  4:VentricleR
    5:CaudateL      6:CaudateR      7:LentiformL  8:LentiformR
    9:ThalamusL    10:ThalamusR    11:CorpusCallosum
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label, binary_fill_holes

LABEL_NAMES = {
    1: "HippocampusL", 2: "HippocampusR",
    3: "VentricleL",   4: "VentricleR",
    5: "CaudateL",     6: "CaudateR",
    7: "LentiformL",   8: "LentiformR",
    9: "ThalamusL",   10: "ThalamusR",
    11: "CorpusCallosum",
}

# Estructuras que esperamos single-connected (todas bilaterales son 1 por hemisferio)
# Corpus callosum puede tener discontinuidades por resolución — aplicar con tolerancia
SINGLE_COMPONENT_LABELS = set(range(1, 12))

# Umbral mínimo de vóxeles para conservar un componente secundario
# (evitar eliminar estructuras genuinamente fragmentadas por baja resolución)
MIN_SECONDARY_COMPONENT_FRACTION = 0.05  # si un componente > 5% del mayor, alertar


def largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """Devuelve una máscara binaria con solo el componente conectado más grande."""
    if not binary_mask.any():
        return binary_mask.copy()

    labeled, n_components = cc_label(binary_mask)
    if n_components <= 1:
        return binary_mask.copy()

    component_sizes = [
        (labeled == i).sum() for i in range(1, n_components + 1)
    ]
    largest_idx = int(np.argmax(component_sizes)) + 1
    return (labeled == largest_idx).astype(binary_mask.dtype)


def fill_small_holes(binary_mask: np.ndarray, max_hole_voxels: int = 200) -> np.ndarray:
    """
    Rellena agujeros internos pequeños (< max_hole_voxels).
    Útil para mejorar HD en ventrículos y corpus callosum.
    """
    filled = binary_fill_holes(binary_mask)
    holes = filled.astype(int) - binary_mask.astype(int)

    # Solo rellenar si el total de agujeros es manejable
    if holes.sum() <= max_hole_voxels:
        return filled.astype(binary_mask.dtype)
    return binary_mask.copy()


def postprocess_case(
    seg: np.ndarray,
    labels: list[int],
    fill_holes: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Postprocesa un volumen de segmentación completo.

    Returns:
        seg_clean: array postprocesado
        stats: dict con estadísticas de cambios por label
    """
    seg_clean = seg.copy()
    stats = {}

    for lbl in labels:
        mask = (seg == lbl)
        if not mask.any():
            stats[lbl] = {"original": 0, "clean": 0, "removed": 0, "filled": 0}
            continue

        original_vx = int(mask.sum())

        # 1. Componente conectado más grande
        labeled, n = cc_label(mask)
        if n > 1:
            sizes = [(labeled == i).sum() for i in range(1, n + 1)]
            largest_size = max(sizes)
            second_size  = sorted(sizes)[-2] if len(sizes) >= 2 else 0

            if second_size > largest_size * MIN_SECONDARY_COMPONENT_FRACTION and verbose:
                print(f"  [AVISO] Label {lbl} ({LABEL_NAMES.get(lbl,'?')}): "
                      f"componente secundario grande ({second_size} vx = "
                      f"{second_size/largest_size*100:.1f}% del mayor)")

            mask_clean = largest_connected_component(mask)
        else:
            mask_clean = mask.copy()

        # 2. Rellenar agujeros internos pequeños
        filled_vx = 0
        if fill_holes:
            mask_filled = fill_small_holes(mask_clean)
            filled_vx   = int(mask_filled.sum()) - int(mask_clean.sum())
            mask_clean  = mask_filled

        clean_vx = int(mask_clean.sum())
        removed  = original_vx - clean_vx + filled_vx

        # Actualizar segmentación limpia
        seg_clean[seg == lbl]      = 0
        seg_clean[mask_clean > 0]  = lbl

        stats[lbl] = {
            "name":     LABEL_NAMES.get(lbl, f"label_{lbl}"),
            "original": original_vx,
            "clean":    clean_vx,
            "removed":  original_vx - (clean_vx - filled_vx),
            "filled":   filled_vx,
        }

    return seg_clean, stats


def compute_hd_estimate(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """
    Estimación rápida del HD para un label dado.
    Usa distancias entre centroides de componentes como proxy.
    """
    from scipy.ndimage import center_of_mass, distance_transform_edt

    pred_mask = (pred == label)
    gt_mask   = (gt   == label)

    if not pred_mask.any() or not gt_mask.any():
        return float('inf')

    # Distancia de superficie
    pred_surface = pred_mask ^ binary_fill_holes(pred_mask)
    gt_surface   = gt_mask   ^ binary_fill_holes(gt_mask)

    if not pred_surface.any() or not gt_surface.any():
        return 0.0

    dist_pred_to_gt = distance_transform_edt(~gt_surface)[pred_surface]
    dist_gt_to_pred = distance_transform_edt(~pred_surface)[gt_surface]

    hd = float(max(dist_pred_to_gt.max(), dist_gt_to_pred.max()))
    return hd


def main():
    parser = argparse.ArgumentParser(
        description="Postprocesa predicciones nnU-Net para mejorar HD/HD95/ASSD"
    )
    parser.add_argument("--input",  required=True,
                        help="Carpeta con predicciones nnU-Net (.nii.gz)")
    parser.add_argument("--output", required=True,
                        help="Carpeta destino para predicciones postprocesadas")
    parser.add_argument("--gt",     default=None,
                        help="(Opcional) Carpeta con GT para estimar impacto en HD")
    parser.add_argument("--labels", nargs="+", type=int,
                        default=list(range(1, 12)),
                        help="Labels a postprocesar (default: 1-11)")
    parser.add_argument("--no-fill-holes", action="store_true",
                        help="Desactivar relleno de agujeros internos")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(input_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"No se encontraron archivos .nii.gz en {input_dir}")
        return

    print(f"Postprocesando {len(pred_files)} casos...")
    print(f"Labels: {args.labels}")
    print(f"Relleno de agujeros: {not args.no_fill_holes}")
    print()

    all_stats = {}
    hd_before_all = {}
    hd_after_all  = {}

    for pred_path in pred_files:
        case_id = pred_path.name.replace(".nii.gz", "")
        img     = nib.load(str(pred_path))
        seg     = img.get_fdata().astype(np.int32)

        # GT para comparación (opcional)
        gt_seg = None
        if args.gt:
            gt_path = Path(args.gt) / pred_path.name
            if not gt_path.exists():
                # nnU-Net labelsTr usa nombre sin _0000
                gt_path = Path(args.gt) / f"{case_id}.nii.gz"
            if gt_path.exists():
                gt_seg = nib.load(str(gt_path)).get_fdata().astype(np.int32)

        # HD antes del postproceso (solo hipocampos — los más críticos en challenge)
        if gt_seg is not None:
            hd_before_all[case_id] = {
                lbl: compute_hd_estimate(seg, gt_seg, lbl)
                for lbl in [1, 2]  # HippocampusL, HippocampusR
            }

        # Postprocesar
        seg_clean, stats = postprocess_case(
            seg, args.labels,
            fill_holes=not args.no_fill_holes,
            verbose=args.verbose,
        )

        # HD después
        if gt_seg is not None:
            hd_after_all[case_id] = {
                lbl: compute_hd_estimate(seg_clean, gt_seg, lbl)
                for lbl in [1, 2]
            }

        # Guardar
        out_img = nib.Nifti1Image(seg_clean.astype(np.int16), img.affine, img.header)
        nib.save(out_img, str(output_dir / pred_path.name))

        # Reportar cambios significativos
        removed_total = sum(s.get("removed", 0) for s in stats.values())
        filled_total  = sum(s.get("filled",  0) for s in stats.values())

        if removed_total > 0 or filled_total > 0:
            print(f"  {case_id}: -{removed_total} vx eliminados, +{filled_total} vx rellenos")
            if args.verbose:
                for lbl, s in stats.items():
                    if s.get("removed", 0) > 0 or s.get("filled", 0) > 0:
                        print(f"    Label {lbl} ({s['name']}): "
                              f"{s['original']} → {s['clean']} vx")

        all_stats[case_id] = stats

    # Resumen global
    print(f"\n{'='*50}")
    print("RESUMEN")
    print(f"{'='*50}")
    total_removed = sum(
        s.get("removed", 0)
        for case_stats in all_stats.values()
        for s in case_stats.values()
    )
    total_filled = sum(
        s.get("filled", 0)
        for case_stats in all_stats.values()
        for s in case_stats.values()
    )
    print(f"Total vóxeles eliminados (floaters): {total_removed}")
    print(f"Total vóxeles rellenados (agujeros): {total_filled}")
    print(f"Casos modificados: {sum(1 for cs in all_stats.values() if any(s.get('removed',0)+s.get('filled',0) > 0 for s in cs.values()))}/{len(pred_files)}")

    # Impacto en HD (si hay GT)
    if hd_before_all and hd_after_all:
        print(f"\n{'='*50}")
        print("IMPACTO EN HD (Hipocampos — estructuras críticas)")
        print(f"{'='*50}")
        for lbl_id, lbl_name in [(1, "HippocampusL"), (2, "HippocampusR")]:
            before_vals = [v[lbl_id] for v in hd_before_all.values() if lbl_id in v and v[lbl_id] != float('inf')]
            after_vals  = [v[lbl_id] for v in hd_after_all.values()  if lbl_id in v and v[lbl_id] != float('inf')]
            if before_vals and after_vals:
                print(f"  {lbl_name}: HD antes={np.mean(before_vals):.2f}mm → después={np.mean(after_vals):.2f}mm")

    # Guardar estadísticas JSON
    stats_path = output_dir / "postprocess_stats.json"
    with open(str(stats_path), "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nEstadísticas guardadas en: {stats_path}")
    print(f"Predicciones postprocesadas en: {output_dir}")


if __name__ == "__main__":
    main()
