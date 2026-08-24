"""
predict_task2.py — Generación de predicciones Task 2, LISA Challenge 2026

Pasos:
  1. Prepara imagesTs/ con las CISO del val set (renombradas a formato nnU-Net)
  2. Corre nnUNetv2_predict con los folds disponibles
  3. Aplica postprocesamiento (elimina floaters, rellena agujeros)
  4. Reporta qué casos pudieron predecirse y cuáles no tienen CISO

Uso:
    conda activate lisa2026
    python predict_task2.py
    python predict_task2.py --folds 0 1 2    # especificar folds si no están todos
    python predict_task2.py --trainer nnUNetTrainerResEncL_BoundaryLoss

Prerequisitos:
    - Al menos fold 0 entrenado con el trainer indicado
    - Variables nnUNet_raw / nnUNet_preprocessed / nnUNet_results definidas
      (se activan automáticamente con: conda activate lisa2026)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label, binary_fill_holes

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
VAL_DIR      = Path(r"C:\Users\rmarcar\Desktop\lisa-challenge2026\data\val")
NNUNET_RAW   = Path(r"C:\Users\rmarcar\Desktop\lisa-challenge2026\nnunet_workspace\nnUNet_raw")
NNUNET_PRE   = Path(r"C:\Users\rmarcar\Desktop\lisa-challenge2026\nnunet_workspace\nnUNet_preprocessed")
NNUNET_RES   = Path(r"C:\Users\rmarcar\Desktop\lisa-challenge2026\nnunet_workspace\nnUNet_results")
DATASET_ID   = "001"
DATASET_NAME = "Dataset001_LISA_Task2"

LABEL_NAMES = {
    1: "HippocampusL", 2: "HippocampusR",
    3: "VentricleL",   4: "VentricleR",
    5: "CaudateL",     6: "CaudateR",
    7: "LentiformL",   8: "LentiformR",
    9: "ThalamusL",   10: "ThalamusR",
    11: "CorpusCallosum",
}
# ---------------------------------------------------------------------------


def find_val_ciso_cases(val_dir: Path) -> tuple[list, list]:
    """
    Separa los casos de val/complete en dos grupos:
      - con_ciso: tienen CISO → predicibles por Task 2
      - sin_ciso: solo tienen LF → no predicibles sin Task 1b
    """
    con_ciso, sin_ciso = [], []
    for case_dir in sorted((val_dir / "complete").iterdir()):
        if not case_dir.is_dir():
            continue
        cid  = case_dir.name
        ciso = case_dir / f"lisa_validation_{cid}_ciso.nii.gz"
        if ciso.exists():
            con_ciso.append((cid, ciso))
        else:
            sin_ciso.append(cid)
    return con_ciso, sin_ciso


def prepare_images_ts(cases: list, nnunet_raw: Path, dataset_name: str) -> Path:
    """
    Crea imagesTs/ con las CISO renombradas al formato nnU-Net:
      lisa_validation_XXXX_ciso.nii.gz → lisa_validation_XXXX_0000.nii.gz
    """
    images_ts = nnunet_raw / dataset_name / "imagesTs"
    images_ts.mkdir(parents=True, exist_ok=True)

    copied = 0
    for cid, ciso_path in cases:
        dest = images_ts / f"lisa_validation_{cid}_0000.nii.gz"
        if not dest.exists():
            shutil.copy2(str(ciso_path), str(dest))
            copied += 1

    print(f"  imagesTs/: {copied} archivos nuevos copiados ({len(cases)} total)")
    return images_ts


def run_nnunet_predict(
    images_ts: Path,
    output_dir: Path,
    dataset_id: str,
    trainer: str,
    folds: list[int],
    config: str = "3d_fullres",
) -> bool:
    """Ejecuta nnUNetv2_predict y devuelve True si tuvo éxito."""
    output_dir.mkdir(parents=True, exist_ok=True)

    folds_str = " ".join(str(f) for f in folds)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(images_ts),
        "-o", str(output_dir),
        "-d", dataset_id,
        "-c", config,
        "-tr", trainer,
        "-f", *[str(f) for f in folds],
        "--save_probabilities",  # guarda probabilidades — útil para ensemble futuro
    ]

    # Inyectar variables nnUNet correctas — el shell puede tener las de otro proyecto
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PRE)
    env["nnUNet_results"]      = str(NNUNET_RES)

    print(f"\n  Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)

    if result.returncode != 0:
        print(f"  [ERROR] nnUNetv2_predict falló (código {result.returncode})")
        return False
    return True


def postprocess_predictions(pred_dir: Path, output_dir: Path) -> dict:
    """
    Aplica postprocesamiento anti-HD a todas las predicciones:
    - Mantiene solo el componente conectado más grande por label
    - Rellena agujeros internos pequeños
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_global = {}

    pred_files = sorted(pred_dir.glob("lisa_validation_*.nii.gz"))
    if not pred_files:
        print("  [AVISO] No se encontraron predicciones en:", pred_dir)
        return {}

    for pred_path in pred_files:
        img = nib.load(str(pred_path))
        seg = img.get_fdata().astype(np.int32)
        seg_clean = seg.copy()
        case_changes = 0

        for lbl in range(1, 12):
            mask = (seg == lbl)
            if not mask.any():
                continue

            labeled, n = cc_label(mask)
            if n > 1:
                sizes = [(labeled == i).sum() for i in range(1, n + 1)]
                largest_idx = int(np.argmax(sizes)) + 1
                mask_clean = (labeled == largest_idx)
                removed = int(mask.sum()) - int(mask_clean.sum())
                if removed > 0:
                    seg_clean[seg == lbl] = 0
                    seg_clean[mask_clean]  = lbl
                    case_changes += removed

            # Rellenar agujeros internos pequeños
            mask_curr = (seg_clean == lbl)
            filled = binary_fill_holes(mask_curr)
            added = int(filled.sum()) - int(mask_curr.sum())
            if 0 < added <= 300:
                seg_clean[filled & ~mask_curr] = lbl
                case_changes += added

        out_img = nib.Nifti1Image(seg_clean.astype(np.int16), img.affine, img.header)
        nib.save(out_img, str(output_dir / pred_path.name))
        stats_global[pred_path.stem] = case_changes

    total_changed = sum(v > 0 for v in stats_global.values())
    print(f"  Postprocesados: {len(stats_global)} casos, {total_changed} modificados")
    return stats_global


def print_summary(cases_with_ciso: list, cases_without_ciso: list,
                  output_dir: Path, trainer: str, folds: list) -> None:
    """Imprime resumen final y próximos pasos."""
    pred_files = list(output_dir.glob("lisa_validation_*.nii.gz"))

    print("\n" + "=" * 60)
    print("  RESUMEN — Task 2 Predictions")
    print("=" * 60)
    print(f"\n  Trainer:       {trainer}")
    print(f"  Folds usados:  {folds}")
    print(f"  Casos con CISO (predicibles): {len(cases_with_ciso)}")
    for cid, _ in cases_with_ciso:
        status = "OK" if (output_dir / f"lisa_validation_{cid}.nii.gz").exists() else "FALTA"
        print(f"    [{status}] validation_{cid}")

    if cases_without_ciso:
        print(f"\n  Casos SIN CISO (requieren Task 1b primero): {len(cases_without_ciso)}")
        for cid in cases_without_ciso:
            print(f"    [PENDIENTE] validation_{cid}")

    print(f"\n  Predicciones generadas en: {output_dir}")
    print(f"  Total archivos .nii.gz: {len(pred_files)}")

    print("\n  PRÓXIMOS PASOS:")
    print("  1. Subir predicciones a la plataforma del challenge (Synapse/Grand Challenge)")
    print("  2. Verificar formato aceptado: .nii.gz, mismo spacing/shape que las CISO de entrada")
    print("  3. Correr fold 0 de ResEncL y Primus para comparar")
    print("  4. Cuando haya 5 folds del ganador -> regenerar predicciones con ensemble")


def main():
    parser = argparse.ArgumentParser(description="Genera predicciones Task 2 — LISA 2026")
    parser.add_argument("--trainer", default="nnUNetTrainerBoundaryLoss",
                        help="Trainer a usar (default: nnUNetTrainerBoundaryLoss)")
    parser.add_argument("--folds", nargs="+", type=int, default=[0],
                        help="Folds a usar (default: [0])")
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Saltar inferencia (solo postprocesar predicciones existentes)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TASK 2 PREDICTION PIPELINE — LISA Challenge 2026")
    print("=" * 60)

    # 1. Encontrar casos con/sin CISO
    print("\n[1/4] Analizando val set...")
    cases_ciso, cases_no_ciso = find_val_ciso_cases(VAL_DIR)
    print(f"  Casos con CISO: {len(cases_ciso)} -> {[c for c,_ in cases_ciso]}")
    print(f"  Casos sin CISO: {len(cases_no_ciso)} -> {cases_no_ciso} (necesitan Task 1b)")

    if not cases_ciso:
        print("  [ERROR] No se encontraron casos con CISO en val/complete/")
        sys.exit(1)

    # 2. Preparar imagesTs
    print("\n[2/4] Preparando imagesTs/...")
    images_ts = prepare_images_ts(cases_ciso, NNUNET_RAW, DATASET_NAME)

    # 3. Inferencia
    trainer_short = args.trainer.replace("nnUNetTrainer", "").replace("nnUNet_", "")
    folds_str = "_".join(str(f) for f in args.folds)
    raw_output = NNUNET_RES / DATASET_NAME / f"predictions_val_{trainer_short}_f{folds_str}"

    if not args.skip_predict:
        print(f"\n[3/4] Ejecutando inferencia (trainer={args.trainer}, folds={args.folds})...")

        # Verificar que el checkpoint del fold existe
        trainer_dir = NNUNET_RES / DATASET_NAME / f"{args.trainer}__nnUNetPlans__{args.config}"
        for fold in args.folds:
            ckpt = trainer_dir / f"fold_{fold}" / "checkpoint_final.pth"
            if not ckpt.exists():
                print(f"  [ERROR] Checkpoint no encontrado: {ckpt}")
                print(f"  Entrenar primero: nnUNetv2_train {DATASET_ID} {args.config} {fold} -tr {args.trainer}")
                sys.exit(1)
            print(f"  Checkpoint fold_{fold}: OK")

        success = run_nnunet_predict(
            images_ts, raw_output, DATASET_ID,
            args.trainer, args.folds, args.config
        )
        if not success:
            sys.exit(1)
    else:
        print(f"\n[3/4] Saltando inferencia — usando predicciones en: {raw_output}")

    # 4. Postprocesamiento
    print("\n[4/4] Aplicando postprocesamiento...")
    final_output = raw_output.parent / f"{raw_output.name}_postproc"
    postprocess_predictions(raw_output, final_output)

    # Guardar JSON de registro
    log = {
        "trainer": args.trainer,
        "folds": args.folds,
        "cases_with_ciso": [c for c, _ in cases_ciso],
        "cases_without_ciso": cases_no_ciso,
        "raw_output": str(raw_output),
        "final_output": str(final_output),
    }
    (final_output / "prediction_log.json").write_text(json.dumps(log, indent=2))

    print_summary(cases_ciso, cases_no_ciso, final_output, args.trainer, args.folds)


if __name__ == "__main__":
    main()
