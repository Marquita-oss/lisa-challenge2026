"""
Convierte el dataset LISA Task 2 al formato nnU-Net v2.

Uso:
    python prepare_nnunet.py

Prerequisitos:
    pip install nibabel numpy nnunetv2

Variables de entorno requeridas por nnU-Net (ajustar según tu máquina):
    set nnUNet_raw=C:/Users/rmarcar/Desktop/lisa-challenge2026/nnunet_workspace/nnUNet_raw
    set nnUNet_preprocessed=C:/Users/rmarcar/Desktop/lisa-challenge2026/nnunet_workspace/nnUNet_preprocessed
    set nnUNet_results=C:/Users/rmarcar/Desktop/lisa-challenge2026/nnunet_workspace/nnUNet_results
"""

import json
import shutil
import sys
import numpy as np
import nibabel as nib
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURACIÓN — ajustar si es necesario
# ---------------------------------------------------------------------------
TRAIN_DIR   = Path(r"C:/Users/rmarcar/Desktop/lisa-challenge2026/data/train")
NNUNET_RAW  = Path(r"C:/Users/rmarcar/Desktop/lisa-challenge2026/nnunet_workspace/nnUNet_raw")
DATASET_ID  = "001"
DATASET_NAME = "LISA_Task2"

# Mapeo oficial — Synapse wiki syn72118611, LISA Challenge 2026
# Fuente: wiki del proyecto, última modificación 05/06/2026
# Nota: "Lentiform" = núcleo lenticular = putamen + globo pálido combinados
LABEL_MAP = {
    "background":      0,
    "HippocampusL":    1,
    "HippocampusR":    2,
    "VentricleL":      3,
    "VentricleR":      4,
    "CaudateL":        5,
    "CaudateR":        6,
    "LentiformL":      7,
    "LentiformR":      8,
    "ThalamusL":       9,
    "ThalamusR":       10,
    "CorpusCallosum":  11,
}
# ---------------------------------------------------------------------------


def find_g1_cases(train_dir: Path) -> list[tuple[str, Path, Path]]:
    """Retorna lista de (case_id, ciso_path, seg_path) para los 79 casos G1."""
    cases = []
    for case_dir in sorted(train_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        cid = case_dir.name
        ciso = case_dir / f"lisa_{cid}_ciso.nii.gz"
        seg  = case_dir / f"lisa_{cid}_seg.nii.gz"
        if ciso.exists() and seg.exists():
            cases.append((cid, ciso, seg))
    return cases


def verify_labels(cases: list, expected_labels: set) -> bool:
    """Verifica que todos los casos tengan exactamente los labels esperados."""
    print("\n[1/4] Verificando consistencia de labels...")
    errors = []
    for cid, _, seg_path in cases:
        data = nib.load(str(seg_path)).get_fdata().astype(np.int32)
        found = set(np.unique(data).tolist())
        if not found.issubset(expected_labels | {0}):
            extra = found - expected_labels - {0}
            errors.append(f"  CASO {cid}: labels inesperados {extra}")
    if errors:
        print("ERRORES de labels:")
        for e in errors:
            print(e)
        return False
    print(f"  OK — todos los casos tienen labels dentro de {sorted(expected_labels)}")
    return True


def verify_spacing(cases: list) -> bool:
    """Verifica que CISO y SEG tengan spacing 1mm isométrico y shapes coincidentes."""
    print("\n[2/4] Verificando spacing y shapes...")
    errors = []
    shapes = set()
    for cid, ciso_path, seg_path in cases:
        ciso_img = nib.load(str(ciso_path))
        seg_img  = nib.load(str(seg_path))

        # Shape debe coincidir
        if ciso_img.shape != seg_img.shape:
            errors.append(f"  CASO {cid}: shape mismatch CISO={ciso_img.shape} SEG={seg_img.shape}")
            continue

        # Spacing (tolerancia 0.01 mm)
        zooms = ciso_img.header.get_zooms()
        if not all(abs(z - 1.0) < 0.01 for z in zooms[:3]):
            errors.append(f"  CASO {cid}: spacing no es 1mm isotrópico: {zooms}")

        shapes.add(ciso_img.shape)

    if errors:
        print("ERRORES de geometría:")
        for e in errors:
            print(e)
        return False

    print(f"  OK — shapes presentes: {shapes}")
    print(f"  OK — spacing 1mm isométrico en todos los casos")
    return True


def build_dataset_dir(dataset_id: str, dataset_name: str, nnunet_raw: Path) -> tuple[Path, Path, Path]:
    """Crea la estructura de directorios de nnU-Net."""
    dataset_dir = nnunet_raw / f"Dataset{dataset_id}_{dataset_name}"
    images_tr   = dataset_dir / "imagesTr"
    labels_tr   = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    return dataset_dir, images_tr, labels_tr


def copy_files(cases: list, images_tr: Path, labels_tr: Path) -> None:
    """Copia archivos con el naming correcto de nnU-Net v2."""
    print(f"\n[3/4] Copiando {len(cases)} casos...")
    for i, (cid, ciso_path, seg_path) in enumerate(cases):
        # imagesTr: lisa_XXXX_0000.nii.gz  (0000 = único canal CISO)
        dest_img = images_tr / f"lisa_{cid}_0000.nii.gz"
        # labelsTr: lisa_XXXX.nii.gz
        dest_seg = labels_tr / f"lisa_{cid}.nii.gz"

        shutil.copy2(str(ciso_path), str(dest_img))
        shutil.copy2(str(seg_path),  str(dest_seg))

        if (i + 1) % 10 == 0 or (i + 1) == len(cases):
            print(f"  {i+1}/{len(cases)} casos copiados")


def write_dataset_json(dataset_dir: Path, label_map: dict, n_training: int) -> None:
    """Genera el dataset.json requerido por nnU-Net v2."""
    dataset_json = {
        "channel_names": {
            "0": "CISO"
        },
        "labels": label_map,
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient"
    }
    out_path = dataset_dir / "dataset.json"
    with open(str(out_path), "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\n[4/4] dataset.json escrito en {out_path}")


def print_next_steps(dataset_id: str, dataset_dir: Path) -> None:
    print("\n" + "="*60)
    print("CONVERSIÓN COMPLETADA")
    print("="*60)
    print("\nPróximos pasos:")
    print("\n1. Configurar variables de entorno (PowerShell):")
    workspace = dataset_dir.parent.parent
    print(f'   $env:nnUNet_raw          = "{workspace / "nnUNet_raw"}"')
    print(f'   $env:nnUNet_preprocessed = "{workspace / "nnUNet_preprocessed"}"')
    print(f'   $env:nnUNet_results      = "{workspace / "nnUNet_results"}"')
    print(f"\n2. Instalar nnU-Net v2:")
    print(f"   pip install nnunetv2")
    print(f"\n3. Verificar dataset y planificar:")
    print(f"   nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")
    print(f"\n4. Entrenar fold 0 (baseline):")
    print(f"   nnUNetv2_train {dataset_id} 3d_fullres 0")
    print(f"\n5. Leer DSC tras fold 0:")
    results_path = workspace / "nnUNet_results" / f"Dataset{dataset_id}_LISA_Task2"
    print(f"   {results_path}")
    print(f"   -> nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/summary.json")
    print("\nADVERTENCIA: verificar LABEL_MAP en este script contra la documentación")
    print("oficial del challenge antes de interpretar métricas por estructura.")


def main():
    print("="*60)
    print(f"Preparando Dataset{DATASET_ID}_{DATASET_NAME} para nnU-Net v2")
    print("="*60)

    # 1. Recopilar casos G1
    cases = find_g1_cases(TRAIN_DIR)
    print(f"\nCasos G1 encontrados: {len(cases)}")
    if len(cases) == 0:
        print("ERROR: no se encontraron casos con ciso + seg. Verificar TRAIN_DIR.")
        sys.exit(1)

    # 2. Verificaciones
    expected_labels = set(LABEL_MAP.values()) - {0}
    if not verify_labels(cases, expected_labels):
        print("\nABORTANDO — corregir errores de labels antes de continuar.")
        sys.exit(1)

    if not verify_spacing(cases):
        print("\nABORTANDO — corregir errores de geometría antes de continuar.")
        sys.exit(1)

    # 3. Crear estructura y copiar
    dataset_dir, images_tr, labels_tr = build_dataset_dir(DATASET_ID, DATASET_NAME, NNUNET_RAW)
    copy_files(cases, images_tr, labels_tr)

    # 4. Generar dataset.json
    write_dataset_json(dataset_dir, LABEL_MAP, len(cases))

    # 5. Instrucciones siguientes
    print_next_steps(DATASET_ID, dataset_dir)


if __name__ == "__main__":
    main()
