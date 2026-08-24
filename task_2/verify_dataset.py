"""
Verificación independiente del dataset LISA Task 2 en formato nnU-Net v2.

Ejecutar ANTES de nnUNetv2_plan_and_preprocess para confirmar que todo está correcto.

Uso:
    python verify_dataset.py
"""

import json
import sys
import numpy as np
import nibabel as nib
from pathlib import Path

# ---------------------------------------------------------------------------
TRAIN_DIR    = Path(r"C:/Users/rmarcar/Desktop/lisa-challenge2026/data/train")
NNUNET_RAW   = Path(r"C:/Users/rmarcar/Desktop/lisa-challenge2026/nnunet_workspace/nnUNet_raw")
DATASET_NAME = "Dataset001_LISA_Task2"
EXPECTED_LABELS = set(range(0, 12))   # 0 = fondo, 1–11 = estructuras
# ---------------------------------------------------------------------------

OK   = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors   = []
warnings = []


def section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# 1. DATOS EN data/train/
# ---------------------------------------------------------------------------
section("1. Casos G1 en data/train/")

g1_cases = []
for case_dir in sorted(TRAIN_DIR.iterdir()):
    if not case_dir.is_dir():
        continue
    cid  = case_dir.name
    ciso = case_dir / f"lisa_{cid}_ciso.nii.gz"
    seg  = case_dir / f"lisa_{cid}_seg.nii.gz"
    if ciso.exists() and seg.exists():
        g1_cases.append((cid, ciso, seg))

print(f"  Casos G1 (ciso + seg): {len(g1_cases)}")
if len(g1_cases) < 79:
    msg = f"Solo {len(g1_cases)} casos G1 — esperados 79"
    print(f"  {WARN} {msg}")
    warnings.append(msg)
else:
    print(f"  {OK} 79 casos G1 encontrados")


# ---------------------------------------------------------------------------
# 2. LABELS: consistencia entre todos los casos
# ---------------------------------------------------------------------------
section("2. Consistencia de labels en seg.nii.gz")

label_counts = {i: [] for i in range(1, 12)}
shape_set    = set()
spacing_set  = set()

for cid, ciso_path, seg_path in g1_cases:
    seg_img  = nib.load(str(seg_path))
    ciso_img = nib.load(str(ciso_path))
    seg_data = seg_img.get_fdata().astype(np.int32)

    found_labels = set(np.unique(seg_data).tolist())

    # Labels inesperados
    extra = found_labels - EXPECTED_LABELS
    if extra:
        msg = f"Caso {cid}: labels inesperados {sorted(extra)}"
        print(f"  {FAIL} {msg}")
        errors.append(msg)

    # Labels faltantes (excluir 0)
    missing = (EXPECTED_LABELS - {0}) - found_labels
    if missing:
        msg = f"Caso {cid}: labels faltantes {sorted(missing)}"
        print(f"  {WARN} {msg}")
        warnings.append(msg)

    # Acumular volúmenes por label
    for lbl in range(1, 12):
        label_counts[lbl].append(int((seg_data == lbl).sum()))

    # Shape
    if seg_img.shape != ciso_img.shape:
        msg = f"Caso {cid}: shape CISO={ciso_img.shape} ≠ SEG={seg_img.shape}"
        print(f"  {FAIL} {msg}")
        errors.append(msg)

    shape_set.add(ciso_img.shape)

    # Spacing
    zooms = tuple(round(float(z), 3) for z in ciso_img.header.get_zooms()[:3])
    if not all(abs(z - 1.0) < 0.02 for z in zooms):
        msg = f"Caso {cid}: spacing no es 1mm: {zooms}"
        print(f"  {WARN} {msg}")
        warnings.append(msg)
    spacing_set.add(zooms)

if not errors:
    print(f"  {OK} Labels 0–11 consistentes en todos los casos")
print(f"  {OK} Shapes encontrados: {sorted(shape_set)}")
print(f"  {OK} Spacings encontrados: {sorted(spacing_set)}")


# ---------------------------------------------------------------------------
# 3. ESTRUCTURA nnU-Net en nnunet_workspace/
# ---------------------------------------------------------------------------
section("3. Estructura nnU-Net v2")

dataset_dir = NNUNET_RAW / DATASET_NAME
images_tr   = dataset_dir / "imagesTr"
labels_tr   = dataset_dir / "labelsTr"
dataset_json = dataset_dir / "dataset.json"

for path, label in [
    (NNUNET_RAW,     "nnUNet_raw/"),
    (dataset_dir,    "Dataset001_LISA_Task2/"),
    (images_tr,      "imagesTr/"),
    (labels_tr,      "labelsTr/"),
    (dataset_json,   "dataset.json"),
]:
    if path.exists():
        print(f"  {OK} {label}")
    else:
        msg = f"No existe: {path}"
        print(f"  {FAIL} {msg}")
        errors.append(msg)

# Contar archivos
if images_tr.exists():
    imgs = list(images_tr.glob("*_0000.nii.gz"))
    segs = list(labels_tr.glob("*.nii.gz")) if labels_tr.exists() else []
    print(f"  {OK} imagesTr: {len(imgs)} archivos")
    print(f"  {OK} labelsTr: {len(segs)} archivos")
    if len(imgs) != len(segs):
        msg = f"imagesTr ({len(imgs)}) y labelsTr ({len(segs)}) tienen distinto número de archivos"
        print(f"  {FAIL} {msg}")
        errors.append(msg)
    if len(imgs) != len(g1_cases):
        msg = f"nnU-Net tiene {len(imgs)} casos pero G1 tiene {len(g1_cases)}"
        print(f"  {WARN} {msg}")
        warnings.append(msg)

# Validar dataset.json
if dataset_json.exists():
    with open(str(dataset_json)) as f:
        dj = json.load(f)
    required_keys = {"channel_names", "labels", "numTraining", "file_ending"}
    missing_keys  = required_keys - set(dj.keys())
    if missing_keys:
        msg = f"dataset.json faltan claves: {missing_keys}"
        print(f"  {FAIL} {msg}")
        errors.append(msg)
    else:
        n_labels = len(dj["labels"]) - 1  # excluir background
        print(f"  {OK} dataset.json: {n_labels} estructuras + background, {dj['numTraining']} casos")


# ---------------------------------------------------------------------------
# 4. RESUMEN DE VOLÚMENES POR LABEL
# ---------------------------------------------------------------------------
section("4. Volúmenes por label (media ± std, mm³ a 1mm iso)")

print(f"  {'Label':>6}  {'Media':>8}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
print(f"  {'-'*50}")
for lbl in range(1, 12):
    vals = label_counts[lbl]
    if vals:
        print(f"  {lbl:>6}  {np.mean(vals):>8.0f}  {np.std(vals):>7.0f}  "
              f"{np.min(vals):>7.0f}  {np.max(vals):>7.0f}")

print()
print("  Interpreta estos volúmenes junto a verify_labels.py para confirmar el mapeo.")


# ---------------------------------------------------------------------------
# 5. RESULTADO FINAL
# ---------------------------------------------------------------------------
section("5. Resultado")

if errors:
    print(f"  {FAIL} {len(errors)} ERROR(ES) — corregir antes de continuar:\n")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
elif warnings:
    print(f"  {WARN} {len(warnings)} ADVERTENCIA(S) — revisar:\n")
    for w in warnings:
        print(f"    - {w}")
    print(f"\n  Dataset usable con precaución.")
else:
    print(f"  {OK} Todo correcto — ejecutar:")
    print(f"\n     nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity\n")
