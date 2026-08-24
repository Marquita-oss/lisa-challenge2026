"""Convierte el zip de submission de float32 a int16 (mismo affine/dims/nombres).

No re-corre el modelo: extrae el zip existente, redondea cada volumen a int16 y
re-empaqueta. Reduce a la mitad RAM/disco del scorer, manteniendo la integridad
espacial y dimensional exigida por el challenge.
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import nibabel as nib

SRC = "LISA_enhanced_predictions_float32_backup.zip"
DST = "LISA_enhanced_predictions.zip"

work = Path(tempfile.mkdtemp(prefix="lisa_int16_"))
in_dir = work / "in"
out_dir = work / "out"
in_dir.mkdir()
out_dir.mkdir()

with zipfile.ZipFile(SRC, "r") as zf:
    zf.extractall(in_dir)

names = sorted(p for p in in_dir.rglob("*.nii.gz"))
print(f"Archivos a convertir: {len(names)}")

global_min, global_max = np.inf, -np.inf
n_clipped = 0

for i, p in enumerate(names, 1):
    img = nib.load(str(p))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    global_min = min(global_min, float(data.min()))
    global_max = max(global_max, float(data.max()))

    rounded = np.rint(data)
    clipped = np.clip(rounded, 0, 65535)
    if not np.array_equal(rounded, clipped):
        n_clipped += 1
    out = clipped.astype(np.uint16)

    new_img = nib.Nifti1Image(out, img.affine, img.header)
    new_img.header.set_data_dtype(np.uint16)
    new_img.header.set_slope_inter(1, 0)   # sin reescalado: valores tal cual

    nib.save(new_img, str(out_dir / p.name))
    if i % 20 == 0 or i == len(names):
        print(f"  [{i}/{len(names)}] {p.name}")

# Re-empaquetar plano (solo nombres de archivo)
dst_path = Path(DST)
if dst_path.exists():
    dst_path.unlink()
with zipfile.ZipFile(dst_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(out_dir.glob("*.nii.gz")):
        zf.write(f, arcname=f.name)

size_mb = dst_path.stat().st_size / 1024 / 1024
print(f"\nRango global de valores: [{global_min:.2f}, {global_max:.2f}]")
print(f"Archivos que requirieron clip a int16: {n_clipped}")
print(f"ZIP creado: {dst_path}  ({size_mb:.1f} MB)")

shutil.rmtree(work, ignore_errors=True)
