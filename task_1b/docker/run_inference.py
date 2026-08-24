"""run_inference.py — Entrypoint de inferencia Docker para LISA 2026 Task 1b (mejora de calidad).

Replica el pipeline "native_clean" (mejor submission conocida localmente: FID 131,
BRISQUEd -4.76 vs LF crudo — ver docs/IMPROVEMENTS_TASK_1B.md / memoria del proyecto):
por cada slice del volumen, clip a percentiles [p1, p99] (elimina outliers extremos
sin alterar la resolucion ni el contraste global) y empaquetado final a uint16.
No usa GPU ni red neuronal: los denoisers entrenados (v1/v2/v3/GAN/perceptual)
quedaron rechazados localmente por alucinar detalle (ver memoria del proyecto).

Lee todos los .nii.gz bajo $INPUT_DIR (recursivo), preserva shape/spacing/affine,
y escribe cada volumen mejorado directamente en $OUTPUT_DIR (SIN comprimir en zip,
por instruccion explicita de la fase de testing 2026) con el nombre exacto
confirmado por el equipo: nombre de entrada + '_enhanced' antes de la extension.
    LISA_TESTING_0001_LF_axi.nii.gz -> LISA_TESTING_0001_LF_axi_enhanced.nii.gz
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

INPUT_DIR = Path(os.environ.get('INPUT_DIR', '/input'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', '/output'))


def build_output_filename(input_filename: str) -> str:
    """Preserva el nombre de entrada tal cual, solo agrega '_enhanced' antes de la
    extension. No reordena ni reformatea el nombre (evita asumir un patron de ID/
    plano que puede no calzar con la convencion real del set de testing).
    """
    if input_filename.lower().endswith('.nii.gz'):
        return input_filename[:-len('.nii.gz')] + '_enhanced.nii.gz'
    stem, ext = os.path.splitext(input_filename)
    return f'{stem}_enhanced{ext}'


def collect_lf_files(directory: Path) -> list:
    return [f for f in sorted(directory.rglob('*.nii.gz')) if '_ciso' not in f.name.lower()]


def enhance_and_pack(nii_path: Path):
    """Clip por-slice a [p1, p99] (preserva shape/affine) + empaquetado uint16 final."""
    img = nib.load(str(nii_path))
    vol = img.get_fdata(dtype=np.float32)

    if vol.ndim == 2:
        vol = vol[:, :, None]
        thin = 2
        squeeze_back = True
    else:
        thin = int(np.argmin(vol.shape))
        squeeze_back = False

    out_slices = []
    for i in range(vol.shape[thin]):
        idx = [slice(None)] * 3
        idx[thin] = i
        s_raw = vol[tuple(idx)].copy()
        p1 = float(np.percentile(s_raw, 1))
        p99 = float(np.percentile(s_raw, 99))
        d = p99 - p1
        if d < 1e-8:
            out_slices.append(s_raw.astype(np.float32))
            continue
        sn = np.clip((s_raw - p1) / d, 0, 1).astype(np.float32)
        out_slices.append((sn * d + p1).astype(np.float32))

    enh = np.stack(out_slices, axis=thin)
    if squeeze_back:
        enh = enh[:, :, 0]

    rounded = np.rint(enh)
    clipped = np.clip(rounded, 0, 65535).astype(np.uint16)

    out_img = nib.Nifti1Image(clipped, img.affine, img.header)
    out_img.header.set_data_dtype(np.uint16)
    out_img.header.set_slope_inter(1, 0)
    return out_img


def main():
    t0 = time.time()
    print("===== LISA 2026 Task 1b — Quality Improvement Inference =====")
    print(f"[INFO] DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] INPUT_DIR={INPUT_DIR}")
    print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] Python: {sys.version.split()[0]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Contenido de {INPUT_DIR}:")
    try:
        for p in sorted(INPUT_DIR.iterdir())[:40]:
            print(f"    {p.name}")
    except Exception as e:
        print(f"  [WARN] no se pudo listar INPUT_DIR: {e}")

    files = collect_lf_files(INPUT_DIR)
    print(f"[INFO] Archivos .nii.gz encontrados (recursivo): {len(files)}")
    if not files:
        print(f"[ERROR] No se encontraron archivos .nii.gz en {INPUT_DIR}. Abortando.")
        sys.exit(1)

    processed, errors = 0, 0
    for f in files:
        out_name = build_output_filename(f.name)
        out_path = OUTPUT_DIR / out_name
        try:
            out_img = enhance_and_pack(f)
            nib.save(out_img, str(out_path))
            processed += 1
            if processed % 10 == 0 or processed == len(files):
                print(f"  [{processed}/{len(files)}] {f.name} -> {out_name}")
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
            errors += 1

    print(f"\n[INFO] Procesados: {processed}  Errores: {errors}")
    print(f"[INFO] Contenido final de {OUTPUT_DIR}:")
    for p in sorted(OUTPUT_DIR.iterdir())[:10]:
        print(f"    {p.name}")
    print(f"[INFO] Tiempo total: {time.time()-t0:.1f}s")

    if errors:
        print(f"[WARN] {errors} caso(s) fallaron y no tienen archivo de salida "
              f"(no se aborta la corrida por esto: se prefiere entregar los "
              f"{processed} casos validos antes que perderlos todos).")
    if processed == 0:
        print("[ERROR] Ningun caso se pudo procesar. Abortando.")
        sys.exit(1)
    print("===== DONE =====")


if __name__ == '__main__':
    main()
