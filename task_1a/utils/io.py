import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def load_lf_as_image(nii_path: Path) -> Image.Image:
    """Carga un NIfTI LF y retorna una imagen PIL RGB normalizada."""
    nii = nib.load(str(nii_path))
    vol = nii.get_fdata(dtype=np.float32)

    if vol.ndim == 2:
        img = vol
    elif vol.ndim == 3:
        # El plano 2D es el eje con menos slices
        thin_axis = int(np.argmin(vol.shape))
        mid = vol.shape[thin_axis] // 2
        idx = [slice(None)] * 3
        idx[thin_axis] = mid
        img = vol[tuple(idx)]
    else:
        raise ValueError(f"Dimensiones NIfTI inesperadas: {vol.ndim} en {nii_path}")

    p1, p99 = np.percentile(img, [1, 99])
    denom = p99 - p1
    img = np.clip(img, p1, p99)
    img = (img - p1) / denom if denom > 1e-8 else np.zeros_like(img)

    img_u8 = (img * 255).astype(np.uint8)
    img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
    return Image.fromarray(img_rgb)


_MULTICHANNEL_CACHE: dict = {}


def load_lf_multichannel(nii_path: Path) -> Image.Image:
    """Carga 3 slices equidistantes (P25/P50/P75) del eje grueso como canales RGB reales.

    Cada canal contiene información independiente del volumen, a diferencia de replicar
    un solo slice. Los artefactos de motion, banding y zipper se distribuyen a lo
    largo del volumen y son más detectables con múltiples posiciones.

    Cachea la imagen PIL preprocesada (pre-resize, ~52 KB/img) en memoria: las 532
    imágenes de train ocupan ~28 MB y se cargan una sola vez (no por época). Esto
    evita que la GPU quede hambrienta de I/O durante el entrenamiento k-fold.
    Las transforms de torchvision no mutan la imagen, así que es seguro compartirla.
    """
    key = str(nii_path)
    cached = _MULTICHANNEL_CACHE.get(key)
    if cached is not None:
        return cached

    nii = nib.load(str(nii_path))
    vol = nii.get_fdata(dtype=np.float32)

    if vol.ndim == 2:
        raw_slices = [vol, vol, vol]
    elif vol.ndim == 3:
        thin_axis = int(np.argmin(vol.shape))
        n = vol.shape[thin_axis]
        positions = [max(0, int(n * p)) for p in (0.25, 0.50, 0.75)]
        raw_slices = []
        for pos in positions:
            idx = [slice(None)] * 3
            idx[thin_axis] = pos
            raw_slices.append(vol[tuple(idx)])
    else:
        raise ValueError(f"Dimensiones NIfTI inesperadas: {vol.ndim} en {nii_path}")

    channels = []
    for s in raw_slices:
        p1, p99 = np.percentile(s, [1, 99])
        denom = p99 - p1
        s = np.clip(s, p1, p99)
        s = (s - p1) / denom if denom > 1e-8 else np.zeros_like(s)
        channels.append((s * 255).astype(np.uint8))

    img = Image.fromarray(np.stack(channels, axis=-1))
    _MULTICHANNEL_CACHE[key] = img
    return img


def find_image_path(filename: str, train_dir: Path) -> Path | None:
    """Resuelve un filename del CSV a su ruta en data/train/XXXX/.

    El CSV usa mayúsculas (LISA_0001_LF_axi.nii.gz); en disco está en minúsculas
    (lisa_0001_lf_axi.nii.gz). Se aplica .lower() al construir la ruta.
    """
    m = re.search(r'(?i)lisa_(\d+)_', filename)
    if not m:
        return None
    case_id = m.group(1)
    path = train_dir / case_id / filename.lower()
    return path if path.exists() else None
