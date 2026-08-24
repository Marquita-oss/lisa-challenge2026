"""NIfTI I/O utilities for Task 1b (slice-level denoising)."""
import re
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np


# ── Loading ───────────────────────────────────────────────────────────────────

def load_volume(nii_path: Path) -> Tuple[np.ndarray, object, int]:
    """Load a NIfTI LF image and return (volume, img_obj, thin_axis).

    Returns the raw float32 array (H, W, D) with thin_axis identifying
    the slice direction (smallest dimension of the volume).
    """
    img = nib.load(str(nii_path))
    vol = img.get_fdata(dtype=np.float32)

    if vol.ndim == 2:
        vol = vol[:, :, np.newaxis]  # treat as single slice
        thin_axis = 2
    elif vol.ndim == 3:
        thin_axis = int(np.argmin(vol.shape))
    else:
        raise ValueError(f"Unexpected NIfTI dims {vol.ndim}: {nii_path}")

    return vol, img, thin_axis


def extract_slices(vol: np.ndarray, thin_axis: int) -> List[np.ndarray]:
    """Extract all 2-D slices along thin_axis as a list of (H, W) arrays."""
    n = vol.shape[thin_axis]
    slices = []
    for i in range(n):
        idx = [slice(None)] * 3
        idx[thin_axis] = i
        slices.append(vol[tuple(idx)])
    return slices


def normalize_slice(s: np.ndarray) -> np.ndarray:
    """Normalize a 2-D float slice to [0, 1] using 1st/99th percentiles."""
    p1, p99 = np.percentile(s, [1, 99])
    denom = p99 - p1
    s = np.clip(s, p1, p99)
    return ((s - p1) / denom if denom > 1e-8 else np.zeros_like(s)).astype(np.float32)


def denormalize_slice(s_norm: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """Undo normalize_slice given the original percentiles."""
    denom = p99 - p1
    return (s_norm * denom + p1).astype(np.float32)


def load_normalized_slices(nii_path: Path) -> Tuple[List[np.ndarray], List[Tuple], int]:
    """Load a LF NIfTI and return (norm_slices, percentiles_list, thin_axis).

    norm_slices:      list of (H, W) float32 in [0, 1]
    percentiles_list: list of (p1, p99) tuples — needed to re-scale on save
    """
    vol, _, thin_axis = load_volume(nii_path)
    raw_slices = extract_slices(vol, thin_axis)
    norm_slices, pcts = [], []
    for s in raw_slices:
        p1, p99 = float(np.percentile(s, 1)), float(np.percentile(s, 99))
        pcts.append((p1, p99))
        norm_slices.append(normalize_slice(s))
    return norm_slices, pcts, thin_axis


# ── Saving ────────────────────────────────────────────────────────────────────

def assemble_and_save(
    enhanced_slices: List[np.ndarray],
    reference_img: object,
    thin_axis: int,
    out_path: Path,
) -> None:
    """Reassemble enhanced slices into a NIfTI volume and save.

    enhanced_slices: list of (H, W) float32 — already in original intensity scale.
    reference_img:   nibabel image object (provides affine and header).
    """
    # Stack along thin_axis
    stack = np.stack(enhanced_slices, axis=thin_axis).astype(np.float32)

    out_img = nib.Nifti1Image(stack, reference_img.affine, reference_img.header)
    out_img.header.set_data_dtype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))


# ── Path helpers (mirrors task_1a) ────────────────────────────────────────────

def find_image_path(filename: str, train_dir: Path) -> Path | None:
    """Resolve a CSV filename to its path under data/train/XXXX/."""
    m = re.search(r'(?i)lisa_(\d+)_', filename)
    if not m:
        return None
    case_id = m.group(1)
    path = train_dir / case_id / filename.lower()
    return path if path.exists() else None


def find_val_path(filename: str, val_dir: Path) -> Path | None:
    """Resolve a validation filename (lisa_validation_XXXX_...) to its path."""
    m = re.search(r'(?i)lisa_validation_(\d+)_', filename)
    if not m:
        return None
    case_id = m.group(1)
    path = val_dir / case_id / filename.lower()
    return path if path.exists() else None
