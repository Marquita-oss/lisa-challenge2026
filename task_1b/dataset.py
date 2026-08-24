"""Dataset classes for Task 1b (denoising / quality improvement).

Training strategy: supervised learning on synthetic degradation.
  - Clean images come from the nonoise_nomotion partition (301 images).
  - Each clean image is degraded on-the-fly with Rician noise ± motion ghosting.
  - The model learns to map degraded → clean.

At inference time the model is applied to the actual noisy / motion images
from the other three partitions and to the blind validation set.
"""
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset

from task_1b.config import TRAIN_DIR, RANDOM_SEED, IMG_SIZE
from task_1b.utils.io import load_normalized_slices, find_image_path
from task_1b.augmentations import composite_degrade


# ── Synthetic degradation helpers ─────────────────────────────────────────────

def add_rician_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Rician noise model for MRI: magnitude of two noisy quadrature signals."""
    n1 = np.random.normal(0, sigma, img.shape).astype(np.float32)
    n2 = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.sqrt((img + n1) ** 2 + n2 ** 2)
    return np.clip(noisy, 0.0, 1.0)


def add_motion_ghosting(img: np.ndarray, n_ghosts: int, alpha: float) -> np.ndarray:
    """Simulate motion ghosting via k-space phase corruption.

    Ghost echoes are added along the phase-encode direction (axis=0 by convention).
    """
    H, W = img.shape
    kspace = np.fft.fft2(img)
    kspace_shift = np.fft.fftshift(kspace)

    ghost = np.zeros_like(kspace_shift)
    for g in range(1, n_ghosts + 1):
        shift = int(H / (n_ghosts + 1) * g)
        rolled = np.roll(kspace_shift, shift, axis=0)
        phase  = np.random.uniform(0, 2 * np.pi)
        ghost += rolled * np.exp(1j * phase)

    corrupted = kspace_shift + alpha * ghost / max(n_ghosts, 1)
    corrupted = np.fft.ifftshift(corrupted)
    result = np.abs(np.fft.ifft2(corrupted)).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def degrade(
    img: np.ndarray,
    noise_sigma: float,
    add_motion: bool,
    n_ghosts: int,
    motion_alpha: float,
) -> np.ndarray:
    deg = add_rician_noise(img, noise_sigma)
    if add_motion:
        deg = add_motion_ghosting(deg, n_ghosts, motion_alpha)
    return deg


# ── Dataset: synthetic pairs from clean images ─────────────────────────────────

class SyntheticDenoiseDataset(Dataset):
    """Each sample: (degraded_slice, clean_slice) with on-the-fly augmentation.

    Loads from the nonoise_nomotion CSV partition. Every NIfTI volume
    contributes all its 2-D slices as independent training samples.
    """

    def __init__(
        self,
        csv_path: Path,
        train_dir: Path = TRAIN_DIR,
        augment: bool = True,
        noise_sigma_range: Tuple[float, float] = (0.03, 0.18),
        motion_prob: float = 0.50,
        motion_ghosts_range: Tuple[int, int] = (2, 6),
        motion_alpha_range: Tuple[float, float] = (0.04, 0.25),
        patch_size: Optional[int] = None,
        seed: int = RANDOM_SEED,
    ):
        self.augment = augment
        self.noise_sigma_range = noise_sigma_range
        self.motion_prob       = motion_prob
        self.motion_ghosts_range = motion_ghosts_range
        self.motion_alpha_range  = motion_alpha_range
        self.patch_size = patch_size

        df = pd.read_csv(csv_path)
        self.samples: List[Tuple[Path, int]] = []  # (nii_path, slice_idx)

        for fn in df['filename']:
            path = find_image_path(fn, train_dir)
            if path is None:
                continue
            try:
                norm_slices, _, _ = load_normalized_slices(path)
                for i in range(len(norm_slices)):
                    self.samples.append((path, i))
            except Exception:
                continue

        random.seed(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, slice_idx = self.samples[idx]
        norm_slices, _, _ = load_normalized_slices(path)
        clean = norm_slices[slice_idx]  # (H, W) float32 in [0, 1]

        # Augmentation on clean before degradation (spatial only)
        if self.augment:
            clean = self._spatial_augment(clean)

        # Sample degradation params
        sigma      = random.uniform(*self.noise_sigma_range)
        do_motion  = random.random() < self.motion_prob
        n_ghosts   = random.randint(*self.motion_ghosts_range)
        alpha      = random.uniform(*self.motion_alpha_range)

        degraded = degrade(clean, sigma, do_motion, n_ghosts, alpha)

        # Resize to fixed IMG_SIZE so all batch elements have the same shape
        clean_t    = self._to_tensor(self._resize(clean))
        degraded_t = self._to_tensor(self._resize(degraded))
        return degraded_t, clean_t

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _spatial_augment(img: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.5:
            img = np.flipud(img)
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k)
        return np.ascontiguousarray(img)

    @staticmethod
    def _resize(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
        """Resize to (size × size) using bilinear interpolation."""
        pil = PILImage.fromarray((img * 255).astype(np.uint8)).resize(
            (size, size), PILImage.BILINEAR)
        return np.array(pil, dtype=np.float32) / 255.0

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


# ── Dataset: real noisy images for evaluation / inference ──────────────────────

class InferenceDataset(Dataset):
    """Load (degraded) LF images from a partition CSV or a directory.

    Returns (slice_tensor, path_str, slice_idx, original_H, original_W).
    """

    def __init__(self, csv_path: Optional[Path] = None,
                 image_dir: Optional[Path] = None,
                 train_dir: Path = TRAIN_DIR):
        self.records: List[Tuple[Path, int]] = []
        self.meta: List[Tuple[str, int, int, int]] = []  # (path_str, slice_idx, H, W)

        paths: List[Path] = []
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            for fn in df['filename']:
                p = find_image_path(fn, train_dir)
                if p:
                    paths.append(p)
        elif image_dir is not None:
            paths = sorted(image_dir.rglob('*.nii.gz'))

        for path in paths:
            try:
                norm_slices, _, _ = load_normalized_slices(path)
                for i, s in enumerate(norm_slices):
                    H, W = s.shape
                    self.records.append((path, i))
                    self.meta.append((str(path), i, H, W))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, slice_idx = self.records[idx]
        norm_slices, _, _ = load_normalized_slices(path)
        s = norm_slices[slice_idx]
        H, W = s.shape
        resized = SyntheticDenoiseDataset._resize(s)
        tensor = SyntheticDenoiseDataset._to_tensor(resized)
        path_str, _, _, _ = self.meta[idx]
        return tensor, path_str, slice_idx, H, W


# ── Dataset v2: physics-based composite degradation ──────────────────────────

class SyntheticDenoiseDatasetV2(Dataset):
    """v2 variant: uses composite_degrade() (physics-based artifacts) instead
    of the simpler Rician + ghosting pipeline of v1.

    Same split-by-case_id logic; augment flag still controls spatial flips/rots.
    """

    def __init__(
        self,
        csv_path: Path,
        train_dir: Path = TRAIN_DIR,
        augment: bool = True,
        seed: int = RANDOM_SEED,
    ):
        self.augment = augment

        df = pd.read_csv(csv_path)
        self.samples: List[Tuple[Path, int]] = []

        for fn in df['filename']:
            path = find_image_path(fn, train_dir)
            if path is None:
                continue
            try:
                norm_slices, _, _ = load_normalized_slices(path)
                for i in range(len(norm_slices)):
                    self.samples.append((path, i))
            except Exception:
                continue

        random.seed(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, slice_idx = self.samples[idx]
        norm_slices, _, _ = load_normalized_slices(path)
        clean = norm_slices[slice_idx]

        if self.augment:
            clean = SyntheticDenoiseDataset._spatial_augment(clean)

        degraded = composite_degrade(clean)

        clean_t    = SyntheticDenoiseDataset._to_tensor(SyntheticDenoiseDataset._resize(clean))
        degraded_t = SyntheticDenoiseDataset._to_tensor(SyntheticDenoiseDataset._resize(degraded))
        return degraded_t, clean_t


# ── Train / validation split ───────────────────────────────────────────────────

def build_split(
    csv_path: Path,
    val_split: float = 0.20,
    seed: int = RANDOM_SEED,
    train_dir: Path = TRAIN_DIR,
) -> Tuple["SyntheticDenoiseDataset", "SyntheticDenoiseDataset"]:
    """Return (train_dataset, val_dataset) split by case_id to avoid leakage."""
    df = pd.read_csv(csv_path)

    def extract_case(fn: str) -> str:
        m = re.search(r'(?i)lisa_(\d+)_', fn)
        return m.group(1) if m else fn

    df['case_id'] = df['filename'].apply(extract_case)
    cases = df['case_id'].unique().tolist()
    random.seed(seed)
    random.shuffle(cases)
    n_val = max(1, int(len(cases) * val_split))
    val_cases  = set(cases[:n_val])
    train_cases = set(cases[n_val:])

    import tempfile, os
    def write_subset(case_set):
        sub = df[df['case_id'].isin(case_set)].copy()
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        sub.to_csv(tmp.name, index=False)
        tmp.close()
        return Path(tmp.name)

    train_csv = write_subset(train_cases)
    val_csv   = write_subset(val_cases)

    train_ds = SyntheticDenoiseDataset(train_csv, train_dir=train_dir, augment=True)
    val_ds   = SyntheticDenoiseDataset(val_csv,   train_dir=train_dir, augment=False)

    os.unlink(train_csv)
    os.unlink(val_csv)
    return train_ds, val_ds


def build_split_v2(
    csv_path: Path,
    val_split: float = 0.20,
    seed: int = RANDOM_SEED,
    train_dir: Path = TRAIN_DIR,
) -> Tuple["SyntheticDenoiseDatasetV2", "SyntheticDenoiseDatasetV2"]:
    """Same case-level split as build_split but returns V2 datasets."""
    df = pd.read_csv(csv_path)

    def extract_case(fn: str) -> str:
        m = re.search(r'(?i)lisa_(\d+)_', fn)
        return m.group(1) if m else fn

    df['case_id'] = df['filename'].apply(extract_case)
    cases = df['case_id'].unique().tolist()
    random.seed(seed)
    random.shuffle(cases)
    n_val = max(1, int(len(cases) * val_split))
    val_cases   = set(cases[:n_val])
    train_cases = set(cases[n_val:])

    import tempfile, os

    def write_subset(case_set):
        sub = df[df['case_id'].isin(case_set)].copy()
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        sub.to_csv(tmp.name, index=False)
        tmp.close()
        return Path(tmp.name)

    train_csv = write_subset(train_cases)
    val_csv   = write_subset(val_cases)

    train_ds = SyntheticDenoiseDatasetV2(train_csv, train_dir=train_dir, augment=True)
    val_ds   = SyntheticDenoiseDatasetV2(val_csv,   train_dir=train_dir, augment=False)

    os.unlink(train_csv)
    os.unlink(val_csv)
    return train_ds, val_ds


# ── Dataset rapido desde cache .npy ───────────────────────────────────────────

class NpySliceDataset(Dataset):
    """Dataset rapido que lee slices pre-procesadas desde archivos .npy.

    Cada archivo .npy contiene una sola slice (IMG_SIZE, IMG_SIZE) float32
    ya normalizada a [0, 1] y redimensionada. Esto elimina el cuello de
    botella de I/O: no hay decompress gzip, no hay nibabel, no hay PIL resize
    en el hot-path del DataLoader.

    Generado por: python task_1b/preprocess_to_npy.py
    """

    def __init__(
        self,
        npy_dir: Path,
        augment: bool = True,
        use_physics_degrade: bool = True,
        noise_sigma_range=(0.03, 0.18),
        motion_prob: float = 0.50,
        motion_ghosts_range=(2, 6),
        motion_alpha_range=(0.04, 0.25),
        seed: int = RANDOM_SEED,
    ):
        self.augment = augment
        self.use_physics = use_physics_degrade
        self.noise_sigma_range   = noise_sigma_range
        self.motion_prob         = motion_prob
        self.motion_ghosts_range = motion_ghosts_range
        self.motion_alpha_range  = motion_alpha_range

        npy_dir = Path(npy_dir)
        if not npy_dir.exists():
            raise FileNotFoundError(
                f"Cache .npy no encontrado: {npy_dir}\n"
                f"Ejecutar primero: python task_1b/preprocess_to_npy.py"
            )
        self.files = sorted(npy_dir.glob('*.npy'))
        if not self.files:
            raise RuntimeError(f"No se encontraron archivos .npy en {npy_dir}")

        random.seed(seed)

    @classmethod
    def from_file_list(
        cls,
        file_list: list,
        augment: bool = True,
        use_physics_degrade: bool = True,
        noise_sigma_range=(0.03, 0.18),
        motion_prob: float = 0.50,
        motion_ghosts_range=(2, 6),
        motion_alpha_range=(0.04, 0.25),
        seed: int = RANDOM_SEED,
    ) -> 'NpySliceDataset':
        """Crea un NpySliceDataset desde una lista explicita de paths .npy.

        Evita la necesidad de crear directorios temporales o symlinks,
        lo que resuelve el WinError 1314 en Windows sin permisos de admin.
        """
        obj = cls.__new__(cls)
        obj.augment             = augment
        obj.use_physics         = use_physics_degrade
        obj.noise_sigma_range   = noise_sigma_range
        obj.motion_prob         = motion_prob
        obj.motion_ghosts_range = motion_ghosts_range
        obj.motion_alpha_range  = motion_alpha_range
        obj.files               = sorted(file_list)
        if not obj.files:
            raise RuntimeError("La lista de archivos .npy esta vacia.")
        random.seed(seed)
        return obj

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        # Lectura directa del array pre-procesado — sin decompresion ni redimensionado
        clean = np.load(self.files[idx])   # (H, W) float32 en [0, 1]

        # Augmentacion espacial sobre la imagen limpia
        if self.augment:
            clean = SyntheticDenoiseDataset._spatial_augment(clean)

        # Degradacion
        if self.use_physics:
            degraded = composite_degrade(clean)
        else:
            sigma     = random.uniform(*self.noise_sigma_range)
            do_motion = random.random() < self.motion_prob
            n_ghosts  = random.randint(*self.motion_ghosts_range)
            alpha     = random.uniform(*self.motion_alpha_range)
            degraded  = degrade(clean, sigma, do_motion, n_ghosts, alpha)

        return (
            SyntheticDenoiseDataset._to_tensor(degraded),
            SyntheticDenoiseDataset._to_tensor(clean),
        )


def build_split_npy(
    npy_train_dir: Path,
    val_split: float = 0.20,
    seed: int = RANDOM_SEED,
    use_physics_degrade: bool = True,
) -> tuple:
    """Divide el cache .npy en train/val por case_id (sin fuga de datos).

    Los archivos .npy tienen nombres como:
      lisa_0042_lf_sag_s005.npy  ->  case_id = '0042'
    El split se hace a nivel de case_id, no de slice, para evitar leakage.

    Devuelve (train_dataset, val_dataset) de tipo NpySliceDataset.
    Usa listas de paths — compatible con Windows sin permisos de admin.
    """
    npy_train_dir = Path(npy_train_dir)
    all_files = sorted(npy_train_dir.glob('*.npy'))
    if not all_files:
        raise RuntimeError(f"No hay archivos .npy en {npy_train_dir}")

    def get_case(f: Path) -> str:
        m = re.search(r'(?i)lisa_?(?:lf_)?(\d+)', f.stem)
        return m.group(1) if m else f.stem

    cases = sorted(set(get_case(f) for f in all_files))
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_val = max(1, int(len(cases) * val_split))
    val_cases   = set(cases[:n_val])
    train_cases = set(cases[n_val:])

    train_files = [f for f in all_files if get_case(f) in train_cases]
    val_files   = [f for f in all_files if get_case(f) in val_cases]

    print(f"  Split NPY: {len(train_cases)} casos train ({len(train_files)} slices) | "
          f"{len(val_cases)} casos val ({len(val_files)} slices)")

    train_ds = NpySliceDataset.from_file_list(
        train_files, augment=True,
        use_physics_degrade=use_physics_degrade, seed=seed,
    )
    val_ds = NpySliceDataset.from_file_list(
        val_files, augment=False,
        use_physics_degrade=False, seed=seed,
    )

    return train_ds, val_ds
