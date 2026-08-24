"""
data.py — Splits k-fold, Dataset 3D y sampler para Task 1A.

Provee:
  - legacy_kfold_splits : reproduce EXACTAMENTE el split de los checkpoints v9 existentes
                          (KFold shuffle por case_id), para calcular OOF en Fase 0.
  - stratified_kfold_splits : StratifiedGroupKFold por case_id para el reentrenamiento
                          limpio (cobertura de clases raras en cada fold).
  - OrdinalDataset    : carga multicanal P25/P50/P75 + etiquetas ge1/ge2.
  - get_transforms    : augmentación SOLO geométrica (sin ColorJitter/GaussianBlur).
  - SeverityStratifiedSampler : garantiza ejemplos score>=1 y score>=2 por batch.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    ARTIFACT_COLS, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED, N_FOLDS,
)
from utils.io import load_lf_multichannel, find_image_path


# ---------------------------------------------------------------------------
# DataFrame + filtrado de archivos en disco
# ---------------------------------------------------------------------------

def load_dataframe(csv_path: Path, data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['case_id'] = df['filename'].str.extract(r'(?i)lisa_(\d+)_')[0]
    mask = df['filename'].apply(lambda f: find_image_path(f, data_dir) is not None)
    missing = int((~mask).sum())
    if missing:
        print(f"  [WARNING] {missing} archivos del CSV no encontrados en disco — omitidos")
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def legacy_kfold_splits(csv_path: Path, data_dir: Path, n_splits: int = N_FOLDS):
    """Reproduce el split de 02_train_v9_kfold.py (para OOF con checkpoints existentes).

    IMPORTANTE: idéntico al original — KFold(shuffle=True, random_state=RANDOM_SEED)
    sobre sorted(case_ids). NO cambiar: define a qué fold pertenece cada imagen.
    Devuelve lista de (df_train, df_val) y también los índices val (sobre el df filtrado).
    """
    df = load_dataframe(csv_path, data_dir)
    cases = np.array(sorted(df['case_id'].unique()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    splits = []
    for train_idx, val_idx in kf.split(cases):
        val_cases = set(cases[val_idx])
        df_val   = df[df['case_id'].isin(val_cases)].reset_index()   # conserva índice global en 'index'
        df_train = df[~df['case_id'].isin(val_cases)].reset_index(drop=True)
        splits.append((df_train, df_val))
    return df, splits


def stratified_kfold_splits(csv_path: Path, data_dir: Path, n_splits: int = N_FOLDS):
    """StratifiedGroupKFold por case_id, estratificando por la clase positiva más rara.

    Garantiza que artefactos raros (Banding 4%, Positioning 13%) aparezcan en todos
    los folds. Grupo = case_id (sin leakage entre planos del mismo paciente).
    """
    df = load_dataframe(csv_path, data_dir)
    # Etiqueta de estratificación: índice de la clase positiva más rara presente,
    # o len(ARTIFACT_COLS) si la imagen no tiene artefactos.
    rarity_order = ['Banding', 'Positioning', 'Noise', 'Distortion', 'Motion', 'Zipper', 'Contrast']
    rank = {c: i for i, c in enumerate(rarity_order)}
    def strat_label(row):
        present = [c for c in ARTIFACT_COLS if row[c] >= 1]
        if not present:
            return len(ARTIFACT_COLS)
        return rank[min(present, key=lambda c: rank[c])]
    y = df.apply(strat_label, axis=1).values
    groups = df['case_id'].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    splits = []
    for train_idx, val_idx in sgkf.split(df, y, groups):
        df_val   = df.iloc[val_idx].reset_index()
        df_train = df.iloc[train_idx].reset_index(drop=True)
        splits.append((df_train, df_val))
    return df, splits


# ---------------------------------------------------------------------------
# Transforms — SOLO geométricas (la intensidad MRI es señal diagnóstica)
# ---------------------------------------------------------------------------

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OrdinalDataset(Dataset):
    """Devuelve (img, g1, g2) donde g1=(score>=1), g2=(score>=2) por clase."""
    def __init__(self, df: pd.DataFrame, data_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = find_image_path(row['filename'], self.data_dir)
        if path is None:
            raise FileNotFoundError(f"No encontrado: {row['filename']}")
        img = load_lf_multichannel(path)
        if self.transform:
            img = self.transform(img)
        scores = row[ARTIFACT_COLS].values.astype(float)
        g1 = torch.tensor((scores >= 1).astype(np.float32))
        g2 = torch.tensor((scores >= 2).astype(np.float32))
        return img, g1, g2


# ---------------------------------------------------------------------------
# Sampler (movido de train_v9_module.py, sin cambios de comportamiento)
# ---------------------------------------------------------------------------

class SeverityStratifiedSampler:
    """Garantiza ejemplos score>=1 y score>=2 (uno por artefacto, cycling) en cada batch."""
    def __init__(self, df, artifact_cols, batch_size, seed=42):
        n_anchors   = 2 * len(artifact_cols)
        self.n_fill = max(batch_size - n_anchors, 4)
        self.rng    = np.random.default_rng(seed)
        self.pos_ge1, self.ptr_ge1 = [], []
        self.pos_ge2, self.ptr_ge2 = [], []
        for col in artifact_cols:
            ge1 = np.where(df[col].values >= 1)[0].tolist()
            ge2 = np.where(df[col].values >= 2)[0].tolist()
            self.pos_ge1.append(ge1)
            self.pos_ge2.append(ge2 if ge2 else ge1)
            self.ptr_ge1.append(0)
            self.ptr_ge2.append(0)
        self.all_idx   = list(range(len(df)))
        self.n_batches = max(len(df) // self.n_fill, 1)

    def __iter__(self):
        pool = self.all_idx.copy()
        self.rng.shuffle(pool)
        for i in range(self.n_batches):
            anchors = []
            for c in range(len(self.pos_ge1)):
                if self.pos_ge1[c]:
                    anchors.append(self.pos_ge1[c][self.ptr_ge1[c]])
                    self.ptr_ge1[c] = (self.ptr_ge1[c] + 1) % len(self.pos_ge1[c])
                if self.pos_ge2[c]:
                    anchors.append(self.pos_ge2[c][self.ptr_ge2[c]])
                    self.ptr_ge2[c] = (self.ptr_ge2[c] + 1) % len(self.pos_ge2[c])
            start = (i * self.n_fill) % len(pool)
            end   = start + self.n_fill
            fills = pool[start:end] if end <= len(pool) else list(pool[start:]) + list(pool[:end - len(pool)])
            batch = list(set(fills + anchors))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches
