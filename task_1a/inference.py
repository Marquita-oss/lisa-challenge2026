"""
inference.py — Helpers compartidos de inferencia (carga de modelo, TTA, predicción).

Usado por oof.py (OOF honesto), predict.py (submission ensemble) y train.py.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE, NUM_WORKERS
from model import OrdinalClassifier

# TTA x8: identidad + flips + rotaciones (dihedral). Geométricas, no alteran intensidad.
TTA_OPS = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[3]),
    lambda x: torch.flip(x, dims=[2]),
    lambda x: torch.flip(x, dims=[2, 3]),
    lambda x: torch.rot90(x, k=1, dims=[2, 3]),
    lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3]),
    lambda x: torch.rot90(x, k=3, dims=[2, 3]),
    lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[3]),
]


def val_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(backbone: str, ckpt_path, device, hidden_dim: int = 512):
    model = OrdinalClassifier(backbone=backbone, hidden_dim=hidden_dim, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_probs(model, dataset, device, tta: bool = False):
    """Devuelve (p1, p2) de shape (N, n_artifacts) en el orden del dataset.

    El dataset debe devolver la imagen como PRIMER elemento de cada item.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    ops = TTA_OPS if tta else [TTA_OPS[0]]
    all_p1, all_p2 = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        acc1, acc2 = [], []
        for op in ops:
            lo1, lo2 = model(op(imgs))
            acc1.append(torch.sigmoid(lo1).cpu().numpy())
            acc2.append(torch.sigmoid(lo2).cpu().numpy())
        all_p1.append(np.mean(acc1, axis=0))
        all_p2.append(np.mean(acc2, axis=0))
    return np.concatenate(all_p1), np.concatenate(all_p2)
