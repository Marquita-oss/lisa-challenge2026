"""Task 1b — Entrenamiento supervisado REAL LF -> CISO a resolucion nativa.

Diferencias clave con v2/v3 (que entrenan denoising sintetico sobre imagenes limpias):
  - Target = CISO real (alto campo) remuestreado por RESIZE a la grilla del plano
    LF (alineamiento por array, consistente ~0.6 corr en todos los planos).
  - Entrada/target a resolucion NATIVA con padding a 160x160 (sin resize a 256,
    sin cuantizacion uint8). Ataca el domain gap y el blur de raiz.
  - Validacion honesta: split por case_id de data/train. Los 14 casos
    val/complete quedan para la evaluacion final (eval_local.py).
  - Loss L1+SSIM (CombinedLoss). Init opcional desde best_1b_v2.pth.

Salida: checkpoints/best_1b_lf2ciso.pth, results/02_training_lf2ciso.json

Uso:
  python task_1b/02_train_lf2ciso.py --init-from-v2 --epochs 60
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize as sk_resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_1b.config import (
    TRAIN_DIR, CHECKPOINTS_DIR, RESULTS_DIR, RANDOM_SEED, LR, WEIGHT_DECAY,
)
from task_1b.model import build_model, count_params
from task_1b.losses import CombinedLoss
from task_1b.utils.metrics import psnr_torch

CANVAS = 160  # multiplo de 16 que cubre la max dim en-plano (147)


def norm01(a: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(a, [1, 99])
    d = p99 - p1
    if d < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - p1) / d, 0, 1).astype(np.float32)


def pad_to(a: np.ndarray, size: int = CANVAS) -> np.ndarray:
    H, W = a.shape
    out = np.zeros((size, size), np.float32)
    out[:H, :W] = a
    return out


def build_pairs(case_dirs, min_fg=0.03):
    """Lista de (lf_slice_f16, ciso_slice_f16, H, W) a resolucion nativa."""
    pairs = []
    for cdir in case_dirs:
        cid = cdir.name
        ciso_p = cdir / f'lisa_{cid}_ciso.nii.gz'
        if not ciso_p.exists():
            continue
        ciso = norm01(nib.load(str(ciso_p)).get_fdata(dtype=np.float32))
        for plane in ['axi', 'cor', 'sag']:
            lf_p = cdir / f'lisa_{cid}_lf_{plane}.nii.gz'
            if not lf_p.exists():
                continue
            lf_vol = nib.load(str(lf_p)).get_fdata(dtype=np.float32)
            if lf_vol.ndim != 3:
                continue
            thin = int(np.argmin(lf_vol.shape))
            ciso_rs = sk_resize(ciso, lf_vol.shape, order=1, preserve_range=True,
                                anti_aliasing=True).astype(np.float32)
            n = lf_vol.shape[thin]
            for i in range(n):
                idx = [slice(None)] * 3
                idx[thin] = i
                lf_s = lf_vol[tuple(idx)]
                ci_s = ciso_rs[tuple(idx)]
                if lf_s.max() <= 1e-6:
                    continue
                lf_n = norm01(lf_s)
                if float((lf_n > 0.1).mean()) < min_fg:
                    continue
                ci_n = norm01(ci_s)
                H, W = lf_n.shape
                pairs.append((lf_n.astype(np.float16), ci_n.astype(np.float16), H, W))
    return pairs


class LF2CISODataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lf, ci, H, W = self.pairs[idx]
        lf = lf.astype(np.float32)
        ci = ci.astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                lf, ci = np.fliplr(lf), np.fliplr(ci)
            if np.random.rand() < 0.5:
                lf, ci = np.flipud(lf), np.flipud(ci)
        lf = pad_to(np.ascontiguousarray(lf))
        ci = pad_to(np.ascontiguousarray(ci))
        return (torch.from_numpy(lf)[None], torch.from_numpy(ci)[None])


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    tot_loss, tot_psnr, n = 0.0, 0.0, 0
    torch.set_grad_enabled(train)
    for lf, ci in loader:
        lf, ci = lf.to(device), ci.to(device)
        if train:
            optimizer.zero_grad()
        pred = model(lf)
        loss = criterion(pred, ci)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        tot_loss += float(loss.item())
        tot_psnr += psnr_torch(pred, ci)
        n += 1
    torch.set_grad_enabled(True)
    return tot_loss / max(n, 1), tot_psnr / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-from-v2', action='store_true')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--val-split', type=float, default=0.15)
    args = ap.parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  CANVAS={CANVAS}")

    cases = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(len(cases))
    n_val = max(1, int(len(cases) * args.val_split))
    val_cases = [cases[i] for i in perm[:n_val]]
    train_cases = [cases[i] for i in perm[n_val:]]
    print(f"Casos: {len(cases)}  train={len(train_cases)}  val={len(val_cases)}")

    print("Construyendo pares LF->CISO (puede tardar)...")
    t0 = time.time()
    train_pairs = build_pairs(train_cases)
    val_pairs = build_pairs(val_cases)
    print(f"  train slices: {len(train_pairs)}  val slices: {len(val_pairs)}  "
          f"({time.time()-t0:.0f}s)")

    train_loader = DataLoader(LF2CISODataset(train_pairs, augment=True),
                              batch_size=args.batch, shuffle=True, num_workers=0,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(LF2CISODataset(val_pairs, augment=False),
                            batch_size=args.batch, shuffle=False, num_workers=0,
                            pin_memory=(device.type == 'cuda'))

    model = build_model(device)
    print(f"Model: ResUNet — {count_params(model):,} params")
    if args.init_from_v2:
        v2 = CHECKPOINTS_DIR / 'best_1b_v2.pth'
        if v2.exists():
            ck = torch.load(v2, map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state'])
            print(f"  Init desde v2 (epoch {ck['epoch']})")

    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=LR * 0.01)

    best_psnr, patience, history = -float('inf'), 0, []
    print(f"\nEntrenando hasta {args.epochs} epochs (patience={args.patience})\n")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_psnr = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_psnr = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_psnr': tr_psnr,
                        'val_loss': va_loss, 'val_psnr': va_psnr})
        flag = ''
        if va_psnr > best_psnr:
            best_psnr, patience = va_psnr, 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_psnr': float(va_psnr), 'canvas': CANVAS},
                       CHECKPOINTS_DIR / 'best_1b_lf2ciso.pth')
            flag = ' [saved]'
        else:
            patience += 1
        print(f"Epoch {epoch:3d}/{args.epochs} | train {tr_loss:.4f} PSNR {tr_psnr:.2f} | "
              f"val {va_loss:.4f} PSNR {va_psnr:.2f}{flag}")
        if patience >= args.patience:
            print(f"\nEarly stopping en epoch {epoch}.")
            break

    print(f"\nBest val PSNR (LF->CISO held-out): {best_psnr:.2f} dB")
    (RESULTS_DIR / '02_training_lf2ciso.json').write_text(json.dumps(
        {'best_val_psnr': float(best_psnr), 'epochs': len(history),
         'canvas': CANVAS, 'history': history}, indent=2))


if __name__ == '__main__':
    main()
