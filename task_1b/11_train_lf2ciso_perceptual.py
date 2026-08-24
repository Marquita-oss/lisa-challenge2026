"""Opcion 2 (fase 2): entrenar LF->CISO REGISTRADO con L1+SSIM+VGG-perceptual.

Diferencias con el lf2ciso fallido:
  - Target = CISO REGISTRADO (rigid+MI, corr ~0.63) via cache de fase 1, no array-resize.
  - Loss anade termino PERCEPTUAL VGG (robusto a desalineamiento residual, transfiere
    textura de alto campo sin exigir match pixel-exacto -> evita el blur de L1 puro).
  - Guarda checkpoints periodicos en checkpoints/perc/ para seleccion por FID held-out (fase 3).

Uso:
  python task_1b/11_train_lf2ciso_perceptual.py --epochs 60 --l1 0.5 --ssim 0.3 --perc 0.1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from task_1b.config import CHECKPOINTS_DIR, RESULTS_DIR, TASK_DIR, RANDOM_SEED, LR, WEIGHT_DECAY
from task_1b.model import build_model, count_params
from task_1b.utils.metrics import ssim_torch, psnr_torch

CANVAS = 160
CACHE = TASK_DIR / 'reg_cache' / 'train'


def pad_to(a, size=CANVAS):
    H, W = a.shape
    out = np.zeros((size, size), np.float32)
    out[:H, :W] = a
    return out


class RegPairDataset(Dataset):
    """Slices from cached registered npz (one file = one case/plane)."""

    def __init__(self, files, augment=False):
        self.index = []       # (file_idx, slice_idx)
        self.arrays = []      # lazy-loaded (lf, ci) per file
        self.files = files
        for fi, f in enumerate(files):
            d = np.load(f)
            n = d['lf'].shape[0]
            self.arrays.append((d['lf'], d['ci']))
            self.index += [(fi, i) for i in range(n)]
        self.augment = augment

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, si = self.index[idx]
        lf = self.arrays[fi][0][si].astype(np.float32)
        ci = self.arrays[fi][1][si].astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                lf, ci = np.fliplr(lf), np.fliplr(ci)
            if np.random.rand() < 0.5:
                lf, ci = np.flipud(lf), np.flipud(ci)
        lf = pad_to(np.ascontiguousarray(lf))
        ci = pad_to(np.ascontiguousarray(ci))
        return torch.from_numpy(lf)[None], torch.from_numpy(ci)[None]


class VGGPerceptual(nn.Module):
    """Perceptual loss on VGG16 features (relu1_2, relu2_2, relu3_3). Frozen."""

    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        # slice boundaries for relu1_2(3), relu2_2(8), relu3_3(15)
        self.blocks = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:16],
        ]).to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        p = (pred.repeat(1, 3, 1, 1) - self.mean) / self.std
        t = (target.repeat(1, 3, 1, 1) - self.mean) / self.std
        loss = 0.0
        for blk in self.blocks:
            p = blk(p); t = blk(t)
            loss = loss + self.l1(p, t)
        return loss


def run_epoch(model, loader, vgg, w, optimizer, device, train):
    model.train() if train else model.eval()
    torch.set_grad_enabled(train)
    agg = {'loss': 0, 'l1': 0, 'ssim': 0, 'perc': 0, 'psnr': 0, 'n': 0}
    l1fn = nn.L1Loss()
    for lf, ci in loader:
        lf, ci = lf.to(device), ci.to(device)
        if train:
            optimizer.zero_grad()
        pred = model(lf)
        l1 = l1fn(pred, ci)
        ssim = 1.0 - ssim_torch(pred, ci)
        perc = vgg(pred, ci)
        loss = w['l1'] * l1 + w['ssim'] * ssim + w['perc'] * perc
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        agg['loss'] += float(loss.item()); agg['l1'] += float(l1.item())
        agg['ssim'] += float(ssim.item()); agg['perc'] += float(perc.item())
        agg['psnr'] += psnr_torch(pred, ci); agg['n'] += 1
    torch.set_grad_enabled(True)
    n = max(agg['n'], 1)
    return {k: agg[k] / n for k in ['loss', 'l1', 'ssim', 'perc', 'psnr']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--l1', type=float, default=0.5)
    ap.add_argument('--ssim', type=float, default=0.3)
    ap.add_argument('--perc', type=float, default=0.1)
    ap.add_argument('--n-val-cases', type=int, default=10)
    ap.add_argument('--save-every', type=int, default=5)
    ap.add_argument('--min-corr', type=float, default=0.30,
                    help='drop cached planes whose registration corr is below this (misaligned target)')
    ap.add_argument('--init-from-v2', action='store_true', default=True)
    args = ap.parse_args()

    np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w = {'l1': args.l1, 'ssim': args.ssim, 'perc': args.perc}
    print(f"Device {device}  weights {w}")

    files = sorted(CACHE.glob('*.npz'))
    if not files:
        print(f"ERROR: no cache en {CACHE}. Corre 10_build_registered_pairs.py --split train"); return
    # drop poorly-registered planes (misaligned target hurts training)
    qc = TASK_DIR / 'reg_cache' / 'registration_qc.csv'
    if qc.exists() and args.min_corr > 0:
        import csv
        good = {(r['case'], r['plane']): float(r['corr']) for r in csv.DictReader(open(qc))
                if r['split'] == 'train'}
        kept = []
        for f in files:
            cid, plane = f.stem.split('_')[0], f.stem.split('_')[1]
            if good.get((cid, plane), 1.0) >= args.min_corr:
                kept.append(f)
        print(f"QC filter (corr>={args.min_corr}): {len(kept)}/{len(files)} planes kept")
        files = kept
    # split by CASE id (prefix before first _)
    cases = sorted({f.name.split('_')[0] for f in files})
    rng = np.random.RandomState(RANDOM_SEED)
    val_cases = set(rng.choice(cases, size=min(args.n_val_cases, len(cases)//2), replace=False))
    tr_files = [f for f in files if f.name.split('_')[0] not in val_cases]
    va_files = [f for f in files if f.name.split('_')[0] in val_cases]
    print(f"Casos: {len(cases)}  train {len(cases)-len(val_cases)} / val {len(val_cases)}")
    print(f"Cache files: train {len(tr_files)}  val {len(va_files)}")

    tr = RegPairDataset(tr_files, augment=True)
    va = RegPairDataset(va_files, augment=False)
    print(f"Slices: train {len(tr)}  val {len(va)}")
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0,
                     pin_memory=(device.type == 'cuda'))
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=0,
                     pin_memory=(device.type == 'cuda'))

    model = build_model(device)
    print(f"Model ResUNet {count_params(model):,} params")
    if args.init_from_v2:
        v2 = CHECKPOINTS_DIR / 'best_1b_v2.pth'
        if v2.exists():
            ck = torch.load(v2, map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state']); print(f"  init desde v2 (epoch {ck['epoch']})")

    vgg = VGGPerceptual(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=LR * 0.01)

    ckdir = CHECKPOINTS_DIR / 'perc'; ckdir.mkdir(parents=True, exist_ok=True)
    best_val, patience, history = float('inf'), 0, []
    print(f"\nEntrenando {args.epochs} epochs (patience {args.patience})\n")
    for epoch in range(1, args.epochs + 1):
        tr_m = run_epoch(model, trl, vgg, w, optimizer, device, True)
        va_m = run_epoch(model, val, vgg, w, optimizer, device, False)
        scheduler.step()
        history.append({'epoch': epoch, 'train': tr_m, 'val': va_m})
        flag = ''
        if va_m['loss'] < best_val:
            best_val, patience = va_m['loss'], 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': best_val, 'val_psnr': va_m['psnr'], 'canvas': CANVAS,
                        'weights': w}, CHECKPOINTS_DIR / 'best_1b_perc.pth')
            flag = ' [best]'
        else:
            patience += 1
        if epoch % args.save_every == 0:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_psnr': va_m['psnr'], 'canvas': CANVAS, 'weights': w},
                       ckdir / f'ep{epoch:03d}.pth')
        print(f"E{epoch:3d} | tr loss {tr_m['loss']:.4f} l1 {tr_m['l1']:.3f} ssim {tr_m['ssim']:.3f} "
              f"perc {tr_m['perc']:.3f} PSNR {tr_m['psnr']:.2f} | va loss {va_m['loss']:.4f} "
              f"PSNR {va_m['psnr']:.2f}{flag}")
        if patience >= args.patience:
            print(f"\nEarly stop epoch {epoch}."); break

    (RESULTS_DIR / '11_train_perceptual.json').write_text(json.dumps(
        {'best_val_loss': best_val, 'weights': w, 'history': history}, indent=2))
    print(f"\nBest val loss {best_val:.4f}  ckpts -> {ckdir}")


if __name__ == '__main__':
    main()
