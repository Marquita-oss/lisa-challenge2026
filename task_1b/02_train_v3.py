"""Task 1b — Entrenamiento v3 con cache .npy (GPU-bound).

Diferencias respecto a v2 (02_train_v2.py):
  - Usa NpySliceDataset en vez de SyntheticDenoiseDatasetV2
  - Los slices ya estan normalizados y redimensionados en disco
  - El DataLoader no hace decompress gzip ni PIL resize en el hot-path
  - NUM_WORKERS > 0 funciona de forma segura (no hay nibabel en workers)
  - pin_memory=True + persistent_workers=True activos por defecto con CUDA

Prerequisito (ejecutar una sola vez):
  python task_1b/preprocess_to_npy.py

Uso:
  python task_1b/02_train_v3.py
  python task_1b/02_train_v3.py --no-adv          # solo L1+SSIM, sin adversarial
  python task_1b/02_train_v3.py --init-from-v1    # fine-tune desde best_1b.pth
  python task_1b/02_train_v3.py --init-from-v2    # fine-tune desde best_1b_v2.pth
  python task_1b/02_train_v3.py --workers 4       # num workers DataLoader
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_1b.config import (
    BATCH_SIZE, EPOCHS, PATIENCE, LR, WEIGHT_DECAY,
    VAL_SPLIT, CHECKPOINTS_DIR, RESULTS_DIR, RANDOM_SEED,
    TASK_DIR, TASK_1A_CHECKPOINT, TASK_1A_THRESHOLDS,
    ADV_WARMUP_EPOCHS, LAMBDA_ADV,
)
from task_1b.dataset import build_split_npy
from task_1b.losses import CombinedLoss, CombinedLossWithAdversarial, Task1aDiscriminator
from task_1b.model import build_model, count_params
from task_1b.utils.metrics import psnr_torch

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NPY_TRAIN_DIR = TASK_DIR / 'npy_cache' / 'train_nonoise_nomotion'
CKPT_V3 = 'best_1b_v3.pth'


def train_epoch(model, loader, criterion, optimizer, device, adv=False):
    model.train()
    total_loss, total_l1, total_ssim, total_adv = 0.0, 0.0, 0.0, 0.0
    n = 0

    for degraded, clean in loader:
        degraded, clean = degraded.to(device, non_blocking=True), \
                          clean.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(degraded)
        if pred.shape != clean.shape:
            pred = pred[:, :, :clean.shape[2], :clean.shape[3]]

        if adv:
            loss, components = criterion(pred, clean)
        else:
            loss = criterion(pred, clean)
            components = {}

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_l1   += components.get('l1', 0.0)
        total_ssim += components.get('ssim', 0.0)
        total_adv  += components.get('adv', 0.0)
        n += 1

    d = max(n, 1)
    return {
        'loss': total_loss / d,
        'l1':   total_l1 / d,
        'ssim': total_ssim / d,
        'adv':  total_adv / d,
    }


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_psnr, n = 0.0, 0
    for degraded, clean in loader:
        degraded, clean = degraded.to(device, non_blocking=True), \
                          clean.to(device, non_blocking=True)
        pred = model(degraded)
        if pred.shape != clean.shape:
            pred = pred[:, :, :clean.shape[2], :clean.shape[3]]
        total_psnr += psnr_torch(pred, clean)
        n += 1
    return total_psnr / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-adv', action='store_true',
                        help='Desactivar loss adversarial (solo L1+SSIM)')
    parser.add_argument('--init-from-v1', action='store_true',
                        help='Fine-tune desde best_1b.pth')
    parser.add_argument('--init-from-v2', action='store_true',
                        help='Fine-tune desde best_1b_v2.pth')
    parser.add_argument('--workers', type=int, default=2,
                        help='Numero de workers del DataLoader (default: 2)')
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_adv = not args.no_adv

    print("=" * 60)
    print("Task 1b — Entrenamiento v3 (cache .npy)")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  NPY cache:   {NPY_TRAIN_DIR}")
    print(f"  Workers:     {args.workers}")
    print(f"  Adversarial: {'ON' if use_adv else 'OFF'}")
    print(f"  Checkpoint:  {CKPT_V3}")

    if not NPY_TRAIN_DIR.exists():
        print(f"\n[ERROR] Cache .npy no encontrado: {NPY_TRAIN_DIR}")
        print("  Ejecutar primero:")
        print("    python task_1b/preprocess_to_npy.py")
        sys.exit(1)

    print("\nConstruyendo datasets desde cache .npy...")
    train_ds, val_ds = build_split_npy(
        NPY_TRAIN_DIR,
        val_split=VAL_SPLIT,
        seed=RANDOM_SEED,
        use_physics_degrade=True,
    )
    print(f"  Train slices: {len(train_ds)}")
    print(f"  Val   slices: {len(val_ds)}")

    cuda = device.type == 'cuda'
    # persistent_workers evita recrear procesos entre epochs
    # prefetch_factor=2 prellena 2 batches por worker
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=args.workers,
        pin_memory=cuda,
        persistent_workers=(args.workers > 0),
        prefetch_factor=(2 if args.workers > 0 else None),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    model = build_model(device)
    print(f"\nModelo: ResUNet — {count_params(model):,} parametros")

    # Fine-tune desde checkpoint previo
    if args.init_from_v2:
        src = CHECKPOINTS_DIR / 'best_1b_v2.pth'
        ckpt = torch.load(src, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        print(f"  Inicializado desde best_1b_v2.pth (epoch {ckpt['epoch']})")
    elif args.init_from_v1:
        src = CHECKPOINTS_DIR / 'best_1b.pth'
        ckpt = torch.load(src, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        print(f"  Inicializado desde best_1b.pth (epoch {ckpt['epoch']})")

    # Loss
    if use_adv and TASK_1A_CHECKPOINT.exists():
        print(f"  Cargando discriminador Task 1a: {TASK_1A_CHECKPOINT.name}")
        discriminator = Task1aDiscriminator(TASK_1A_CHECKPOINT, device)
        criterion_adv = CombinedLossWithAdversarial(discriminator)
        criterion_rec = CombinedLoss()   # para las primeras epochs sin adv
        use_adv_loss  = True
    else:
        if use_adv:
            print("  [AVISO] Checkpoint Task 1a no encontrado — usando solo L1+SSIM")
        criterion_rec = CombinedLoss()
        use_adv_loss  = False

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_psnr = -float('inf')
    patience_count = 0
    history = []

    print(f"\nEntrenando hasta {EPOCHS} epochs (patience={PATIENCE} en val PSNR)...\n")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        adv_active = use_adv_loss and (epoch > ADV_WARMUP_EPOCHS)
        if use_adv_loss:
            criterion_adv.adv_active = adv_active
            criterion = criterion_adv
            is_adv = True
        else:
            criterion = criterion_rec
            is_adv = False

        train_stats = train_epoch(model, train_loader, criterion, optimizer, device, adv=is_adv)
        val_psnr    = val_epoch(model, val_loader, device)
        scheduler.step()

        lr_now = float(optimizer.param_groups[0]['lr'])
        row = {
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'val_psnr': float(val_psnr),
            'lr': lr_now,
            'adv_active': adv_active,
            **{k: v for k, v in train_stats.items() if k != 'loss'},
        }
        history.append(row)

        flag = ''
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_psnr': float(val_psnr),
            }, CHECKPOINTS_DIR / CKPT_V3)
            flag = ' [saved]'
        else:
            patience_count += 1

        adv_str = f" adv={train_stats['adv']:.4f}" if adv_active else ""
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"loss={train_stats['loss']:.4f} "
            f"l1={train_stats['l1']:.4f} ssim={train_stats['ssim']:.4f}"
            f"{adv_str} | "
            f"valPSNR={val_psnr:.2f}dB | lr={lr_now:.2e}{flag}"
        )

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping en epoch {epoch}.")
            break

    elapsed = time.time() - t0
    print(f"\nEntrenamiento completado en {elapsed/60:.1f} min")
    print(f"Mejor val PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoint guardado: {CHECKPOINTS_DIR / CKPT_V3}")

    results = {
        'checkpoint': CKPT_V3,
        'best_val_psnr': float(best_psnr),
        'epochs_trained': len(history),
        'elapsed_minutes': float(elapsed / 60),
        'adversarial': use_adv_loss,
        'history': history,
    }
    out = RESULTS_DIR / '02_training_v3.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"Resultados guardados en {out}")
    print(f"\nPara generar la submission con este checkpoint:")
    print(f"  python task_1b/04_predict_submission.py --ckpt {CKPT_V3}")


if __name__ == '__main__':
    main()
