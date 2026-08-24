"""Task 1B v2 — Training with physics-based augmentation + adversarial loss.

Differences from v1 (02_train.py):
  - SyntheticDenoiseDatasetV2: composite_degrade() instead of Rician+ghosting.
    Physics-based artifacts: k-space noise, Gibbs, banding, zipper, bias field,
    motion ghosting v2 (rotation+translation+line-swap).
  - CombinedLossWithAdversarial: frozen Task 1A classifier penalises Noise/Motion
    in the enhanced output starting from epoch ADV_WARMUP_EPOCHS+1.
  - Optional --init-from-v1: load best_1b.pth as starting weights (fine-tune).
  - Optional --use-npy: acelera el entrenamiento usando el cache .npy pre-procesado
    (elimina el cuello de botella de I/O). Requiere ejecutar primero:
    python task_1b/preprocess_to_npy.py

Saves best checkpoint to checkpoints/best_1b_v2.pth.
Results written to results/02_training_v2.json.

Usage:
  python task_1b/02_train_v2.py
  python task_1b/02_train_v2.py --init-from-v1
  python task_1b/02_train_v2.py --use-npy --workers 2
  python task_1b/02_train_v2.py --init-from-v1 --use-npy --workers 2
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
    CSV_NONOISE_NOMOTION, BATCH_SIZE, EPOCHS, PATIENCE, LR, WEIGHT_DECAY,
    NUM_WORKERS, VAL_SPLIT, CHECKPOINTS_DIR, RESULTS_DIR, RANDOM_SEED,
    TASK_1A_CHECKPOINT, MIN_PSNR, LAMBDA_ADV, ADV_WARMUP_EPOCHS,
    CKPT_V2, TRAIN_RESULTS_V2, TASK_DIR,
)
from task_1b.dataset import build_split_v2, build_split_npy
from task_1b.losses import CombinedLossWithAdversarial, Task1aDiscriminator
from task_1b.model import build_model, count_params
from task_1b.utils.metrics import psnr_torch

NPY_TRAIN_DIR = TASK_DIR / 'npy_cache' / 'train_nonoise_nomotion'

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_l1 = total_ssim = total_adv = 0.0
    for degraded, clean in loader:
        degraded, clean = degraded.to(device), clean.to(device)
        optimizer.zero_grad()
        pred = model(degraded)
        loss, parts = criterion(pred, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_l1   += parts['l1']
        total_ssim += parts['ssim']
        total_adv  += parts['adv']
    n = max(len(loader), 1)
    return total_loss / n, total_l1 / n, total_ssim / n, total_adv / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    for degraded, clean in loader:
        degraded, clean = degraded.to(device), clean.to(device)
        pred = model(degraded)
        loss, _ = criterion(pred, clean)
        total_loss += loss.item()
        total_psnr += psnr_torch(pred, clean)
        n += 1
    return total_loss / max(n, 1), total_psnr / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-from-v1', action='store_true',
                        help='Initialise weights from best_1b.pth (fine-tune v1 -> v2)')
    parser.add_argument('--use-npy', action='store_true',
                        help='Usar cache .npy para acelerar I/O (requiere preprocess_to_npy.py)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Num workers DataLoader (default: 0 sin NPY, 2 con NPY)')
    args = parser.parse_args()

    # Con NPY el default de workers es 2 (multiprocessing seguro sin nibabel)
    n_workers = args.workers if args.workers is not None else (2 if args.use_npy else NUM_WORKERS)

    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    if not CSV_NONOISE_NOMOTION.exists():
        print(f"[ERROR] Partition CSV not found: {CSV_NONOISE_NOMOTION}")
        sys.exit(1)

    # ── Discriminator ──────────────────────────────────────────────────────────
    if not TASK_1A_CHECKPOINT.exists():
        print(f"[ERROR] Task 1A checkpoint not found: {TASK_1A_CHECKPOINT}")
        sys.exit(1)

    print(f"\nLoading frozen Task 1A discriminator from:\n  {TASK_1A_CHECKPOINT}")
    discriminator = Task1aDiscriminator(TASK_1A_CHECKPOINT, device)
    print("  Task 1A discriminator loaded and frozen.")

    # ── Datasets ───────────────────────────────────────────────────────────
    if args.use_npy:
        if not NPY_TRAIN_DIR.exists():
            print(f"[ERROR] Cache .npy no encontrado: {NPY_TRAIN_DIR}")
            print("  Ejecutar primero: python task_1b/preprocess_to_npy.py")
            sys.exit(1)
        print(f"\nUsando cache .npy: {NPY_TRAIN_DIR}")
        train_ds, val_ds = build_split_npy(
            NPY_TRAIN_DIR,
            val_split=VAL_SPLIT,
            seed=RANDOM_SEED,
            use_physics_degrade=True,
        )
        print("  (degradacion composite_degrade aplicada en tiempo real)")
    else:
        print("\nBuilding v2 datasets (physics-based degradation, split by case_id)...")
        train_ds, val_ds = build_split_v2(
            CSV_NONOISE_NOMOTION,
            val_split=VAL_SPLIT,
            seed=RANDOM_SEED,
        )
    print(f"  Train slices: {len(train_ds)}")
    print(f"  Val   slices: {len(val_ds)}")

    if len(train_ds) == 0:
        print("[ERROR] No training samples found. Check that data/train/ is populated.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=n_workers, pin_memory=device.type == 'cuda',
        persistent_workers=(n_workers > 0),
        prefetch_factor=(2 if n_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=n_workers, pin_memory=device.type == 'cuda',
        persistent_workers=(n_workers > 0),
        prefetch_factor=(2 if n_workers > 0 else None),
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_model(device)
    print(f"\nModel: ResUNet — {count_params(model):,} parameters")

    if args.init_from_v1:
        v1_ckpt = CHECKPOINTS_DIR / 'best_1b.pth'
        if v1_ckpt.exists():
            ckpt = torch.load(v1_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            print(f"  Initialised from v1 checkpoint (epoch {ckpt['epoch']}, "
                  f"PSNR={ckpt['val_psnr']:.2f} dB)")
        else:
            print(f"  [WARN] --init-from-v1 requested but {v1_ckpt} not found. "
                  f"Training from scratch.")

    # ── Loss / optimiser ───────────────────────────────────────────────────────
    criterion = CombinedLossWithAdversarial(
        discriminator=discriminator,
        lambda_adv=LAMBDA_ADV,
    )
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_psnr = -float('inf')
    patience_count = 0
    history = []

    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE} on val PSNR)")
    print(f"Adversarial loss (lambda={LAMBDA_ADV}) activates after epoch {ADV_WARMUP_EPOCHS}\n")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Activate adversarial term after warmup
        if epoch == ADV_WARMUP_EPOCHS + 1:
            criterion.adv_active = True
            print(f"  [Epoch {epoch}] Adversarial loss activated.")

        train_loss, tl1, tssim, tadv = train_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = float(optimizer.param_groups[0]['lr'])
        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_l1':   float(tl1),
            'train_ssim': float(tssim),
            'train_adv':  float(tadv),
            'val_loss':   float(val_loss),
            'val_psnr':   float(val_psnr),
            'lr':         lr_now,
            'adv_active': criterion.adv_active,
        })

        flag = ''
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_count = 0
            ckpt_path = CHECKPOINTS_DIR / CKPT_V2
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_psnr': float(val_psnr),
                'val_loss': float(val_loss),
            }, ckpt_path)
            flag = ' [saved]'
        else:
            patience_count += 1

        adv_tag = f' adv={tadv:.4f}' if criterion.adv_active else ''
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train={train_loss:.4f} (l1={tl1:.4f} ssim={tssim:.4f}{adv_tag}) | "
              f"val={val_loss:.4f} | PSNR={val_psnr:.2f} dB | lr={lr_now:.2e}{flag}")

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs).")
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best val PSNR: {best_psnr:.2f} dB")
    passed = best_psnr >= MIN_PSNR
    print(f"Criterion (>={MIN_PSNR} dB): {'PASS' if passed else 'FAIL'}")

    results = {
        'version': 'v2',
        'init_from_v1': args.init_from_v1,
        'lambda_adv': float(LAMBDA_ADV),
        'adv_warmup_epochs': ADV_WARMUP_EPOCHS,
        'best_val_psnr': float(best_psnr),
        'criterion_pass': bool(passed),
        'epochs_trained': len(history),
        'elapsed_minutes': float(elapsed / 60),
        'history': history,
    }
    out = RESULTS_DIR / TRAIN_RESULTS_V2
    out.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out}")


if __name__ == '__main__':
    main()
