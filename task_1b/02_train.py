"""Task 1b — Training.

Trains a ResUNet denoiser on synthetic degradation of nonoise_nomotion images.
Early stopping on validation PSNR. Saves best checkpoint to checkpoints/best_1b.pth.
Results written to results/02_training.json.
"""
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
    NOISE_SIGMA_MIN, NOISE_SIGMA_MAX, MOTION_PROB,
    MOTION_GHOSTS_MIN, MOTION_GHOSTS_MAX, MOTION_ALPHA_MIN, MOTION_ALPHA_MAX,
)
from task_1b.dataset import build_split
from task_1b.losses import CombinedLoss
from task_1b.model import build_model, count_params
from task_1b.utils.metrics import psnr_torch

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for degraded, clean in loader:
        degraded, clean = degraded.to(device), clean.to(device)
        optimizer.zero_grad()
        pred = model(degraded)
        # Crop to clean size (both may differ if padding was asymmetric)
        if pred.shape != clean.shape:
            pred = pred[:, :, :clean.shape[2], :clean.shape[3]]
        loss = criterion(pred, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    for degraded, clean in loader:
        degraded, clean = degraded.to(device), clean.to(device)
        pred = model(degraded)
        if pred.shape != clean.shape:
            pred = pred[:, :, :clean.shape[2], :clean.shape[3]]
        loss = criterion(pred, clean)
        total_loss += loss.item()
        total_psnr += psnr_torch(pred, clean)
        n += 1
    return total_loss / max(n, 1), total_psnr / max(n, 1)


def main():
    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    if not CSV_NONOISE_NOMOTION.exists():
        print(f"[ERROR] Partition CSV not found: {CSV_NONOISE_NOMOTION}")
        sys.exit(1)

    print("\nBuilding datasets (split by case_id)...")
    train_ds, val_ds = build_split(
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
        num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=device.type == 'cuda',
    )

    model = build_model(device)
    print(f"\nModel: ResUNet — {count_params(model):,} parameters")

    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    best_psnr = -float('inf')
    patience_count = 0
    history = []

    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE} on val PSNR)...\n")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = float(optimizer.param_groups[0]['lr'])
        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_psnr': float(val_psnr),
            'lr': lr_now,
        })

        flag = ''
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_count = 0
            ckpt = CHECKPOINTS_DIR / 'best_1b.pth'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_psnr': float(val_psnr),
                'val_loss': float(val_loss),
            }, ckpt)
            flag = ' [saved]'
        else:
            patience_count += 1

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_PSNR={val_psnr:.2f} dB | lr={lr_now:.2e}{flag}")

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best val PSNR: {best_psnr:.2f} dB")

    from task_1b.config import MIN_PSNR
    passed = best_psnr >= MIN_PSNR
    print(f"Criterion (>={MIN_PSNR} dB): {'PASS' if passed else 'FAIL'}")

    results = {
        'best_val_psnr': float(best_psnr),
        'criterion_pass': bool(passed),
        'epochs_trained': len(history),
        'elapsed_minutes': float(elapsed / 60),
        'history': history,
    }
    out = RESULTS_DIR / '02_training.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out}")


if __name__ == '__main__':
    main()
