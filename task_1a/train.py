"""
train.py — Entrenamiento limpio k-fold de un backbone para Task 1A.

Correcciones vs pipeline viejo:
  - Augmentación SOLO geométrica (sin ColorJitter/GaussianBlur).
  - StratifiedGroupKFold por case_id (cobertura de clases raras en cada fold).
  - weight_decay alto, grad clip, warmup+cosine, early-stop sobre el challenge_score.
  - Guarda OOF best-epoch a results/oof_{label}.npz (estimación honesta para calibrar).

Uso:
  python train.py --label effb4     --backbone efficientnet_b4
  python train.py --label convnexts --backbone convnext_small

Salida: checkpoints/best_{label}_fold{k}.pth, results/train_{label}.json, results/oof_{label}.npz
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CSV_PATH, TRAIN_DIR, RESULTS_DIR, CHECKPOINTS_DIR, ARTIFACT_COLS,
    BATCH_SIZE, EPOCHS, PATIENCE, WARMUP_EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP,
    DROPOUT_1, DROPOUT_2, HIDDEN_DIM, RANDOM_SEED, NUM_WORKERS, N_FOLDS,
    FOCAL_GAMMA, LOSS_W2, LOSS_W_MONO,
)
from data import stratified_kfold_splits, OrdinalDataset, get_transforms, SeverityStratifiedSampler
from model import OrdinalClassifier
from losses import OrdinalFocalLoss
from metrics import challenge_score, scores_from_probs

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def compute_alpha(df, threshold, device):
    labels = df[ARTIFACT_COLS].values
    pos = (labels >= threshold).astype(float).sum(axis=0)
    return torch.tensor((len(labels) - pos) / len(labels), dtype=torch.float32).to(device)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, total_n = 0.0, 0
    p1s, p2s, g1s, g2s = [], [], [], []
    for imgs, g1, g2 in loader:
        imgs, g1, g2 = imgs.to(device), g1.to(device), g2.to(device)
        with torch.set_grad_enabled(train):
            lo1, lo2 = model(imgs)
            loss, _ = criterion(lo1, lo2, g1, g2)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        n = len(imgs)
        total_loss += loss.item() * n
        total_n += n
        p1s.append(torch.sigmoid(lo1).detach().cpu().numpy())
        p2s.append(torch.sigmoid(lo2).detach().cpu().numpy())
        g1s.append(g1.cpu().numpy()); g2s.append(g2.cpu().numpy())
    return (total_loss / max(total_n, 1),
            np.concatenate(p1s), np.concatenate(p2s),
            np.concatenate(g1s), np.concatenate(g2s))


def score_at_half(p1, p2, g1, g2):
    gt = g1.astype(int) + g2.astype(int)
    pred = scores_from_probs(p1, p2, 0.5, 0.5, ARTIFACT_COLS)
    return challenge_score(gt, pred)['score']


def train_fold(fold_idx, df_train, df_val, backbone, label, device):
    ckpt_path = CHECKPOINTS_DIR / f'best_{label}_fold{fold_idx}.pth'
    print(f"\n{'='*60}\nFOLD {fold_idx+1}/{N_FOLDS}  train={len(df_train)}  val={len(df_val)}\n{'='*60}")

    train_ds = OrdinalDataset(df_train, TRAIN_DIR, get_transforms(True))
    val_ds   = OrdinalDataset(df_val,   TRAIN_DIR, get_transforms(False))
    sampler  = SeverityStratifiedSampler(df_train, ARTIFACT_COLS, BATCH_SIZE, RANDOM_SEED + fold_idx)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = OrdinalClassifier(backbone=backbone, dropout1=DROPOUT_1, dropout2=DROPOUT_2,
                              hidden_dim=HIDDEN_DIM, pretrained=True).to(device)
    criterion = OrdinalFocalLoss(FOCAL_GAMMA, compute_alpha(df_train, 1, device),
                                 compute_alpha(df_train, 2, device), LOSS_W2, LOSS_W_MONO).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS),
        CosineAnnealingLR(optimizer, T_max=max(EPOCHS - WARMUP_EPOCHS, 1), eta_min=1e-6),
    ], milestones=[WARMUP_EPOCHS])

    best_score, best_epoch, patience_c, history = 0.0, 0, 0, []
    best_val = None
    for epoch in range(1, EPOCHS + 1):
        tl, *_ = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl, vp1, vp2, vg1, vg2 = run_epoch(model, val_loader, criterion, optimizer, device, False)
        vscore = score_at_half(vp1, vp2, vg1, vg2)
        sched.step()
        is_best = vscore > best_score
        if is_best:
            best_score, best_epoch, patience_c = vscore, epoch, 0
            best_val = (vp1, vp2, vg1, vg2)
            torch.save({'fold': fold_idx, 'epoch': epoch, 'backbone': backbone,
                        'model_state': model.state_dict(), 'val_score': float(vscore)}, ckpt_path)
        else:
            patience_c += 1
        history.append({'epoch': epoch, 'train_loss': round(tl, 4),
                        'val_loss': round(vl, 4), 'val_score': round(vscore, 4)})
        print(f"  ep {epoch:3d}  loss {tl:.4f}  vloss {vl:.4f}  score {vscore:.4f}  {'*' if is_best else ''}")
        if patience_c >= PATIENCE:
            print(f"  early stop @ {epoch}; best={best_score:.4f} (ep {best_epoch})")
            break
    return best_score, best_epoch, history, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--backbone', required=True)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True); CHECKPOINTS_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  backbone={args.backbone}  label={args.label}")

    df, splits = stratified_kfold_splits(CSV_PATH, TRAIN_DIR, N_FOLDS)
    N = len(df)
    oof_p1 = np.full((N, len(ARTIFACT_COLS)), np.nan, dtype=np.float32)
    oof_p2 = np.full((N, len(ARTIFACT_COLS)), np.nan, dtype=np.float32)
    oof_fold = np.full(N, -1, dtype=int)

    results = []
    for k, (df_train, df_val) in enumerate(splits):
        bs, be, hist, best_val = train_fold(k, df_train, df_val, args.backbone, args.label, device)
        vp1, vp2, _, _ = best_val
        gidx = df_val['index'].values
        oof_p1[gidx] = vp1; oof_p2[gidx] = vp2; oof_fold[gidx] = k
        results.append({'fold': k, 'best_score': bs, 'best_epoch': be, 'history': hist})

    scores = [r['best_score'] for r in results]
    gt_ord = (df[ARTIFACT_COLS].values >= 1).astype(int) + (df[ARTIFACT_COLS].values >= 2).astype(int)
    assert not np.isnan(oof_p1).any()
    oof_score = challenge_score(gt_ord, scores_from_probs(oof_p1, oof_p2, 0.5, 0.5, ARTIFACT_COLS))['score']

    print(f"\n{'='*60}\nRESUMEN {args.label}")
    for r in results:
        print(f"  fold {r['fold']}: {r['best_score']:.4f} (ep {r['best_epoch']})")
    print(f"  media folds: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"  OOF score @0.5 (honesto): {oof_score:.4f}")

    np.savez(RESULTS_DIR / f'oof_{args.label}.npz',
             p1=oof_p1, p2=oof_p2, gt=gt_ord, fold=oof_fold, n_folds=N_FOLDS)
    (RESULTS_DIR / f'train_{args.label}.json').write_text(json.dumps({
        'label': args.label, 'backbone': args.backbone,
        'scores_mean': float(np.mean(scores)), 'scores_std': float(np.std(scores)),
        'oof_score': oof_score, 'folds': results,
    }, indent=2))
    print(f"  Guardado: oof_{args.label}.npz, train_{args.label}.json")


if __name__ == '__main__':
    main()
