"""
oof.py — Genera predicciones out-of-fold (OOF) honestas para un backbone.

Cada fold predice SOLO su val held-out (imágenes que su modelo nunca vio).
Concatenadas cubren las 532 imágenes sin fuga -> estimación que iguala el leaderboard.

Uso (Fase 0, reutilizando checkpoints v9 existentes):
  python oof.py --label effb4 --backbone efficientnet_b4 \
                --ckpt-pattern best_1a_v9_fold{k}.pth --img-size 256 --split legacy

Uso (tras reentrenar limpio):
  python oof.py --label effb4 --backbone efficientnet_b4 \
                --ckpt-pattern best_effb4_fold{k}.pth --split stratified

Salida: results/oof_{label}.npz  (p1, p2, gt_ordinal, n_folds)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CSV_PATH, TRAIN_DIR, RESULTS_DIR, CHECKPOINTS_DIR, ARTIFACT_COLS, N_FOLDS
from data import legacy_kfold_splits, stratified_kfold_splits, OrdinalDataset
from inference import load_model, predict_probs, val_transform
from metrics import challenge_score, scores_from_probs


def generate_oof(label, backbone, ckpt_pattern, img_size, split, tta, hidden_dim=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  backbone={backbone}  split={split}  tta={tta}")

    splitter = legacy_kfold_splits if split == 'legacy' else stratified_kfold_splits
    df, splits = splitter(CSV_PATH, TRAIN_DIR, N_FOLDS)
    N = len(df)
    print(f"Imágenes: {N}  |  folds: {len(splits)}")

    oof_p1 = np.full((N, len(ARTIFACT_COLS)), np.nan, dtype=np.float32)
    oof_p2 = np.full((N, len(ARTIFACT_COLS)), np.nan, dtype=np.float32)
    oof_fold = np.full(N, -1, dtype=int)

    for k, (_, df_val) in enumerate(splits):
        ckpt = CHECKPOINTS_DIR / ckpt_pattern.format(k=k)
        if not ckpt.exists():
            print(f"  [ERROR] falta checkpoint: {ckpt}")
            sys.exit(1)
        model, meta = load_model(backbone, ckpt, device, hidden_dim)
        ds = OrdinalDataset(df_val, TRAIN_DIR, val_transform(img_size))
        p1, p2 = predict_probs(model, ds, device, tta=tta)
        gidx = df_val['index'].values
        oof_p1[gidx] = p1
        oof_p2[gidx] = p2
        oof_fold[gidx] = k
        vs = meta.get('val_score', float('nan')) if isinstance(meta, dict) else float('nan')
        print(f"  fold {k}: {len(df_val):3d} val imgs  (ckpt val_score={vs:.4f})")

    assert not np.isnan(oof_p1).any(), "Cobertura OOF incompleta — alguna imagen sin predicción"

    scores = df[ARTIFACT_COLS].values
    gt_ord = (scores >= 1).astype(int) + (scores >= 2).astype(int)

    base = challenge_score(gt_ord, scores_from_probs(oof_p1, oof_p2, 0.5, 0.5, ARTIFACT_COLS))
    print(f"\n  OOF score @0.5/0.5 (HONESTO, sin fuga): {base['score']:.4f}")
    print(f"    f1_micro={base['f1_micro']:.4f}  f1_macro={base['f1_macro']:.4f}")

    out = RESULTS_DIR / f'oof_{label}.npz'
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(out, p1=oof_p1, p2=oof_p2, gt=gt_ord, fold=oof_fold, n_folds=len(splits))
    print(f"  Guardado: {out}")
    return base['score']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--ckpt-pattern', default='best_{label}_fold{k}.pth')
    ap.add_argument('--img-size', type=int, default=256)
    ap.add_argument('--split', choices=['legacy', 'stratified'], default='legacy')
    ap.add_argument('--no-tta', action='store_true')
    ap.add_argument('--hidden-dim', type=int, default=512)
    args = ap.parse_args()
    pattern = args.ckpt_pattern.replace('{label}', args.label)
    generate_oof(args.label, args.backbone, pattern, args.img_size,
                 args.split, tta=not args.no_tta, hidden_dim=args.hidden_dim)


if __name__ == '__main__':
    main()
