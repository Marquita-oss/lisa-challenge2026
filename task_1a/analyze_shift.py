"""
analyze_shift.py — Diagnostica desplazamiento de distribución train(OOF) -> test.

Compara, por clase:
  - prior de train (tasa gt>=1)
  - probabilidades OOF en train (media p1, tasa de positivos)
  - probabilidades en el test ciego (media p1, tasa de positivos)
Si el test predice muchos menos positivos que el train con las mismas probabilidades,
hay covariate shift (el modelo es menos confiado en las imágenes de test).

No entrena nada: solo inferencia con los checkpoints del ensemble.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (RESULTS_DIR, CHECKPOINTS_DIR, ARTIFACT_COLS, N_FOLDS,
                    VAL_SINGLE_DIR, VAL_COMPLETE_DIR, BACKBONES, IMG_SIZE)
from inference import load_model, predict_probs, val_transform
from predict import collect_lf_files, SubmissionDataset
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Train OOF (ensemble = promedio de backbones) ---
p1s, gts = [], None
for _, lab in BACKBONES:
    d = np.load(RESULTS_DIR / f'oof_{lab}.npz')
    p1s.append(d['p1'])
    if gts is None:
        gts = d['gt']
oof_p1 = np.mean(p1s, axis=0)
train_prior = (gts >= 1).mean(axis=0)

# --- Test (ensemble 10 modelos, sin TTA para rapidez) ---
files = collect_lf_files(VAL_SINGLE_DIR) + collect_lf_files(VAL_COMPLETE_DIR)
acc = []
for backbone, lab in BACKBONES:
    ds = SubmissionDataset(files, val_transform(IMG_SIZE))
    for k in range(N_FOLDS):
        ck = CHECKPOINTS_DIR / f'best_{lab}_fold{k}.pth'
        if ck.exists():
            m, _ = load_model(backbone, ck, device)
            p1, _ = predict_probs(m, ds, device, tta=False)
            acc.append(p1)
test_p1 = np.mean(acc, axis=0)

thr = json.loads((RESULTS_DIR / 'thresholds.json').read_text())['t1_per_class']

print(f"\n{'Clase':13s} {'prior':>6s} {'oof_mean':>9s} {'oof_pos%':>8s} {'test_mean':>10s} {'test_pos%@t':>11s} {'test_pos%@.5':>12s}")
print("-" * 74)
for k, col in enumerate(ARTIFACT_COLS):
    t1 = thr[col]
    print(f"{col:13s} {train_prior[k]:6.2f} {oof_p1[:,k].mean():9.3f} "
          f"{(oof_p1[:,k]>=t1).mean():8.2f} {test_p1[:,k].mean():10.3f} "
          f"{(test_p1[:,k]>=t1).mean():11.2f} {(test_p1[:,k]>=0.5).mean():12.2f}")

print(f"\nTasa global positivos: train_prior={train_prior.mean():.3f}  "
      f"OOF@t={np.mean([(oof_p1[:,k]>=thr[c]).mean() for k,c in enumerate(ARTIFACT_COLS)]):.3f}  "
      f"TEST@t={np.mean([(test_p1[:,k]>=thr[c]).mean() for k,c in enumerate(ARTIFACT_COLS)]):.3f}")
np.savez(RESULTS_DIR / 'test_probs.npz', p1=test_p1, files=[f.name for f in files])
print(f"Guardado test_probs.npz")
