"""
predict.py — Submission Task 1A (114 filas) por ensemble de folds/backbones + TTA.

Promedia las probabilidades de todos los checkpoints disponibles, aplica los umbrales
calibrados (results/thresholds.json) y escribe LISA_LF_QC_predictions.csv en la raíz.

Mantiene las 114 filas (72 single_plane + 14 complete x 3 planos), una predicción
independiente por imagen-plano (consistente con el entrenamiento per-plano).

Uso (Fase 0, checkpoints v9 existentes):
  python predict.py --spec efficientnet_b4:best_1a_v9_fold{k}.pth:256

Uso (pipeline nuevo, ensemble B4+ConvNeXt):
  python predict.py --spec efficientnet_b4:best_effb4_fold{k}.pth:256 \
                    --spec convnext_small:best_convnexts_fold{k}.pth:256
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    VAL_SINGLE_DIR, VAL_COMPLETE_DIR, ROOT, RESULTS_DIR, CHECKPOINTS_DIR,
    ARTIFACT_COLS, N_FOLDS, N_SUBMISSION_ROWS, SUBMISSION_FILE, BACKBONES, IMG_SIZE,
)
from inference import load_model, predict_probs, val_transform
from utils.io import load_lf_multichannel
from calibrate import load_oof


def extract_patient_id(fname: str) -> str:
    m = re.search(r'(?i)lisa_validation_(\d+)', fname)
    if m:
        return f'LISA_LF_{m.group(1)}'
    m = re.search(r'(?i)lisa_lf_(\d+)', fname)
    if m:
        return f'LISA_LF_{m.group(1)}'
    return Path(fname).stem


def collect_lf_files(directory: Path) -> list:
    if not directory.exists():
        return []
    return [f for f in sorted(directory.rglob('*.nii.gz')) if '_ciso' not in f.name.lower()]


class SubmissionDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_lf_multichannel(self.paths[idx])
        return self.transform(img), self.paths[idx].name


def load_thresholds():
    p = RESULTS_DIR / 'thresholds.json'
    if p.exists():
        d = json.loads(p.read_text())
        print(f"  Umbrales: {p.name}  (OOF calibrado score={d.get('calibrated',{}).get('score',0):.4f})")
        return d['t1_per_class'], d['t2_per_class']
    print("  AVISO: sin thresholds.json -> 0.5/0.5")
    return {c: 0.5 for c in ARTIFACT_COLS}, {c: 0.5 for c in ARTIFACT_COLS}


def parse_specs(spec_args):
    if spec_args:
        specs = []
        for s in spec_args:
            backbone, pattern, img = s.split(':')
            specs.append((backbone, pattern, int(img)))
        return specs
    # default: pipeline nuevo
    return [(bb, f'best_{lab}_fold{{k}}.pth', IMG_SIZE) for bb, lab in BACKBONES]


def resolve_thresholds(mode, mean_p1, mean_p2, t1d, t2d, oof_labels, keep_t2=False):
    """Devuelve umbrales por clase según el modo (corrige covariate shift train->test).

    - calibrated: umbrales OOF tal cual (asume misma escala de probabilidad; falla si hay shift).
    - transport : mapea el punto de operación de OOF al espacio de probabilidad del TEST
                  (misma fracción de positivos que en OOF). Neutraliza shift monótono.
    - prior     : fija la fracción de positivos del TEST = prior real de train por clase.
    """
    if mode == 'calibrated':
        return t1d, t2d
    oof_p1, oof_p2, gt, _ = load_oof(oof_labels)
    t1_eff, t2_eff = {}, {}
    for k, col in enumerate(ARTIFACT_COLS):
        if mode == 'transport':
            r1 = float((oof_p1[:, k] >= t1d.get(col, 0.5)).mean())
            r2 = float((oof_p2[:, k] >= t2d.get(col, 0.5)).mean())
        else:  # prior
            r1 = float((gt[:, k] >= 1).mean())
            r2 = float((gt[:, k] >= 2).mean())
        t1_eff[col] = float(np.quantile(mean_p1[:, k], 1 - r1)) if r1 > 0 else 1.01
        if keep_t2:
            t2_eff[col] = t2d.get(col, 0.5)   # nivel severo conservador (calibrado)
        else:
            t2_eff[col] = float(np.quantile(mean_p2[:, k], 1 - r2)) if r2 > 0 else 1.01
    print(f"  Umbrales modo '{mode}' (transportados al test):")
    for col in ARTIFACT_COLS:
        print(f"    {col:13s} t1={t1_eff[col]:.3f} t2={t2_eff[col]:.3f}")
    return t1_eff, t2_eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', action='append', dest='specs',
                    help='backbone:patron_ckpt:img_size  (repetible)')
    ap.add_argument('--threshold-mode', choices=['calibrated', 'transport', 'prior'],
                    default='calibrated', help='corrección de covariate shift train->test')
    ap.add_argument('--oof-labels', nargs='+', default=[lab for _, lab in BACKBONES],
                    help='labels OOF para transport/prior (default: backbones de config)')
    ap.add_argument('--out', default=SUBMISSION_FILE, help='nombre del CSV de salida')
    ap.add_argument('--keep-t2', action='store_true',
                    help='mantener t2 calibrado (severo conservador) en transport/prior')
    args = ap.parse_args()
    specs = parse_specs(args.specs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    files = collect_lf_files(VAL_SINGLE_DIR) + collect_lf_files(VAL_COMPLETE_DIR)
    print(f"Archivos LF: {len(files)} (esperado: {N_SUBMISSION_ROWS})")

    t1d, t2d = load_thresholds()

    acc_p1, acc_p2, n_models = [], [], 0
    for backbone, pattern, img in specs:
        ds = SubmissionDataset(files, val_transform(img))
        for k in range(N_FOLDS):
            ckpt = CHECKPOINTS_DIR / pattern.format(k=k)
            if not ckpt.exists():
                continue
            model, _ = load_model(backbone, ckpt, device)
            p1, p2 = predict_probs(model, ds, device, tta=True)
            acc_p1.append(p1); acc_p2.append(p2)
            n_models += 1
            print(f"  {backbone:18s} fold{k}: OK")
    if n_models == 0:
        print("[ERROR] ningún checkpoint encontrado para los specs dados.")
        sys.exit(1)
    print(f"Modelos en el ensemble: {n_models}")

    mean_p1 = np.mean(acc_p1, axis=0)
    mean_p2 = np.mean(acc_p2, axis=0)

    t1_eff, t2_eff = resolve_thresholds(args.threshold_mode, mean_p1, mean_p2,
                                        t1d, t2d, args.oof_labels, keep_t2=args.keep_t2)

    rows = []
    for i, f in enumerate(files):
        row = {'patient_id': extract_patient_id(f.name)}
        for k, col in enumerate(ARTIFACT_COLS):
            score = 2 if mean_p2[i, k] >= t2_eff.get(col, 0.5) else (1 if mean_p1[i, k] >= t1_eff.get(col, 0.5) else 0)
            row[col] = score
        rows.append(row)

    df_sub = pd.DataFrame(rows)[['patient_id'] + ARTIFACT_COLS]
    out_path = ROOT / args.out
    df_sub.to_csv(out_path, index=False)

    print(f"\nSubmission: {out_path}  ({len(df_sub)} filas)")
    print(f"\n  {'Artefacto':15s}  s=0  s=1  s=2")
    for col in ARTIFACT_COLS:
        vc = df_sub[col].value_counts()
        print(f"  {col:15s}  {int(vc.get(0,0)):3d}  {int(vc.get(1,0)):3d}  {int(vc.get(2,0)):3d}")
    if len(df_sub) != N_SUBMISSION_ROWS:
        print(f"\n[AVISO] {len(df_sub)} filas != {N_SUBMISSION_ROWS} esperadas.")


if __name__ == '__main__':
    main()
