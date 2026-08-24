"""
calibrate.py — Umbrales por clase sobre OOF, SIN fuga.

Clave: el score del challenge (micro = accuracy aplanada sobre celdas 0/1/2) se
descompone aditivamente por clase. Por tanto, optimizar la accuracy por clase es
EXACTAMENTE óptimo para el score global. Sin acoplar clases, sin fuga.

Regla anti-overfit: sólo se acepta el umbral calibrado de una clase si mejora la
accuracy de esa clase en >= CALIB_MIN_FOLDS_IMPROVE de los N_FOLDS; si no, se deja 0.5.
Esto evita repetir el error previo (umbrales que ganaban en val pero caían en test).

Uso:
  python calibrate.py --labels effb4                 # un backbone
  python calibrate.py --labels effb4 convnexts       # ensemble de backbones

Entrada: results/oof_{label}.npz   Salida: results/thresholds.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, ARTIFACT_COLS, T1_GRID, T2_GRID, CALIB_MIN_FOLDS_IMPROVE
from metrics import challenge_score, scores_from_probs


def load_oof(labels):
    """Carga y promedia OOF de varios backbones (ensemble). gt/fold deben coincidir."""
    p1s, p2s, gt, fold = [], [], None, None
    for lab in labels:
        d = np.load(RESULTS_DIR / f'oof_{lab}.npz')
        p1s.append(d['p1']); p2s.append(d['p2'])
        if gt is None:
            gt, fold = d['gt'], d['fold']
        else:
            assert np.array_equal(gt, d['gt']), "gt OOF distinto entre backbones (¿split distinto?)"
            assert np.array_equal(fold, d['fold']), "fold OOF distinto entre backbones"
    return np.mean(p1s, axis=0), np.mean(p2s, axis=0), gt, fold


def _col_pred(p1c, p2c, t1, t2):
    pred = np.zeros(len(p1c), dtype=int)
    pred[p1c >= t1] = 1
    pred[p2c >= t2] = 2
    return pred


def _col_acc(p1c, p2c, gtc, t1, t2):
    return float((_col_pred(p1c, p2c, t1, t2) == gtc).mean())


def calibrate(p1, p2, gt, fold):
    folds = sorted(set(int(f) for f in fold))
    t1_best, t2_best = {}, {}
    for k, col in enumerate(ARTIFACT_COLS):
        p1c, p2c, gtc = p1[:, k], p2[:, k], gt[:, k]
        # mejor (t1,t2) global por accuracy de la clase
        best_acc, bt1, bt2 = -1.0, 0.5, 0.5
        for t1 in T1_GRID:
            for t2 in T2_GRID:
                acc = _col_acc(p1c, p2c, gtc, t1, t2)
                if acc > best_acc:
                    best_acc, bt1, bt2 = acc, t1, t2
        # regla anti-overfit: ¿mejora vs 0.5 en suficientes folds?
        improved = 0
        for f in folds:
            m = fold == f
            a_cal = _col_acc(p1c[m], p2c[m], gtc[m], bt1, bt2)
            a_def = _col_acc(p1c[m], p2c[m], gtc[m], 0.5, 0.5)
            if a_cal >= a_def:
                improved += 1
        if improved >= CALIB_MIN_FOLDS_IMPROVE:
            t1_best[col], t2_best[col] = bt1, bt2
            tag = f"calibrado (mejora {improved}/{len(folds)} folds)"
        else:
            t1_best[col], t2_best[col] = 0.5, 0.5
            tag = f"0.5/0.5 (sólo {improved}/{len(folds)} folds) -> se descarta"
        print(f"  {col:15s} t1={t1_best[col]:.2f} t2={t2_best[col]:.2f}  {tag}")
    return t1_best, t2_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', required=True)
    args = ap.parse_args()

    print(f"Cargando OOF: {args.labels}")
    p1, p2, gt, fold = load_oof(args.labels)

    base = challenge_score(gt, scores_from_probs(p1, p2, 0.5, 0.5, ARTIFACT_COLS))
    print(f"\nBaseline @0.5/0.5: score={base['score']:.4f}  (f1_macro={base['f1_macro']:.4f})")

    print("\nCalibrando por clase (anti-overfit)...")
    t1d, t2d = calibrate(p1, p2, gt, fold)

    cal = challenge_score(gt, scores_from_probs(p1, p2, t1d, t2d, ARTIFACT_COLS))
    print(f"\nCalibrado: score={cal['score']:.4f}  (f1_macro={cal['f1_macro']:.4f})")
    print(f"Ganancia honesta: {cal['score'] - base['score']:+.4f}")

    out = RESULTS_DIR / 'thresholds.json'
    out.write_text(json.dumps({
        'labels': args.labels,
        't1_per_class': t1d,
        't2_per_class': t2d,
        'baseline': base,
        'calibrated': cal,
    }, indent=2))
    print(f"\nGuardado: {out}")


if __name__ == '__main__':
    main()
