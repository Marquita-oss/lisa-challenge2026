"""
task1a_oof_analysis.py — Números nuevos de Task 1A pedidos por los revisores LISA 2026.

Todo sale de las predicciones out-of-fold ya guardadas (sin reentrenar, sin tocar test):
  - Revisor 1: recall de casos positivos por plano de adquisición (axial/coronal/sagital),
    global y por clase de artefacto, para ver si el muestreo P25/P50/P75 captura de forma
    despareja los artefactos localizados.
  - Revisor 2: variación entre folds y detalle por clase (recall de positivos, QWK).

Uso:  python task1a_oof_analysis.py [--out task1a_oof_analysis.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLS = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TASK1A = REPO / 'task_1a'
# Las etiquetas vienen del challenge (acuerdo de uso de Synapse) y no se
# redistribuyen aquí: se esperan bajo data/ o se pasan con --labels.
CSV = REPO / 'data' / 'metadata' / 'lisa_task1a_2026.csv'


def qwk(y_true, y_pred, n_cls=3):
    """Kappa cuadrático ponderado sobre {0,1,2}."""
    O = np.zeros((n_cls, n_cls))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    W = np.array([[(i - j) ** 2 for j in range(n_cls)] for i in range(n_cls)], dtype=float)
    W /= (n_cls - 1) ** 2
    hist_t = np.bincount(y_true, minlength=n_cls)
    hist_p = np.bincount(y_pred, minlength=n_cls)
    E = np.outer(hist_t, hist_p).astype(float)
    E *= O.sum() / E.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=str(CSV),
                    help='CSV de etiquetas del challenge (data/metadata/lisa_task1a_2026.csv)')
    ap.add_argument('--out', default=str(HERE / 'task1a_oof_analysis.json'))
    args = ap.parse_args()

    labels = Path(args.labels)
    if not labels.exists():
        raise SystemExit(
            f"No encuentro el CSV de etiquetas en {labels}.\n"
            "Descárgalo de Synapse bajo el acuerdo de uso del challenge y colócalo en "
            "data/metadata/, o indica su ruta con --labels.")

    df = pd.read_csv(labels)
    plane = df.filename.str.extract(r'(?i)_(axi|cor|sag)\.')[0].str.lower().values

    e = np.load(TASK1A / 'results' / 'oof_effb4.npz')
    c = np.load(TASK1A / 'results' / 'oof_convnexts.npz')
    gt, fold = e['gt'], e['fold']
    assert np.array_equal(gt, (df[CLS].values >= 1).astype(int) + (df[CLS].values >= 2).astype(int)), \
        'el OOF guardado no está alineado fila a fila con el CSV de etiquetas'

    th = json.loads((TASK1A / 'results' / 'thresholds.json').read_text())
    t1 = np.array([th['t1_per_class'][k] for k in CLS])
    t2 = np.array([th['t2_per_class'][k] for k in CLS])

    out = {'protocol': 'OOF 5-fold StratifiedGroupKFold, ensemble 2 backbones, sin TTA '
                       '(el TTA x8 se aplica en el pipeline de submission, no en esta estimación)',
           'n_images': int(len(df)), 'thresholds': {'t1': t1.tolist(), 't2': t2.tolist()}}

    for tag, (p1, p2) in {
        'effb4': (e['p1'], e['p2']),
        'convnexts': (c['p1'], c['p2']),
        'ensemble': ((e['p1'] + c['p1']) / 2, (e['p2'] + c['p2']) / 2),
    }.items():
        blk = {}
        for thr_name, (a, b) in {'at_0.5': (0.5, 0.5), 'calibrated': (t1, t2)}.items():
            pred = np.where(p2 >= b, 2, np.where(p1 >= a, 1, 0))
            blk[thr_name] = {'micro_acc': float((pred == gt).mean())}
        out[tag] = blk

    p1 = (e['p1'] + c['p1']) / 2
    p2 = (e['p2'] + c['p2']) / 2
    pred = np.where(p2 >= t2, 2, np.where(p1 >= t1, 1, 0))
    pos = gt >= 1

    out['overall'] = {
        'micro_acc': float((pred == gt).mean()),
        'n_pos_cells': int(pos.sum()),
        'recall_pos': float((pred[pos] >= 1).mean()),
        'exact_grade_on_pos': float((pred[pos] == gt[pos]).mean()),
    }

    # --- Revisor 2: variación entre folds -----------------------------------
    per_fold = [float((pred[fold == k] == gt[fold == k]).mean()) for k in range(int(fold.max()) + 1)]
    out['per_fold'] = {
        'micro_acc': [round(v, 4) for v in per_fold],
        'mean': round(float(np.mean(per_fold)), 4),
        'std': round(float(np.std(per_fold, ddof=1)), 4),
        'min': round(float(np.min(per_fold)), 4),
        'max': round(float(np.max(per_fold)), 4),
    }

    # --- Revisor 1: por plano de adquisición --------------------------------
    out['per_plane'] = {}
    for pl in ('axi', 'cor', 'sag'):
        m = plane == pl
        pm = gt[m] >= 1
        out['per_plane'][pl] = {
            'n_images': int(m.sum()),
            'n_pos_cells': int(pm.sum()),
            'micro_acc': round(float((pred[m] == gt[m]).mean()), 4),
            'recall_pos': round(float((pred[m][pm] >= 1).mean()), 4),
            'exact_grade_on_pos': round(float((pred[m][pm] == gt[m][pm]).mean()), 4),
        }

    out['per_plane_class'] = {}
    for j, k in enumerate(CLS):
        row = {}
        for pl in ('axi', 'cor', 'sag'):
            m = plane == pl
            pj = gt[m, j] >= 1
            row[pl] = {'n_pos': int(pj.sum()),
                       'recall_pos': round(float((pred[m, j][pj] >= 1).mean()), 4) if pj.sum() else None}
        row['all'] = {'n_pos': int((gt[:, j] >= 1).sum()),
                      'recall_pos': round(float((pred[gt[:, j] >= 1, j] >= 1).mean()), 4)}
        out['per_plane_class'][k] = row

    # --- Revisor 2: detalle por clase ---------------------------------------
    out['per_class'] = {}
    for j, k in enumerate(CLS):
        pj = gt[:, j] >= 1
        out['per_class'][k] = {
            'n1': int((gt[:, j] == 1).sum()), 'n2': int((gt[:, j] == 2).sum()),
            'acc': round(float((pred[:, j] == gt[:, j]).mean()), 4),
            'qwk': round(qwk(gt[:, j], pred[:, j]), 4),
            'recall_pos': round(float((pred[pj, j] >= 1).mean()), 4),
            't1': t1[j], 't2': t2[j],
        }

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ('overall', 'per_fold', 'per_plane')}, indent=2))
    print(f"\nGuardado: {args.out}")


if __name__ == '__main__':
    main()
