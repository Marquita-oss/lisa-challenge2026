"""
metrics.py — La métrica REAL del challenge Task 1A.

El leaderboard rankea con average='micro' sobre la grilla ordinal aplanada 0/1/2.
En multiclase micro: F1=F2=Precision=Recall=Accuracy (por eso en task1a-resultados.csv
las 5 columnas _micro son idénticas por fila). Equivale a accuracy aplanada.

`challenge_score` aquí = exactamente lo que mide el grader. NO usar macro/AUC para
selección de modelo ni de umbrales.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, fbeta_score, precision_score, recall_score,
)


def scores_from_probs(p1: np.ndarray, p2: np.ndarray,
                      t1, t2, cols) -> np.ndarray:
    """Convierte (p_ge1, p_ge2) -> matriz de scores 0/1/2 con umbrales por clase.

    t1/t2 pueden ser escalares o dict {col: thr}. Reglas ordinales:
      score=2 si p2>=t2 ; score=1 si p1>=t1 ; else 0.
    """
    pred = np.zeros_like(p1, dtype=int)
    for k, col in enumerate(cols):
        a = t1[col] if isinstance(t1, dict) else t1
        b = t2[col] if isinstance(t2, dict) else t2
        pred[:, k][p1[:, k] >= a] = 1
        pred[:, k][p2[:, k] >= b] = 2
    return pred


def challenge_score(y_true_ord: np.ndarray, y_pred_ord: np.ndarray) -> dict:
    """Métrica del challenge sobre matrices de scores ordinales (N, C) en {0,1,2}.

    Devuelve el dict con las 5 métricas micro (idénticas) + f1_macro informativo +
    `score` = la cifra de ranking del leaderboard.
    """
    gt   = y_true_ord.flatten()
    pred = y_pred_ord.flatten()
    f1  = f1_score(gt, pred, average='micro', zero_division=0)
    f2  = fbeta_score(gt, pred, beta=2, average='micro', zero_division=0)
    pre = precision_score(gt, pred, average='micro', zero_division=0)
    rec = recall_score(gt, pred, average='micro', zero_division=0)
    acc = accuracy_score(gt, pred)
    f1m = f1_score(gt, pred, average='macro', zero_division=0)
    return {
        'f1_micro': float(f1), 'f2_micro': float(f2),
        'precision_micro': float(pre), 'recall_micro': float(rec),
        'accuracy_micro': float(acc), 'f1_macro': float(f1m),
        'score': float(np.mean([f1, f2, pre, rec, acc])),
    }


def score_from_probs(p1, p2, gt_ord, t1, t2, cols) -> dict:
    """Atajo: aplica umbrales a probabilidades y evalúa contra gt ordinal."""
    pred = scores_from_probs(p1, p2, t1, t2, cols)
    return challenge_score(gt_ord, pred)
