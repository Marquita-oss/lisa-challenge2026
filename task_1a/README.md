# Task 1A — Quality Assessment (clean pipeline)

Ordinal (0/1/2) grading of 7 artifact classes on ultra-low-field MRI.
Single, reproducible pipeline with honest k-fold validation.

## The metric that matters

The leaderboard ranks with `average='micro'` over the flattened 0/1/2
ordinal grid (≈ **flattened accuracy**). That's why in `task1a-resultados.csv`
the 5 `_micro` columns are identical per row. **Do not optimise macro or
AUC** — it sabotages the ranking. The real metric lives in
`metrics.py::challenge_score`.

## Lessons learned (why the submission scored 0.716 with OOF 0.81)

1. **Leaky calibration** — thresholds tuned and evaluated on the same small
   val set overfit; on test they dropped ~0.05–0.09. Now calibration runs
   on OOF with an anti-overfit rule (`calibrate.py`).
2. **ColorJitter/GaussianBlur** in augmentation corrupted MRI intensity
   (which is the Contrast/Noise/Banding signal). Augmentation is now
   geometric-only.
3. **Optimistic validation** — no real OOF. OOF now matches the leaderboard.
4. **Unstratified split** — Banding (4%) was imbalanced across folds. Now
   `StratifiedGroupKFold` by `case_id`.

## Execution order

```bash
# Environment: conda lisa2026
python train.py --label effb4     --backbone efficientnet_b4   # 5 folds
python train.py --label convnexts --backbone convnext_small    # 5 folds
python calibrate.py --labels effb4 convnexts                   # OOF thresholds (leakage-free)
python predict.py --spec efficientnet_b4:best_effb4_fold{k}.pth:256 \
                  --spec convnext_small:best_convnexts_fold{k}.pth:256
# or all at once:
python run_all.py
```

Output: `LISA_LF_QC_predictions.csv` (114 rows) in the project root → upload
to Synapse.

## Re-evaluate OOF from existing checkpoints (no retraining)

```bash
python oof.py --label effb4 --backbone efficientnet_b4 \
              --ckpt-pattern "best_1a_v9_fold{k}.pth" --split legacy
python calibrate.py --labels effb4
python predict.py --spec "efficientnet_b4:best_1a_v9_fold{k}.pth:256"
```

## Modules

| File | Role |
|---|---|
| `config.py` | paths, backbones, hyperparameters, calibration grids |
| `data.py` | k-fold splits (legacy + stratified), 3D Dataset, sampler |
| `model.py` | `OrdinalClassifier(backbone)` dual-head |
| `losses.py` | `OrdinalFocalLoss` + monotonicity penalty |
| `metrics.py` | `challenge_score` (flattened micro/accuracy — the real one) |
| `inference.py` | model loading, TTA×8, prediction |
| `train.py` | trains one backbone k-fold, saves checkpoints + OOF |
| `oof.py` | generates OOF from checkpoints (Phase 0 / re-evaluation) |
| `calibrate.py` | per-class thresholds over OOF, anti-overfit |
| `predict.py` | 114-row submission from ensemble + TTA |
| `run_all.py` | end-to-end orchestrator |

The legacy pipeline (v3..v10) lives in `archive/legacy/`.
