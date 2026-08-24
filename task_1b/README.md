# Task 1B — Quality Improvement (LF → CISO)

Image-quality enhancement from ultra-low-field (0.064 T) toward a
high-field reference (CISO), evaluated on the official leaderboard with
FID, LPIPS, PSNR, BRISQUE, CLIPIQA and FRD.

## Paper finding

A well-normalised native-resolution pass-through (no learned model) beats
all four trained enhancers on the validation leaderboard. This README
documents how to run each pipeline that was evaluated; the diagnosis of why
each one fails to beat the pass-through (self-inflicted preprocessing
damage, a cross-field registration ceiling, metric overfitting) is in the
paper.

## Prerequisites

- Task 1A checkpoint: `task_1a/checkpoints/best_1a.pth` (used as an
  adversarial loss in v2+)
- `data/train/` and `data/val/single_plane/` populated — see `docs/DATA_STRUCTURE.md`

## Pipelines evaluated

### v1 — ResUNet + synthetic Rician noise (completed)

```bash
python 01_explore_data.py
python 02_train.py
python 03_evaluate.py
python 04_predict_submission.py --zip
```

### v2 / v3 — physics-based k-space degradation + Task 1A adversarial loss

```bash
python 02_train_v2.py --init-from-v1      # fine-tune from v1, recommended
python 03_evaluate.py --ckpt best_1b_v2.pth
python 04_predict_submission.py --ckpt best_1b_v2.pth --zip
```

`02_train_v3.py` is the same recipe over a `.npy` cache
(`preprocess_to_npy.py`), GPU-bound and faster to iterate on.

### Dihedral TTA inference (in-distribution, no retraining)

```bash
python 04b_predict_tta.py --ckpt best_1b_v2.pth --zip
```

### Unpaired GAN (PatchGAN, avoids the registration problem)

```bash
python 02_train_gan.py
python 04_predict_submission.py --ckpt <gan_checkpoint> --zip
```

### Resize-based supervised LF→CISO (no real registration) — discarded

```bash
python 02_train_lf2ciso.py
python 04c_predict_lf2ciso.py
```

Target = CISO resampled by array-resize onto the LF plane grid, without any
real geometric registration. The paper discards this route: supervised
translation hallucinates texture.

### "Option 2" — registered LF→CISO (rigid + Mattes MI) + perceptual loss — evaluated and rejected

```bash
python 10_build_registered_pairs.py     # rigid+MI registration (validated: corr 0.36 -> 0.63)
python 11_train_lf2ciso_perceptual.py   # L1 + SSIM + VGG-perceptual on registered pairs
python 12_select_checkpoint.py          # honest checkpoint selection by held-out FID + BRISQUE
```

Rigid registration only reaches a cohort-mean correlation of ~0.5–0.6; the
resulting model still hallucinates relative to the native pass-through. See
the paper's registration section for the full diagnosis.

## Modules

| File | Role |
|---|---|
| `config.py` | paths, hyperparameters |
| `dataset.py` / `augmentations.py` | synthetic datasets and physics-based degradation (Gibbs, banding, zipper, bias field, ghosting) |
| `model.py` | denoiser/enhancer architecture |
| `losses.py` | combined L1 / SSIM / perceptual / adversarial losses |
| `gen_native.py` | generates the native pass-through baseline |
| `eval_fid.py` / `eval_fid_percase.py` | global and per-case FID vs. CISO |
| `eval_noref.py` / `eval_local.py` | no-reference metrics (BRISQUE, CLIPIQA) and local evaluation |
| `probe_registration.py` | validates rigid+MI registration quality |
| `preprocess_to_npy.py` / `pack_uint16.py` / `to_int16.py` | cache and format utilities |
| `sweep_ops.py` / `sweep_fid_ops.py` | sweep of post-processing operations over the pass-through |
