# Task 2 — Subcortical Segmentation (nnU-Net)

3D segmentation of 11 annotated subcortical structures on 0.064 T pediatric
CISO volumes. Residual-encoder nnU-Net v2 with stock components, plus a
structure-specific post-processing step.

## Prerequisites

- `nnunetv2` installed (`pip install nnunetv2`)
- nnU-Net environment variables configured — see `setup_env.ps1`:
  ```powershell
  .\setup_env.ps1
  ```
- `data/` populated with the training cohort (see `docs/DATA_STRUCTURE.md`)

## Execution order

```bash
# 1. Verification and conversion to nnU-Net format
python verify_labels.py
python prepare_nnunet.py
python verify_dataset.py

# 2. Preprocessing and training
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerResEncL
nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerBoundaryLoss   # boundary-loss variant

# 3. Inference and post-processing
python predict_task2.py
python postprocess_predictions.py

# 4. Evaluation
python evaluate_task2.py --trainer ResEncL
python evaluate_task2.py --trainer BoundaryLoss
```

## Why the post-processing step

The challenge ranking weighs Hausdorff Distance as heavily as DSC: a single
floating connected component 50 mm from the true structure can wreck a
case's HD even when DSC is 0.85. `postprocess_predictions.py` keeps the
largest connected component per label (1–11) and fills small internal
holes — applied per structure, not with a uniform threshold.

## Modules

| File | Role |
|---|---|
| `verify_labels.py` | validates the 1–11 label → structure mapping (volume/symmetry inference + Synapse wiki `syn72118611`) |
| `prepare_nnunet.py` | converts NIfTI data into the nnU-Net dataset layout |
| `verify_dataset.py` | integrity check before `plan_and_preprocess` |
| `predict_task2.py` | runs inference with the available fold(s) on the validation set |
| `postprocess_predictions.py` | per-structure connected-component filtering + hole filling |
| `evaluate_task2.py` | DSC, HD, HD95, ASSD, RVE per structure and overall |
| `inspect_checkpoint.py` | inspects a checkpoint (epoch, EMA Dice, architecture, hash) |
| `setup_env.ps1` | configures `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` |
