# LISA Challenge 2026

Code for a team's participation in the [LISA Challenge 2026](https://zenodo.org/records/15081583) —
quality control, image-quality enhancement and segmentation on ultra-low-field
(0.064 T) pediatric brain MRI.

## Tasks

| Task | Description | Code |
|---|---|---|
| **1A** | Ordinal (0/1/2) grading of 7 artifact classes per acquisition plane | [`task_1a/`](task_1a/) |
| **1B** | Low-field image-quality enhancement toward a high-field reference | [`task_1b/`](task_1b/) |
| **2** | 3D segmentation of 11 annotated subcortical structures | [`task_2/`](task_2/) |

## Results (summary)

- **Task 1A**: micro-accuracy **0.839** (leakage-free out-of-fold, ten-model ensemble)
- **Task 1B**: a well-normalised native pass-through (no learned model) beats all four trained enhancers on the validation leaderboard
- **Task 2**: mean Dice **0.784** over eleven structures (residual-encoder nnU-Net, five-fold ensemble)

Full methodology, ablations and diagnosis of each negative result are in the
paper (citation below).

## Repository structure

```
lisa-challenge2026/
├── task_1a/                 # ordinal artifact classification
├── task_1b/                 # image-quality enhancement
└── task_2/                  # subcortical segmentation (nnU-Net v2)
```

Each task folder has its own `README.md` with the exact execution order and
module reference.

> **Note:** `data/`, `nnunet_workspace/`, trained checkpoints (`*.pth`) and
> submission predictions/zips are not in this repository, due to size and,
> for `data/`, the Synapse data-use agreement. Results artifacts (OOF
> predictions, per-run metrics, figures) are likewise not tracked here — the
> headline numbers are reported above and the full results in the paper.

## Requirements

```bash
pip install nnunetv2
pip install torch torchvision
pip install nibabel numpy scikit-learn pandas tqdm
pip install timm                  # EfficientNet-B4 / ConvNeXt pretrained weights
```

Minimal checks before training:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # -> True
nvidia-smi                                                     # -> VRAM >= 16 GB
nnUNetv2_train --help                                          # -> no error
```

## Data

Data is downloaded from [Synapse](https://www.synapse.org/) under the
challenge's data-use agreement (`syn75277286`) and must be placed under
`data/`. It is not redistributed in this repository.

## Quick start

### Task 1A
```bash
cd task_1a/
python run_all.py
```
See [`task_1a/README.md`](task_1a/README.md) for the detailed order, the
challenge metric, and the validation lessons that shaped the current design.

### Task 1B
See [`task_1b/README.md`](task_1b/README.md) — covers the completed v1
pipeline and every variant evaluated (adversarial, registered + perceptual,
GAN) that the paper documents as beaten by the native pass-through.

### Task 2
```bash
cd task_2/
python verify_dataset.py
python prepare_nnunet.py
python verify_labels.py

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerResEncL
```
See [`task_2/README.md`](task_2/README.md) for the full pipeline including
post-processing and evaluation.

## Citation

```bibtex
@inproceedings{marca2026rigor,
  title     = {Rigor over Novelty in Ultra-Low-Field Pediatric Brain MRI: Quality
               Control and Subcortical Segmentation for the LISA Challenge 2026},
  author    = {Marca, Ronald and Guerra, Gabriel and Ortiz-Puerta, David and
               Chabert, Steren and Salas, Rodrigo},
  booktitle = {MICCAI 2026 Workshop on Low-field pediatric brain magnetic
               resonance Image Segmentation and quality Assurance (LISA)},
  year      = {2026}
}
```

DOI and page numbers will be added once the proceedings are published.

## Team

Ronald Marca¹²³ · Gabriel Guerra¹³ · David Ortiz-Puerta²³⁴ · Steren Chabert²³⁴ · Rodrigo Salas²³⁴

¹ PhD Program in Health Sciences and Engineering, Universidad de Valparaíso
² School of Biomedical Engineering, Universidad de Valparaíso
³ Millennium Institute for Intelligent Healthcare Engineering (iHEALTH)
⁴ Center for Interdisciplinary Biomedical and Engineering Research for Health (MEDING)

## References

- [LISA Challenge 2026 — Zenodo](https://zenodo.org/records/15081583)
- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)
- [nnU-Net Revisited — ResEnc (arXiv 2404.09556)](https://arxiv.org/abs/2404.09556)
- [LISA baseline — SFNet + nnU-Net](https://link.springer.com/chapter/10.1007/978-3-031-83008-2_6)
