# Task 2 — Mejoras: Segmentación Multi-estructura 3D

> **Nota de contexto (2026-08-19):** este documento reporta solo `fold_0` (DSC 0.7931,
> BoundaryLoss) como resultado provisional y su "Mejora 1" pedía completar los folds
> 1–4. El paper final reporta el ensemble de **5 folds** (DSC medio 0.7849 sobre las
> 9 estructuras puntuadas — no directamente comparable a este 0.7931 de un solo fold,
> ya medido sobre 11 estructuras). **Aviso de reproducibilidad:** en esta máquina
> local (`nnunet_workspace/nnUNet_results/`) solo existen checkpoints de `fold_0`
> completo y `fold_1` sin `checkpoint_best.pth` para `nnUNetTrainerBoundaryLoss`, y solo
> `fold_0` para `nnUNetTrainerResEncL_BoundaryLoss` — los folds 2–4 que sostienen el
> número final del paper no están en este equipo. Según
> [`docs/DOCKER_SUBMISSION.md`](DOCKER_SUBMISSION.md), Task 2 lo armó y envió Gabriel
> por separado, así que es muy probable que los checkpoints/resultados de los folds
> restantes vivan en su máquina — hay que confirmarlo con él antes de armar el paquete
> de datos crudos para replicación, o el ensemble de 5 folds no se podrá reproducir
> completo desde este repositorio.

## Resultados actuales (fold_0, validación interna)

Dos modelos entrenados, ambos en `nnunet_workspace/nnUNet_results/Dataset001_LISA_Task2/`.

### Comparativa de modelos

| Modelo | DSC foreground | IoU | HD95 (mm) | ASSD (mm) | Estado |
|---|---|---|---|---|---|
| **nnUNetTrainerBoundaryLoss** | **0.7931** | **0.6725** | **2.09** | **0.8184** | **Mejor** |
| nnUNetTrainerResEncL_BoundaryLoss | 0.7914 | 0.6702 | 2.09 | 0.8186 | Muy similar |

**Fuente del benchmark:** LISA 2025.
- Baseline nnU-Net estándar: DSC = **0.61**
- nnU-Net + SFNet (super-resolución): DSC = **0.71**
- Fuente: *[Springer LNCS doi.org/10.1007/978-3-031-83008-2_6](https://link.springer.com/chapter/10.1007/978-3-031-83008-2_6)*

### Comparativa con benchmark

| Método | DSC | vs. baseline 0.61 | vs. SFNet 0.71 |
|---|---|---|---|
| Baseline nnU-Net 2025 | 0.610 | — | — |
| nnU-Net + SFNet 2025 | 0.710 | +0.10 | — |
| **BoundaryLoss (actual)** | **0.793** | **+0.183** | **+0.083** |
| ResEncL_BoundaryLoss (actual) | 0.791 | +0.181 | +0.081 |
| Objetivo WORK_PLAN (máximo) | 0.750 | — | — |

**Los modelos actuales ya superan el objetivo máximo del plan (+0.043).** El DSC 0.793 es una mejora del +30% sobre el baseline de LISA 2025 y +11.7% sobre el pipeline con super-resolución.

### Métricas por estructura — BoundaryLoss (mejor modelo)

| ID | Estructura | DSC | IoU | Observaciones |
|---|---|---|---|---|
| 1 | HippocampusL | 0.632 | 0.487 | **Más débil — estructura pequeña** |
| 2 | HippocampusR | 0.673 | 0.529 | Segunda más débil |
| 11 | CorpusCallosum | 0.738 | 0.596 | Tercera más débil — estructura elongada |
| 4 | VentricleR | 0.776 | 0.641 | Moderado |
| 3 | VentricleL | 0.799 | 0.673 | Moderado |
| 7 | LentiformL | 0.830 | 0.715 | Bueno |
| 5 | CaudateL | 0.820 | 0.699 | Bueno |
| 6 | CaudateR | 0.817 | 0.692 | Bueno |
| 8 | LentiformR | 0.843 | 0.731 | Bueno |
| 9 | ThalamusL | 0.895 | 0.811 | Excelente |
| 10 | ThalamusR | 0.902 | 0.822 | **Mejor estructura** |

### Diferencia entre modelos por estructura

| Estructura | BoundaryLoss | ResEncL | Δ | Ventaja |
|---|---|---|---|---|
| HippocampusL | 0.632 | **0.639** | +0.007 | ResEncL |
| HippocampusR | 0.673 | 0.663 | −0.010 | BoundaryLoss |
| VentricleL | **0.799** | 0.797 | −0.002 | BoundaryLoss ≈ |
| VentricleR | **0.776** | 0.773 | −0.003 | BoundaryLoss ≈ |
| CaudateL | **0.820** | 0.819 | −0.001 | Empate |
| CaudateR | 0.817 | **0.815** | −0.002 | BoundaryLoss ≈ |
| LentiformL | **0.830** | 0.827 | −0.003 | BoundaryLoss |
| LentiformR | **0.843** | 0.837 | −0.006 | BoundaryLoss |
| ThalamusL | **0.895** | 0.894 | −0.001 | Empate |
| ThalamusR | **0.902** | 0.902 | 0.000 | Empate |
| CorpusCallosum | 0.738 | **0.738** | 0.000 | Empate |

**Conclusión:** BoundaryLoss es el modelo preferido (mejor o igual en 9 de 11 estructuras). ResEncL es marginalemente mejor solo en HippocampusL (+0.007).

---

## Limitaciones actuales

### 1. Solo fold_0 entrenado

Los resultados actuales son de un único fold de validación (16 casos de 79). La comparativa con el benchmark de LISA 2025 (que usa 5-fold CV) no es directamente comparable. La varianza entre folds puede ser significativa con solo 79 casos.

### 2. Hipocampos son el talón de Aquiles

DSC = 0.632–0.673 para hipocampos. Estas son las estructuras más pequeñas del dataset y las que más variabilidad inter-caso tienen en MRI de bajo campo.

### 3. Corpus Callosum — estructura elongada

DSC = 0.738. Las estructuras elongadas son particularmente difíciles para nnU-Net estándar porque el patch size 3D puede no capturar la longitud completa del cuerpo calloso.

---

## Mejora 1 — Completar 5-fold cross-validation (impacto: resultado definitivo y comparable)

**Acción inmediata — no requiere cambios de código.**

```bash
# Folds pendientes (fold_0 ya está completo)
nnUNetv2_train 1 3d_fullres 1 -tr nnUNetTrainerBoundaryLoss
nnUNetv2_train 1 3d_fullres 2 -tr nnUNetTrainerBoundaryLoss
nnUNetv2_train 1 3d_fullres 3 -tr nnUNetTrainerBoundaryLoss
nnUNetv2_train 1 3d_fullres 4 -tr nnUNetTrainerBoundaryLoss

# Evaluación del ensemble 5-fold
nnUNetv2_find_best_configuration 1
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d 1 -c 3d_fullres \
    -tr nnUNetTrainerBoundaryLoss --save_probabilities
```

**Tiempo estimado:** ~14–16h por fold × 4 folds = ~56–64h GPU adicionales.

**Por qué es prioritario:** El DSC del ensemble 5-fold es el número que el challenge compara. Fold_0 solo es orientativo.

---

## Mejora 2 — Data Augmentation intensiva para hipocampos (impacto estimado: +0.03–0.05 DSC en hipocampos)

Los hipocampos son pequeños y con alta variabilidad. Aumentar la representación en el entrenamiento con augmentaciones específicas.

### Opción A — Oversample de casos con hipocampos difíciles (dentro de nnU-Net)

Crear un trainer personalizado que aumenta la frecuencia de muestreo de los 5 casos con peor DSC en hipocampos.

```python
# task_2/trainers/nnUNetTrainerBoundaryLossHippoOversample.py
# Colocar en: C:\Users\rmarcar\miniconda3\envs\lisa2026\Lib\site-packages\nnunetv2\training\nnUNetTrainer\variants\

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerBoundaryLoss import nnUNetTrainerBoundaryLoss


class nnUNetTrainerBoundaryLossHippoOversample(nnUNetTrainerBoundaryLoss):
    """
    Igual que BoundaryLoss pero con oversample de casos donde los hipocampos
    son más difíciles (determinado por análisis del fold_0).
    """

    # Casos con DSC hipocampo < 0.50 en fold_0 (IDs observados en summary.json)
    HARD_CASES = ['lisa_0005', 'lisa_0021', 'lisa_0041']

    def get_tr_and_val_datasets(self):
        tr, val = super().get_tr_and_val_datasets()
        # Triplicar la aparición de casos difíciles en train
        extra = [case for case in tr.case_identifiers
                 if any(hc in case for hc in self.HARD_CASES)]
        tr.case_identifiers = tr.case_identifiers + extra * 2
        return tr, val
```

```bash
# Uso
nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerBoundaryLossHippoOversample
```

### Opción B — Elastic deformation más agresiva en patch sampling

Modificar los parámetros de augmentación en `nnUNet_preprocessed/Dataset001_LISA_Task2/nnUNetPlans.json`:

```json
{
  "data_identifier": "nnUNetPlans_3d_fullres",
  "preprocessor_name": "DefaultPreprocessor",
  "batch_size": 2,
  "patch_size": [112, 160, 128],
  "median_image_size_in_voxels": ...,
  "spacing": ...,
  "normalization_schemes": [...],
  "use_mask_for_norm": [...],
  "UNet_class_name": "PlainConvUNet",
  "UNet_base_num_features": 32,
  "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
  "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
  "num_pool_per_axis": [5, 5, 5],
  "pool_op_kernel_sizes": [...],
  "conv_kernel_sizes": [...],
  "unet_max_num_filters": 320,
  "resampling_fn_data": "resample_data_or_seg_to_shape",
  "resampling_fn_seg": "resample_data_or_seg_to_shape",
  "resampling_fn_data_kwargs": {...},
  "resampling_fn_seg_kwargs": {...},
  "resampling_fn_probabilities_kwargs": {...},
  "batch_dice": false
}
```

---

## Mejora 3 — Ensemble BoundaryLoss + ResEncL_BoundaryLoss (impacto estimado: +0.01–0.03 DSC)

Promediar las probabilidades de ambos modelos. El ensemble mejora la robustez, especialmente en los casos donde uno de los modelos es débil.

```bash
# Generar predicciones con probabilidades de ambos modelos
nnUNetv2_predict -i INPUT_DIR -o preds_boundary -d 1 -c 3d_fullres \
    -tr nnUNetTrainerBoundaryLoss --save_probabilities

nnUNetv2_predict -i INPUT_DIR -o preds_resencl -d 1 -c 3d_fullres \
    -tr nnUNetTrainerResEncL_BoundaryLoss --save_probabilities

# Ensemblar
nnUNetv2_ensemble -i preds_boundary preds_resencl -o preds_ensemble -np 4
```

**Prerequisito:** Completar 5-fold de ResEncL_BoundaryLoss también (actualmente solo fold_0).

---

## Mejora 4 — Integración con Task 1B (pipeline Task 1b → Task 2)

Aplicar el denoiser de Task 1B sobre las imágenes CISO antes de la segmentación. En LISA 2025 este pipeline (SFNet + nnU-Net) mejoró DSC de 0.61 → 0.71. Con nuestro denoiser (PSNR +12.84 dB) el impacto podría ser mayor.

### Pipeline de integración

```python
# task_2/apply_task1b_preprocessing.py

import os
import nibabel as nib
import torch
import numpy as np
from task_1b.model import ResUNet
from task_1b.dataset import InferenceDataset


def enhance_ciso_volume(nii_path: str, output_path: str,
                         ckpt: str = 'task_1b/checkpoints/best_1b.pth',
                         img_size: int = 256):
    """
    Aplica el denoiser de Task 1B sobre una imagen CISO 3D slice por slice.
    """
    model = ResUNet().cuda()
    model.load_state_dict(torch.load(ckpt, weights_only=False))
    model.eval()

    nii = nib.load(nii_path)
    vol = nii.get_fdata().astype(np.float32)

    # Normalizar por percentil
    p1, p99 = np.percentile(vol, [1, 99])
    vol_norm = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Procesar slice por slice en el eje más delgado
    thin_axis = np.argmin(vol.shape)
    enhanced = np.zeros_like(vol_norm)

    for i in range(vol.shape[thin_axis]):
        slc = np.take(vol_norm, i, axis=thin_axis)
        # Resize a img_size
        from PIL import Image
        slc_pil = Image.fromarray((slc * 255).astype(np.uint8))
        slc_resized = slc_pil.resize((img_size, img_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(slc_resized) / 255.0).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0).cuda()  # (1,1,H,W)

        with torch.no_grad():
            output = model(tensor)
        output_np = output.squeeze().cpu().numpy()

        # Resize de vuelta a la resolución original
        h, w = slc.shape
        output_pil = Image.fromarray((np.clip(output_np, 0, 1) * 255).astype(np.uint8))
        output_resized = np.array(output_pil.resize((w, h), Image.BILINEAR)) / 255.0

        if thin_axis == 0:
            enhanced[i, :, :] = output_resized
        elif thin_axis == 1:
            enhanced[:, i, :] = output_resized
        else:
            enhanced[:, :, i] = output_resized

    # Desnormalizar y guardar
    enhanced_orig = enhanced * (p99 - p1) + p1
    nib.save(nib.Nifti1Image(enhanced_orig, nii.affine, nii.header), output_path)


if __name__ == '__main__':
    input_dir  = 'data/train'
    output_dir = 'data/train_enhanced'
    os.makedirs(output_dir, exist_ok=True)
    for case in os.listdir(input_dir):
        ciso = os.path.join(input_dir, case, f'lisa_{case}_ciso.nii.gz')
        if os.path.exists(ciso):
            out = os.path.join(output_dir, case, f'lisa_{case}_ciso.nii.gz')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            enhance_ciso_volume(ciso, out)
            print(f'Enhanced: {case}')
```

### Reentrenamiento con imágenes mejoradas

```bash
# 1. Aplicar Task 1B sobre todo el conjunto de entrenamiento
python task_2/apply_task1b_preprocessing.py

# 2. Preparar nuevo dataset nnU-Net con imágenes mejoradas
python task_2/prepare_nnunet.py --input_dir data/train_enhanced --dataset_id 2

# 3. Entrenar
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainerBoundaryLoss
```

**Criterio de éxito:** DSC Dataset002 (imágenes mejoradas) > DSC Dataset001 (originales) + 0.02.

---

## Mejora 5 — PRIMUS-B trainer (Nivel 3 del plan)

PRIMUS-B está disponible como trainer dentro de nnU-Net v2. Solo puede ejecutarse si ya se superó el Nivel 1 y el Nivel 2.

```bash
# Verificar disponibilidad
python -c "from nnunetv2.training.nnUNetTrainer.variants.nnUNet_Primus_B_Trainer import nnUNet_Primus_B_Trainer; print('OK')"

# Si está disponible
nnUNetv2_train 1 3d_fullres 0 -tr nnUNet_Primus_B_Trainer

# Criterio de paso: DSC > 0.793 + 0.010 = 0.803
```

**Cuándo activar:** Solo si el ensemble BoundaryLoss+ResEncL no supera 0.80 DSC.

---

## Roadmap de mejoras

| Prioridad | Mejora | Tiempo GPU | DSC esperado | Prerequisito |
|---|---|---|---|---|
| 1 | 5-fold CV (folds 1–4) BoundaryLoss | ~60h | Resultado definitivo | Ninguno |
| 2 | Ensemble BoundaryLoss + ResEncL | ~60h (ResEncL folds 1–4) + 2h | +0.01–0.03 | Folds 1–4 de ambos |
| 3 | Integración Task 1B → Task 2 | ~6h preprocesado + ~16h/fold | +0.02–0.05 | Task 1B v2 (mejora adversarial) |
| 4 | Oversample hipocampos | ~14h (fold 0) | +0.03–0.05 en hipocampos | Análisis fold_0 |
| 5 | PRIMUS-B | ~16h/fold | Potencial +0.01–0.02 | Folds 1–4 completados |

**DSC objetivo final:** ≥ 0.82 con ensemble 5-fold + integración Task 1B.

### Orden de ejecución recomendado

```
[HOY]       Lanzar nnUNetTrainerBoundaryLoss folds 1,2,3,4 en paralelo
[+3 días]   Evaluar 5-fold mean DSC; decidir si ensemble con ResEncL
[+1 semana] Reentrenar con imágenes mejoradas de Task 1B (si v2 Task 1B disponible)
[+10 días]  Generar submission final con mejor configuración
```

---

## Comandos de evaluación y submission

```bash
# Evaluar fold_0 actual (resumen por estructura)
nnUNetv2_evaluate_folder \
    nnunet_workspace/nnUNet_preprocessed/Dataset001_LISA_Task2/gt_segmentations \
    nnunet_workspace/nnUNet_results/Dataset001_LISA_Task2/nnUNetTrainerBoundaryLoss__nnUNetPlans__3d_fullres/fold_0/validation \
    -djfile nnunet_workspace/nnUNet_preprocessed/Dataset001_LISA_Task2/dataset.json \
    -pfile nnunet_workspace/nnUNet_preprocessed/Dataset001_LISA_Task2/nnUNetPlans.json

# Predicción sobre val/complete (submission Task 2)
nnUNetv2_predict \
    -i data/val/complete \
    -o predictions_val_task2 \
    -d 1 \
    -c 3d_fullres \
    -tr nnUNetTrainerBoundaryLoss \
    -f 0 1 2 3 4 \  # usar todos los folds cuando estén disponibles
    --save_probabilities

# Post-procesado
python task_2/postprocess_predictions.py \
    --input_dir predictions_val_task2 \
    --output_dir predictions_val_task2_postproc
```

---

## Archivos del workspace

```
nnunet_workspace/
├── nnUNet_raw/Dataset001_LISA_Task2/      # 79 casos de entrenamiento
├── nnUNet_preprocessed/Dataset001_LISA_Task2/
│   ├── gt_segmentations/                  # Masks de referencia
│   └── nnUNetPlans.json                   # Configuración automática nnU-Net
└── nnUNet_results/Dataset001_LISA_Task2/
    ├── nnUNetTrainerBoundaryLoss__nnUNetPlans__3d_fullres/
    │   └── fold_0/
    │       ├── checkpoint_final.pth       # 247 MB
    │       ├── checkpoint_best.pth        # 247 MB — DSC 0.7931
    │       └── validation/summary.json   # Métricas por estructura
    └── nnUNetTrainerResEncL_BoundaryLoss__nnUNetPlans__3d_fullres/
        └── fold_0/
            ├── checkpoint_final.pth       # 236 MB
            ├── checkpoint_best.pth        # 236 MB — DSC 0.7914
            └── validation/summary.json
```
