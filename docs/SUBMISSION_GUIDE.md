# LISA Challenge 2026 — Ejecucion y Submission

> **Nota de contexto (2026-08-19):** la secuencia de scripts numerados de Task 1a
> descrita abajo (`00_explore.py`...`05_predict_submission.py`) fue reemplazada por el
> pipeline actual en `task_1a/` (`data.py`, `train.py`, `oof.py`, `calibrate.py`,
> `predict.py`, `run_all.py`) — ver [`task_1a/README.md`](../task_1a/README.md) para el
> orden vigente. La sección de Task 1b sigue describiendo el flujo real; para la
> decisión final de submission de Task 1b (pass-through nativo, sin denoiser entrenado)
> ver [`docs/DOCKER_SUBMISSION.md`](DOCKER_SUBMISSION.md).

Guia de referencia para ejecutar los pipelines de Task 1a y Task 1b y generar
los archivos de submission con el formato exacto requerido por el sistema de
evaluacion de Synapse.

---

## Task 1a — Quality Assessment

### Prerequisitos

- `data/train/` poblado con casos de entrenamiento
- `data/val/single_plane/` poblado con los 72 casos ciegos (1001-1072)
- `data/metadata/lisa_task1a_2026.csv` presente

### Secuencia de ejecucion

Todos los comandos se ejecutan desde la raiz del proyecto.

```
python task_1a/00_explore.py
python task_1a/01_verify_setup.py
python task_1a/02_train.py
python task_1a/03_evaluate.py
python task_1a/04_calibrate_thresholds.py
python task_1a/05_predict_submission.py
```

Scripts opcionales (no necesarios para submission):
```
python task_1a/06_gradcam.py          # visualizacion Grad-CAM
python task_1a/07_paper_figures.py    # figuras para el paper
```

### Gates de entrada y salida

| Script | Gate de entrada | Output | Criterio para avanzar |
|---|---|---|---|
| `00_explore.py` | Ninguno | `results/00_exploration.json` | Siempre |
| `01_verify_setup.py` | `00_exploration.json` | `results/01_setup.json` | `ready_to_train = true` |
| `02_train.py` | `01_setup.json` con `ready_to_train = true` | `checkpoints/best_1a.pth` | Checkpoint guardado |
| `03_evaluate.py` | `checkpoints/best_1a.pth` | `results/03_evaluation.json` | Ver criterios abajo |
| `04_calibrate_thresholds.py` | `checkpoints/best_1a.pth` | `results/04_thresholds.json` | Siempre |
| `05_predict_submission.py` | `best_1a.pth` + `04_thresholds.json` | `LISA_LF_QC_predictions.csv` | Verificar 0 casos faltantes |

### Criterios de evaluacion interna

| Metrica | Criterio minimo |
|---|---|
| AUC promedio (7 clases) | >= 0.80 |
| Challenge score: mean(accuracy, F1, F2, precision, recall) | >= 0.70 |
| Peor AUC por clase | >= 0.65 |

Si el modelo no alcanza el criterio: ejecutar `04_calibrate_thresholds.py` antes
de re-entrenar. La calibracion de umbrales por F2 puede recuperar hasta 5 puntos
en el challenge score sin re-entrenar.

### Formato de submission Task 1a

El sistema de evaluacion de Synapse espera exactamente este archivo:

```
LISA_LF_QC_predictions.csv
```

Estructura del CSV:

| Columna | Descripcion | Valores |
|---|---|---|
| `patient_id` | ID del paciente — formato `LISA_LF_1234` | — |
| `Noise` | Ruido | 0 / 1 / 2 |
| `Zipper` | Artefacto zipper | 0 / 1 / 2 |
| `Positioning` | Mal posicionamiento | 0 / 1 / 2 |
| `Banding` | Artefacto de bandas | 0 / 1 / 2 |
| `Motion` | Movimiento | 0 / 1 / 2 |
| `Contrast` | Problema de contraste | 0 / 1 / 2 |
| `Distortion` | Distorsion geometrica | 0 / 1 / 2 |

Escala de severidad: 0 = sin artefacto, 1 = leve, 2 = severo.

El script `05_predict_submission.py` genera el archivo directamente en la raiz
del proyecto con el nombre correcto. Subir ese archivo a Synapse desde la
pestana Files del proyecto.

Metricas de evaluacion del challenge: F1-score, F2-score, precision, recall, accuracy.

---

## Task 1b — Quality Improvement

### Prerequisitos

- Task 1a completada: `task_1a/checkpoints/best_1a.pth` debe existir
- `data/train/` poblado
- `data/val/single_plane/` poblado con los 72 casos ciegos (1001-1072)
- Particiones CSV en `data/metadata/`:
  - `task_1b_nonoise_nomotion.csv`
  - `task_1b_nonoise_withmotion.csv`
  - `task_1b_withnoise_nomotion.csv`
  - `task_1b_withnoise_withmotion.csv`

### Secuencia de ejecucion

#### Pipeline v1 (ResUNet + Rician noise + ghosting)

```
python task_1b/01_explore_data.py
python task_1b/02_train.py
python task_1b/03_evaluate.py
python task_1b/04_predict_submission.py
python task_1b/04_predict_submission.py --zip    # genera LISA_enhanced_predictions.zip
```

#### Pipeline v2 (physics-based degradation + adversarial Task 1A loss)

Mejora sobre v1: degradacion fisica en k-space (Gibbs, banding, zipper, bias field)
mas loss adversarial con el clasificador de Task 1a congelado.

```
python task_1b/02_train_v2.py
python task_1b/02_train_v2.py --init-from-v1    # fine-tune desde v1, recomendado
python task_1b/03_evaluate.py --ckpt best_1b_v2.pth
python task_1b/04_predict_submission.py --ckpt best_1b_v2.pth --zip
```

### Gates de entrada y salida

| Script | Gate de entrada | Output | Criterio para avanzar |
|---|---|---|---|
| `01_explore_data.py` | Particiones CSV presentes | `results/01_exploration.json` | Siempre |
| `02_train.py` | `01_exploration.json` | `checkpoints/best_1b.pth` | val PSNR >= 28 dB |
| `03_evaluate.py` | `checkpoints/best_1b.pth` | `results/03_evaluation.json` | PSNR >= 28 dB, SSIM >= 0.80 |
| `04_predict_submission.py` | `best_1b.pth` | NIfTI en `submission/` + ZIP | Verificar 72 archivos |

### Criterios de evaluacion interna

| Metrica | Criterio minimo |
|---|---|
| PSNR | >= 28.0 dB |
| SSIM | >= 0.80 |

### Formato de submission Task 1b

El sistema de evaluacion de Synapse espera un unico archivo ZIP:

```
LISA_enhanced_predictions.zip
```

El ZIP debe contener exactamente un archivo `.nii.gz` por imagen mejorada, con
esta convencion de nombres:

```
LISA_VALIDATION_1234_axi_enhanced.nii.gz
LISA_VALIDATION_1234_cor_enhanced.nii.gz
LISA_VALIDATION_1234_sag_enhanced.nii.gz
```

Requisitos de cada archivo NIfTI:
- Formato `.nii.gz`
- Mismo shape, spacing y affine que la imagen de entrada correspondiente
- Un archivo por cada imagen del cohort de validacion (72 casos de single_plane)

El script `04_predict_submission.py --zip` genera los archivos con el nombre
correcto y empaqueta el ZIP listo para subir. Subir ese archivo a Synapse desde
la pestana Files del proyecto.

Metricas de evaluacion del challenge: FID, LPIPS, PSNR, BRISQUE, CLIPIQA, FRD.

---

## Problemas de nomenclatura en 04_predict_submission.py (Task 1b)

El script actual tiene dos incumplimientos respecto a la especificacion del challenge
que deben corregirse antes de generar la submission final:

| Problema | Estado actual | Requerido |
|---|---|---|
| Nombre de archivo | `lisa_validation_1001_lf_axi.nii.gz` | `LISA_VALIDATION_1001_axi_enhanced.nii.gz` |
| Entrega | Carpeta con NIfTI sueltos | `LISA_enhanced_predictions.zip` |

La logica de renombrado correcta es:
1. Extraer el ID numerico del nombre del archivo de entrada
2. Identificar el plano (`axi`, `cor`, `sag`)
3. Construir el nombre: `LISA_VALIDATION_{ID}_{plano}_enhanced.nii.gz`
4. Al terminar, comprimir todos los NIfTI en `LISA_enhanced_predictions.zip`

---

## Limites de submission

Ambas tareas comparten el mismo limite de Synapse:

- 2 submissions por equipo por dia
- Las submissions invalidas no cuentan contra el cupo
- Solo el miembro del equipo que envio puede ver el estado en el Submission Dashboard

---

## Dependencias entre tareas

```
Task 1a  ─────────────────────────────────────────────────────►  Submission 1a
  │                                                                (LISA_LF_QC_predictions.csv)
  │  best_1a.pth + umbrales
  ▼
Task 1b  ─────────────────────────────────────────────────────►  Submission 1b
  (usa clasificador 1a como loss adversarial en v2)               (LISA_enhanced_predictions.zip)
```

Task 1a debe tener AUC >= 0.80 antes de iniciar Task 1b.
Task 2 (segmentacion) es independiente de Task 1b y puede ejecutarse en paralelo.

---

## Resumen rapido de archivos a subir

| Tarea | Archivo | Donde generarlo |
|---|---|---|
| Task 1a | `LISA_LF_QC_predictions.csv` | `python task_1a/05_predict_submission.py` |
| Task 1b | `LISA_enhanced_predictions.zip` | `python task_1b/04_predict_submission.py --zip` |
