# Task 1B — Log de Experimentos

Registro de la pipeline de mejora de calidad de imágenes RM 0.064T mediante
reducción de ruido y artefactos de movimiento.

---

## Objetivo

Dado un conjunto de imágenes RM uLF con artefactos de ruido y/o movimiento,
generar versiones mejoradas que un evaluador automático de calidad (tipo Task 1a)
clasifique como libres de dichos artefactos.

**Criterio de éxito:** PSNR ≥ 28 dB en el conjunto de validación sintético.

---

## Resumen ejecutivo

| Versión | PSNR val | SSIM val | PSNR criterion | Submission | Estado |
|---|---|---|---|---|---|
| **v1** | **31.37 dB** | **0.872** | ✅ PASS | 72/72 archivos | **Enviado** |

---

## Dataset — Particiones Task 1B

Las 532 imágenes de entrenamiento de Task 1a se dividen en 4 grupos **mutuamente
excluyentes** según la combinación de ruido y movimiento:

| Partición | Ruido | Movimiento | Imágenes | Uso en pipeline |
|---|---|---|---|---|
| `nonoise_nomotion` | No | No | 301 | **Referencia limpia — entrada de entrenamiento** |
| `nonoise_withmotion` | No | Sí | 138 | Evaluación calidad (feedback Task 1a) |
| `withnoise_nomotion` | Sí | No | 62 | Evaluación calidad (feedback Task 1a) |
| `withnoise_withmotion` | Sí | Sí | 31 | Evaluación calidad (feedback Task 1a) |

**Conjunto ciego (submission):** `val/single_plane/` — 72 casos (1001–1072),
un único plano LF por caso, sin etiquetas de calidad.

---

## Estrategia de entrenamiento

No existen pares pixel-a-pixel (imagen ruidosa, imagen limpia del mismo plano).
La estrategia adoptada es **degradación sintética supervisada**:

1. Se cargan las 301 imágenes `nonoise_nomotion` como referencias limpias.
2. Por cada imagen, se extraen todos los slices 2D a lo largo del eje delgado.
3. En cada época se aplica degradación on-the-fly:
   - **Ruido Riciano** `sigma ~ U[0.03, 0.18]` (modelo de ruido MRI)
   - **Ghosting de movimiento** (50% de prob.) — corrupción de fase en k-space:
     2–6 ecos de fantasmas, intensidad `alpha ~ U[0.04, 0.25]`
4. La red aprende la transformación `(imagen degradada → imagen limpia)`.
5. Split: 80/20 por `case_id` (sin leakage entre planos del mismo paciente).

---

## v1 — ResUNet con degradación sintética Riciana + ghosting ✅

### Arquitectura — ResUNet

```
Input:  (B, 1, 256, 256)  — slice normalizado [0,1], redimensionado a IMG_SIZE=256
Output: (B, 1, 256, 256)  — slice mejorado (clamped a [0,1])

Encoder (4 niveles, ResBlocks + MaxPool):
  enc1: ResBlock(1 → 32)   → skip1
  enc2: ResBlock(32 → 64)  → skip2
  enc3: ResBlock(64 → 128) → skip3
  enc4: ResBlock(128 → 256) → skip4

Bottleneck:
  ResBlock(256 → 512)

Decoder (4 niveles, Upsample bilinear + concat skip + ResBlock):
  dec4: (512+256) → 256
  dec3: (256+128) → 128
  dec2: (128+64)  → 64
  dec1: (64+32)   → 32

Salida:  Conv(32 → 1)  — predicción del residuo de artefacto

Formulación global: output = clamp(input + residual, 0, 1)
  → La red predice el artefacto a sustraer, no la imagen limpia directamente.
  → Aceleera convergencia y evita reconstruir lo ya correcto (DnCNN style).
```

**Normalización:** InstanceNorm2d + LeakyReLU(0.1) en cada ResBlock.
Motivo: tolerante a batch sizes pequeños y a resoluciones variables de imágenes RM.

**Parámetros totales:** 7 587 937

### Configuración de entrenamiento

```
IMG_SIZE      = 256        # resize de todos los slices (variable → fija)
BATCH_SIZE    = 8
LR            = 2e-4       (AdamW, weight_decay=1e-4)
SCHEDULER     = CosineAnnealingLR (T_max=100, eta_min=2e-6)
EPOCHS        = 100
PATIENCE      = 20         (early stopping sobre val PSNR)
LOSS          = 0.6 × L1 + 0.4 × (1 − SSIM)
NUM_WORKERS   = 0          (Windows)
RANDOM_SEED   = 42
```

**Por qué L1 + SSIM:**
- L1 es robusto a outliers, preserva bordes sin el sesgo de suavizado de L2.
- SSIM penaliza pérdidas estructurales (contraste, textura) que L1 ignora.
- λ_L1=0.6 domina ligeramente para evitar imágenes excesivamente suavizadas.

### Resolución de imágenes

Los volúmenes NIfTI LF tienen resoluciones nativas variables (ej. 128×128, 144×128,
192×256). El dataset cargaba correctamente pero el `DataLoader` fallaba al apilar
tensores de distintos tamaños en un batch (`stack expects equal size`).

**Solución:** Redimensionar cada slice a 256×256 (PIL bilinear) antes de crear
el tensor. Al momento de submission, el slice mejorado se redimensiona de vuelta
a la resolución original antes de guardar el NIfTI.

### Resultados de entrenamiento

| Época mejor | Val PSNR | Early stopping | Tiempo total |
|---|---|---|---|
| ~67 (estimado) | **31.37 dB** | Época 87 (+20 sin mejora) | **872.9 min (~14.5h)** |

Criterio PSNR ≥ 28 dB: **PASS** con +3.37 dB de margen.

### Resultados de evaluación

#### Modo A — PSNR/SSIM sintético (val fold)

| Métrica | Imagen degradada | Imagen mejorada | Mejora |
|---|---|---|---|
| PSNR | 18.40 dB | **31.24 dB** | **+12.84 dB** |
| SSIM | 0.303 | **0.872** | **+0.569** |

La mejora de +12.84 dB sobre imágenes con ruido Riciano + ghosting sintético
confirma que el modelo aprendió a revertir ambos tipos de degradación.

#### Modo B — Feedback Task 1a (imágenes con ruido/movimiento real)

Se evalúa si Task 1a reduce su score de Noise/Motion tras la mejora:

| Partición | Noise antes | Noise después | Motion antes | Motion después |
|---|---|---|---|---|
| withnoise_nomotion | 0.686 | 0.724 | 0.325 | 0.322 |
| nonoise_withmotion | 0.677 | 0.720 | 0.298 | 0.304 |
| withnoise_withmotion | 0.707 | 0.727 | 0.366 | 0.360 |

**Hallazgo crítico:** El score de Noise *aumenta* ligeramente tras la mejora;
el score de Motion permanece prácticamente sin cambio.

**Diagnóstico:** El modelo reduce el ruido térmico Riciano (medido por PSNR/SSIM)
pero no elimina los patrones que Task 1a usa como features discriminativas. Las
razones probables:
- El ruido Riciano sintético tiene una distribución estadística distinta al ruido
  real del Hyperfine SWOOP 0.064T.
- Task 1a fue entrenado con imágenes reales y puede detectar texturas de ruido
  que el denoiser no aprendió a eliminar (posiblemente artefactos de RF, aliasing).
- El denoiser introduce un suavizado global que paradójicamente puede hacer que
  regiones con contraste bajo parezcan más uniformes (confundido con artefacto
  de ruido por el clasificador).

**Impacto en submission:** No bloquea la entrega. El criterio oficial es PSNR ≥ 28 dB,
que se cumple holgadamente. Sin embargo, si la evaluación del challenge usa un
clasificador tipo Task 1a como métrica secundaria, el ranking podría verse afectado.

### Submission generado

```
task_1b/submission/single_plane/
  1001/ lisa_validation_1001_lf_???.nii.gz
  ...
  1072/ lisa_validation_1072_lf_???.nii.gz

Total: 72 archivos | 0 errores | 30.5 segundos
```

---

## Bugs resueltos durante el desarrollo

| Bug | Causa | Fix |
|---|---|---|
| `stack expects equal size` | Slices de distintas resoluciones nativas en el mismo batch | Redimensionar todos a 256×256 con `_resize()` en lugar de `_pad16()` |
| `UnicodeEncodeError` en print | Consola Windows cp1252 no soporta `✓`/`✗`/`≥` | Reemplazar por ASCII (`PASS`/`FAIL`/`>=`) |
| `TypeError: bool is not JSON serializable` | `psnr_torch` retorna `numpy.float`, `>= MIN_PSNR` retorna `numpy.bool_` | `float()`/`bool()` explícito antes de serializar |
| `UnpicklingError` en `torch.load` | PyTorch 2.6 cambió default a `weights_only=True`; checkpoint contiene `numpy.float` en metadata | `weights_only=False` en todos los `torch.load` + `float()` en `torch.save` metadata |
| `AttributeError: _pad16` en submission | `_pad16` fue renombrado a `_resize` en dataset pero no en `04_predict_submission.py` | Actualizar llamada + lógica de post-procesado (crop → resize-back) |

---

## Limitaciones y trabajo futuro

### Limitación 1 — Domain gap en la degradación sintética
El ruido Riciano sintético es una aproximación del ruido térmico real. El scanner
Hyperfine SWOOP 0.064T tiene características específicas (B0 bajo, RF coils
distintos, artefactos de susceptibilidad) que no modelamos. El feedback de Task 1a
(scores de Noise que no mejoran) evidencia este gap.

**Camino de mejora:** Añadir un término de pérdida adversarial usando Task 1a
como discriminador:
```python
L_total = L1 + SSIM + λ_adv × L_task1a(enhanced, target_noise=0, target_motion=0)
```
Esto requiere ~14h adicionales de entrenamiento.

### Limitación 2 — Sin pares reales
No existen pares pixel-a-pixel (misma escena, con y sin artefacto) en el dataset.
Pares reales permitirían supervisión directa y eliminarían el domain gap.

**Alternativas a explorar:**
- Noise2Void / Noise2Self (self-supervised, sin pares)
- CycleGAN entre particiones `nonoise_nomotion` ↔ `withnoise_*` (unpaired)

### Limitación 3 — Procesado slice-a-slice sin contexto 3D
El modelo procesa cada slice 2D de forma independiente. Los artefactos de
movimiento en MRI se manifiestan como incoherencias entre slices (ghosting a
lo largo del eje de codificación de fase). Un modelo 3D o pseudo-3D podría
capturar mejor este comportamiento.

---

## Archivos generados

```
task_1b/
├── config.py                    # Hiperparámetros y rutas
├── dataset.py                   # SyntheticDenoiseDataset, InferenceDataset, build_split
├── model.py                     # ResUNet (7.6M params)
├── losses.py                    # CombinedLoss (L1 + SSIM)
├── 01_explore_data.py           # Exploración particiones
├── 02_train.py                  # Entrenamiento con early stopping
├── 03_evaluate.py               # Evaluación PSNR/SSIM + feedback Task 1a
├── 04_predict_submission.py     # Generación de NIfTI mejorados
├── EXECUTION_ORDER.md           # Guía de ejecución
├── checkpoints/
│   └── best_1b.pth              # Mejor checkpoint (época ~67, PSNR=31.37 dB)
├── results/
│   ├── 01_exploration.json      # Estadísticas de particiones
│   ├── 02_training.json         # PSNR final, épocas, tiempo
│   ├── 03_evaluation.json       # PSNR/SSIM sintético + feedback Task 1a
│   └── 04_submission.json       # Resumen del submission (72/72 archivos)
└── submission/
    └── single_plane/            # 72 NIfTI mejorados para el challenge
        ├── 1001/
        └── ...1072/
```

---

## Decisiones de diseño clave

| Decisión | Alternativa considerada | Motivo de la elección |
|---|---|---|
| Degradación sintética en train | Pares reales / CycleGAN | No existen pares reales; CycleGAN es más complejo y lento |
| Ruido Riciano + ghosting FFT | Solo Ruido Gaussiano | El ruido en MRI sigue distribución Riciana; ghosting simula movimiento real |
| ResUNet con InstanceNorm | BatchNorm / GroupNorm | Tolerante a batch=8 y resoluciones variables |
| Residual global (output = input + net) | Predicción directa de imagen limpia | Aprende solo los artefactos, no toda la imagen → convergencia más rápida |
| Resize a 256×256 | Padding a múltiplo de 16 | Padding preserva resoluciones distintas → error de collate en batch |
| L1 + SSIM (0.6/0.4) | L2, Perceptual (VGG) | L1 preserva bordes; SSIM añade estructura; VGG requeriría GPU extra y más tiempo |
