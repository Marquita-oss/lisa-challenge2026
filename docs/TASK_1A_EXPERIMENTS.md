# Task 1A — Log de Experimentos

Registro cronológico de todas las versiones entrenadas, sus configuraciones,
resultados y la decisión que generó cada cambio.

---

## Resumen ejecutivo

| Versión | Challenge @0.5 | Challenge calibrado | AUC | overall_pass | Estado |
|---|---|---|---|---|---|
| v1 | 0.684 | — | 0.878 | ❌ | Archivado |
| v2 | 0.671 | — | 0.883 | ❌ | Archivado |
| v3 | 0.690 | **0.724** | 0.888 | ❌ | Archivado |
| v4-beta | 0.178 | — | 0.500 | ❌ | Descartado |
| v4 | 0.409 | 0.591 | 0.759 | ❌ | Archivado |
| **v5** | **0.704** | **0.711** | **0.891** | ✅ | **Mejor actual** |
| v6 | *pendiente* | *pendiente* | *pendiente* | — | En evaluación |

Challenge score = mean(accuracy, F1, F2, precision, recall) a umbral 0.5.

---

## Dataset

- **532 imágenes** LF (0.064T Hyperfine SWOOP), 244 casos, 3 planos (axi/cor/sag)
- **7 labels** por imagen: Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion
- Escala 0-2 (ausente/leve/severo), binarizado a ≥1 = positivo para evaluación
- Split: 80% train (195 casos, 421 img) / 20% val (49 casos, 111 img), por caso (sin leakage)
- Distribución: Banding 4% → 21 positivos en train. Contrast 37% → más común.

---

## Configuración base (constante en todos los experimentos)

```
Input:        3 slices P25/P50/P75 del eje grueso (5mm) → 3 canales RGB
Backbone:     EfficientNet-B4 o DenseNet169 (ver por versión)
Head:         Dropout(0.4) → Linear(feat→512) → ReLU → Dropout(0.3) → Linear(512→7)
Optimizer:    AdamW, lr=1e-4, weight_decay=1e-2
Scheduler:    CosineAnnealingLR, T_max=EPOCHS
Monitor:      challenge_score = mean(acc, F1, F2, prec, rec) a umbral 0.5
Evaluación:   03_evaluate.py → umbral 0.5
Calibración:  04_calibrate_thresholds.py → optimiza umbral por F2 por clase
```

---

## v1 — Baseline

**Config:**
- Backbone: EfficientNet-B4
- Loss: `BCEWithLogitsLoss + pos_weight` (n_neg/n_pos por clase)
  - Banding pos_weight ≈ 20×, Positioning ≈ 6.7×
- Augmentación: flip H/V + rotation 15° + ColorJitter(brightness, contrast)
- Input: slice central × 3 canales idénticos (no P25/P50/P75)
- Batch: random shuffle
- Monitor: val AUC
- EPOCHS=50, PATIENCE=10

**Resultados:**

| AUC | Acc | F1 | F2 | Prec | Rec | Score @0.5 | Score cal. |
|---|---|---|---|---|---|---|---|
| 0.878 | 0.826 | 0.618 | 0.684 | 0.545 | 0.745 | 0.684 | — |

**Diagnóstico:** AUC bueno (0.878), pero Precision=0.545 y Recall=0.745 muy desbalanceados. El pos_weight agresivo sesgaba las probabilidades hacia positivo. A umbral 0.5 había muchos FP.

**Decisión:** No parchar el umbral. Cambiar la función de pérdida para atacar la causa raíz.

---

## v2 — Focal Loss puro

**Cambios respecto a v1:**
- Loss: `FocalLoss(γ=2)` sin alpha
- Input: P25/P50/P75 → 3 canales reales (primera vez)
- Monitor: challenge_score (en lugar de AUC)
- Eliminación de ColorJitter (LISA 2025 mostró degradación con aug. de intensidad)

**Resultados:**

| AUC | Acc | F1 | F2 | Prec | Rec | Score @0.5 |
|---|---|---|---|---|---|---|
| 0.883 | 0.879 | 0.598 | 0.552 | **0.797** | 0.531 | 0.671 |

**Diagnóstico:** El péndulo se fue al extremo opuesto. Precision alta (0.797) pero Recall bajo (0.531). Sin alpha, el desbalance de clases raras domina: Banding (4%) y Positioning (13%) reciben casi ninguna predicción positiva. Recall@Positioning=0.25, Recall@Banding=0.20.

**Decisión:** Añadir alpha por clase a FocalLoss para reintroducir balance sin el sesgo extremo de pos_weight.

---

## v3 — Focal Loss + alpha por clase

**Cambios respecto a v2:**
- Loss: `FocalLoss(γ=2, α/clase)` donde `α_c = n_neg_c / n_total`
  - Banding: α=0.961, Positioning: α=0.868, Contrast: α=0.630
- Todo lo demás igual a v2

**Resultados:**

| AUC | Acc | F1 | F2 | Prec | Rec | Score @0.5 | Score cal. |
|---|---|---|---|---|---|---|---|
| 0.888 | 0.849 | 0.632 | 0.669 | 0.595 | 0.705 | 0.690 | **0.724** |

Umbrales calibrados: Noise=0.31, Motion=0.29, Distortion=0.37, Positioning=0.57
Challenge score con umbrales calibrados: **0.724** (PASS del criterio ≥ 0.70)

**Diagnóstico:** El modelo discrimina bien (AUC=0.888) pero las probabilidades no están calibradas para umbral 0.5. La calibración de step 04 aporta +0.034 puntos. A umbral 0.5 no supera 0.70 por sí solo.

**Decisión:** El modelo necesitaba ver las clases raras más frecuentemente durante el entrenamiento. Explorar batch sampling estratificado.

---

## v4-beta — Gradient Reweighting (DESCARTADO)

**Cambio respecto a v3:**
- Loss: `GradientReweightedLoss` (BRIQA, ganador LISA 2025)
  - Pondera cada clase inversamente a la norma L2 del gradiente w.r.t. head params
  - `α_c = min_c'(||∇L_c'||) / ||∇L_c||`

**Resultados:**

| AUC | Score @0.5 |
|---|---|
| 0.500 | 0.178 |

**Diagnóstico:** Colapso total. El modelo predice todo negativo. El gradient reweighting de BRIQA está diseñado para balancear SEVERIDADES (0/1/2) de UN artefacto. Aplicado entre 7 ARTEFACTOS distintos, upweightea las clases con gradiente pequeño (raras: Banding) y downweightea las informativas (Zipper, Motion). El modelo colapsa a predecir todo negativo para las clases comunes.

**Decisión:** Descartar gradient reweighting para nuestro setup multi-label. Revertir a FocalLoss+alpha. Probar RotatingBatchSampler de BRIQA.

---

## v4 — Focal Loss + RotatingBatchSampler

**Cambios respecto a v3:**
- Batch: `RotatingBatchSampler` (ratio 2:1 neg:pos, cycling determinista)
- Loss: FocalLoss + alpha (sin cambio)

**Resultados:**

| AUC | Score @0.5 | Score cal. | Batches/época |
|---|---|---|---|
| 0.759 | 0.409 | 0.591 | **9** |

**Diagnóstico:** El sampler causó un problema estructural. En nuestro dataset el 76% de las imágenes tienen al menos un artefacto (clase "positiva" = mayoría). El sampler usa la clase negativa como eje para n_batches: `100 // 11 = 9 batches/época` en lugar de los 26 del shuffle aleatorio. El modelo recibió 3× menos actualizaciones de gradiente. AUC bajó de 0.888 a 0.759 — el modelo no convergió en 50 épocas.

**Decisión:** El concepto de cycling es correcto pero la implementación no es adecuada para nuestra distribución de datos. Rediseñar el sampler para que sea independiente de la distribución pos/neg global y garantice cobertura por clase de artefacto.

---

## v5 — Focal Loss + ArtifactStratifiedSampler ✅ MEJOR ACTUAL

**Cambios respecto a v4:**
- Batch: `ArtifactStratifiedSampler` (nuevo)
  - 7 slots ancla: 1 positivo de cada clase de artefacto, cycling determinista
  - 9 slots relleno: random del dataset completo
  - `n_batches = len(df) // batch_size = 26/época` (igual que random)
- EPOCHS=100, PATIENCE=20

**Resultados:**

| AUC | Acc | F1 | F2 | Prec | Rec | Score @0.5 | Score cal. | Época mejor |
|---|---|---|---|---|---|---|---|---|
| **0.891** | 0.831 | 0.631 | **0.698** | 0.580 | **0.782** | **0.704** | **0.711** | 19 |

**Por clase:**

| Clase | AUC | Recall | F2 | vs v3 F2 |
|---|---|---|---|---|
| Noise | 0.931 | 0.556 | 0.595 | −0.104 |
| Zipper | 0.940 | 0.806 | 0.810 | −0.051 |
| **Positioning** | **0.916** | **1.000** | **0.682** | **+0.090** |
| **Banding** | **0.909** | **0.800** | **0.625** | **+0.069** |
| **Motion** | 0.825 | **0.806** | **0.763** | **+0.106** |
| **Contrast** | 0.881 | 0.800 | **0.736** | **+0.017** |
| **Distortion** | 0.833 | 0.704 | **0.674** | **+0.077** |

`overall_pass = TRUE` a umbral 0.5 por primera vez.
Umbrales calibrados muy cercanos a 0.5 (0.37–0.51) — modelo mejor calibrado que v3.
El sampler garantizó cobertura de Banding (21 ejemplos) en cada batch → recall 0.60→0.80.

**Por qué bajó Precision/Accuracy vs v2-v3:** Es el tradeoff precision-recall esperado. Al capturar más verdaderos positivos (↑recall), también se capturan más FP (↓precision). El net effect en challenge_score es positivo: +0.014 vs v3.

**Decisión:** Criterion passed. Proceder con Fase 1 de mejoras: DenseNet169 + ElasticTransform + TTA.

---

## v6 — DenseNet169 + ElasticTransform + TTA (en evaluación)

**Cambios respecto a v5:**
- Backbone: `densenet169` (antes EfficientNet-B4)
  - Conexiones densas entre capas → mejor reutilización de features para texturas MRI
  - Feat dim: 1664 (vs 1792). El ganador de LISA 2025 lo eligió sobre EfficientNet
- Augmentación: + `ElasticTransform(alpha=50, sigma=5)` en train
  - Simula distorsiones geométricas suaves — uno de los 7 artefactos target
  - Conservador (alpha bajo) para no confundir con Distortion real
- TTA en inference: `05_predict_submission.py` promedia 4 orientaciones
  (original + hflip + vflip + ambas). No requiere re-entrenar.

**Proyección esperada:** challenge_score @0.5 ~0.72–0.73 basado en benchmark LISA 2025.

**Resultados:** *pendientes*

---

## Learnings transversales

### Lo que funciona (validado empíricamente)

| Técnica | Desde | Evidencia |
|---|---|---|
| P25/P50/P75 multichannel | v2 | AUC consistentemente > 0.88 |
| FocalLoss(γ=2) | v2 | Mejor calibración que BCE puro |
| Alpha por clase | v3 | Balance precision/recall sin colapso |
| Sin ColorJitter | v2 | LISA 2025 + AUC mejora |
| ArtifactStratifiedSampler | v5 | overall_pass=True, recall raro +20pp |
| Challenge score como monitor | v2 | Optimiza directamente la métrica real |

### Lo que no funciona (descartado)

| Técnica | Versión | Por qué |
|---|---|---|
| BCE + pos_weight agresivo | v1 | Sesga probs hacia positivo, high FP |
| FocalLoss sin alpha | v2 | Clases raras colapsan a negativo |
| Gradient reweighting multi-label | v4-beta | Diseñado para severidades de 1 clase, no 7 clases distintas |
| RotatingBatchSampler | v4 | Dataset mayoritariamente positivo (76%) → 9 batches/época |

### Insights de LISA 2025 aplicados

| Insight | Fuente | Aplicado en |
|---|---|---|
| P25/P50/P75 multichannel | BRIQA + Lazo-Quispe | v2+ |
| Sin augmentación de intensidad | Lazo-Quispe | v2+ |
| DenseNet169 > EfficientNet para MRI | BRIQA | v6 |
| Rotating batch (concepto) | BRIQA | v4 (mal), v5 (correcto) |

---

## Archivos generados por versión

```
task_1a/
├── results/
│   ├── 00_exploration.json       # exploración del dataset (estático)
│   ├── 01_setup.json             # verificación del entorno (estático)
│   ├── 02_training.json          # historial de la última versión entrenada
│   ├── 03_evaluation.json        # métricas de la última evaluación
│   └── 04_thresholds.json        # umbrales calibrados + challenge score
├── checkpoints/
│   └── best_1a.pth               # mejor checkpoint (sobreescrito por cada entrenamiento)
└── (raíz del proyecto)
    └── submission_task1a.csv     # predicciones para el challenge
```

> Los checkpoints se sobreescriben en cada entrenamiento. Para preservar v5:
> copiar `best_1a.pth` a `checkpoints/best_1a_v5.pth` antes de entrenar v6.
