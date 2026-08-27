# Task 1A — Mejoras: Clasificación de Artefactos

> **Nota de contexto (2026-08-19):** este documento es un roadmap de una fase intermedia
> del proyecto (v7, evaluada por F1 binario). El paper final adoptó una formulación
> distinta — clasificación ordinal de dos cabezas evaluada por micro-accuracy sobre la
> grilla `{0,1,2}` — y alcanzó 0.839, superando el objetivo F1≥0.75 que este documento
> planteaba por otra vía. Ver [`paper_combined/paper.tex`](../paper_combined/paper.tex)
> §Methods/Task 1A y [`task_1a/README.md`](../task_1a/README.md) para el pipeline y los
> resultados vigentes. Se conserva como registro de las alternativas consideradas.

## Resultados actuales (v7)

| Métrica | Valor | Criterio mínimo | Benchmark BRIQA 2025 | Estado |
|---|---|---|---|---|
| AUC promedio (7 clases) | **0.877** | ≥ 0.80 | — | ✅ PASS |
| Challenge score @0.5 | **0.706** | — | — | — |
| Challenge score calibrado | **0.739** | — | — | — |
| F2 promedio calibrado | **0.772** | — | — | — |
| F1 promedio | **0.640** | ≥ 0.70 | **0.799** | ⚠️ -0.159 vs. benchmark |

**Fuente del benchmark:** BRIQA (Best-Ranked IQA), ganador de la tarea equivalente en LISA 2025.
Publicado en: *Springer LNCS [doi.org/10.1007/978-3-031-83008-2_6](https://link.springer.com/chapter/10.1007/978-3-031-83008-2_6)*.

### Métricas por clase (estado actual, umbrales calibrados)

| Label | AUC | F2 @opt | Umbral | Recall | Precision | n_pos |
|---|---|---|---|---|---|---|
| Zipper | 0.932 | 0.873 | 0.45 | 0.917 | 0.733 | 36 |
| Noise | 0.915 | 0.800 | 0.45 | 0.889 | 0.571 | 18 |
| ThalamusR | — | — | — | — | — | — |
| Distortion | 0.875 | 0.789 | 0.33 | 1.000 | 0.429 | 27 |
| Contrast | 0.873 | 0.791 | 0.63 | 0.833 | 0.658 | 30 |
| Banding | 0.858 | 0.714 | 0.51 | 0.800 | 0.500 | 5 |
| Motion | 0.856 | 0.791 | 0.47 | 0.861 | 0.596 | 36 |
| **Positioning** | **0.832** | **0.643** | 0.59 | 0.750 | 0.409 | 12 |

**Clase más débil:** Positioning — peor AUC, peor F2 calibrado, solo 12 positivos en val.

### Diagnóstico del gap vs. BRIQA 2025

La diferencia F1 0.640 → 0.799 (≈ +0.16) se explica por tres factores identificados:

1. **Rebalanceo de gradientes:** BRIQA usa Gradient Reweighting (ponderación adaptativa por norma del gradiente). La v7 actual usa FocalLoss + alpha fija por clase — más rígida.
2. **Muestreo de batches:** BRIQA usa ciclos deterministas sobre los positivos (garantiza que Banding y Positioning aparezcan en cada epoch). La v7 usa `ArtifactStratifiedSampler` que aún puede dejar epochs sin Banding.
3. **Sin ColorJitter:** BRIQA demostró que `brightness/contrast jitter` degrada el rendimiento porque los patrones de intensidad del MRI son diagnósticos. La v7 todavía puede incluirlo — verificar `config.py`.

---

## Mejora 1 — Gradient Reweighting (impacto estimado: +0.05–0.10 F1)

Fuente: BRIQA (LISA 2025). Sustituye la alpha fija de FocalLoss por una ponderación dinámica calculada cada step.

### Implementación

```python
# task_1a/losses.py (crear archivo nuevo)

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReweightingLoss(nn.Module):
    """
    Loss con rebalanceo adaptativo basado en la norma del gradiente por clase.
    - Las clases con gradiente grande (bien aprendidas) reciben menos peso.
    - Las clases con gradiente pequeño (ignoradas o difíciles) reciben más peso.
    """

    def __init__(self, n_classes: int = 7, smoothing: float = 0.05):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        # Pesos exponencialmente suavizados
        self.register_buffer('ema_norms', torch.ones(n_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                head_params: list) -> torch.Tensor:
        """
        logits:      (B, C) — sin sigmoid
        targets:     (B, C) — binario 0/1
        head_params: list[Parameter] de la cabeza clasificadora
        """
        # Loss por clase (sin reducción)
        per_class_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        ).mean(dim=0)  # (C,)

        # Calcular norma L2 del gradiente de la loss de cada clase
        grad_norms = torch.zeros(self.n_classes, device=logits.device)
        for c in range(self.n_classes):
            grads = torch.autograd.grad(
                per_class_loss[c], head_params,
                retain_graph=True, create_graph=False, allow_unused=True
            )
            norm = sum(
                g.norm() for g in grads if g is not None
            )
            grad_norms[c] = float(norm) + 1e-8

        # Actualizar EMA de las normas
        self.ema_norms = (
            (1 - self.smoothing) * self.ema_norms +
            self.smoothing * grad_norms.detach()
        )

        # Pesos inversamente proporcionales a la norma EMA
        min_norm = self.ema_norms.min()
        weights = (min_norm / self.ema_norms).clamp(0.1, 10.0)

        # Loss ponderada
        loss = (per_class_loss * weights).mean()
        return loss
```

### Integración en el loop de entrenamiento

```python
# En 02_train.py — reemplazar criterion

from task_1a.losses import GradientReweightingLoss

head_params = list(model.head.parameters())
criterion = GradientReweightingLoss(n_classes=7).cuda()

# En el loop:
optimizer.zero_grad()
logits = model(imgs)
loss = criterion(logits, labels, head_params)  # <-- pasa head_params
loss.backward()
optimizer.step()
```

> **Nota:** `torch.autograd.grad` con `retain_graph=True` añade ~20% de overhead por step.
> Alternativa más rápida: calcular la norma solo cada 10 steps y usar la EMA del step anterior en los demás.

---

## Mejora 2 — RotatingBatchSampler determinista (impacto estimado: +0.03–0.05 F1 en Banding/Positioning)

Garantiza que Banding (5 positivos en train) aparezca en **todos** los batches de cada epoch.

```python
# task_1a/samplers.py

import math
import torch
from torch.utils.data import Sampler


class RotatingBatchSampler(Sampler):
    """
    Para cada batch: 1 positivo garantizado por clase rara, resto aleatorio.
    Los positivos rotan de forma determinista (ninguno se repite hasta que se agote el ciclo).
    """

    def __init__(self, labels: torch.Tensor, batch_size: int,
                 rare_classes: list, rare_quota: int = 1):
        """
        labels:       (N, C) tensor binario
        batch_size:   tamaño del batch
        rare_classes: índices de clases a garantizar (ej. [3] para Banding)
        rare_quota:   positivos garantizados por clase por batch
        """
        self.batch_size = batch_size
        self.rare_quota = rare_quota

        # Índices positivos por clase rara
        self.rare_pos = {}
        for c in rare_classes:
            pos_idx = labels[:, c].nonzero(as_tuple=True)[0].tolist()
            self.rare_pos[c] = pos_idx

        # Índices "libres" (no forzados en este batch)
        all_idx = set(range(len(labels)))
        forced = set(i for idxs in self.rare_pos.values() for i in idxs)
        self.free_idx = sorted(all_idx - forced)
        self.n_samples = len(labels)

        # Punteros de rotación
        self._reset_pointers()

    def _reset_pointers(self):
        self._ptrs = {c: 0 for c in self.rare_pos}

    def __iter__(self):
        import random
        free = self.free_idx.copy()
        random.shuffle(free)
        free_ptr = 0

        n_batches = math.ceil(self.n_samples / self.batch_size)
        for _ in range(n_batches):
            batch = []
            # 1. Añadir positivos rotativos de clases raras
            for c, pos_list in self.rare_pos.items():
                for _ in range(self.rare_quota):
                    idx = pos_list[self._ptrs[c] % len(pos_list)]
                    batch.append(idx)
                    self._ptrs[c] += 1
            # 2. Rellenar con muestras libres
            n_fill = self.batch_size - len(batch)
            batch += free[free_ptr:free_ptr + n_fill]
            free_ptr += n_fill
            if free_ptr >= len(free):
                free = self.free_idx.copy()
                random.shuffle(free)
                free_ptr = 0
            yield batch

    def __len__(self):
        return math.ceil(self.n_samples / self.batch_size)
```

### Uso en DataLoader

```python
# En 02_train.py

from task_1a.samplers import RotatingBatchSampler

# labels_train: tensor (N_train, 7)
sampler = RotatingBatchSampler(
    labels=labels_train,
    batch_size=BATCH_SIZE,
    rare_classes=[3, 2],  # Banding (índice 3), Positioning (índice 2)
    rare_quota=1
)
train_loader = DataLoader(dataset_train, batch_sampler=sampler, num_workers=0)
```

---

## Mejora 3 — Eliminar ColorJitter (impacto estimado: +0.01–0.03 F1)

Lección de BRIQA: `brightness/contrast jitter` daña el rendimiento en MRI porque la intensidad relativa es un feature diagnóstico. **Eliminar** del pipeline de augmentación.

```python
# En dataset.py — transforms de entrenamiento

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    # SIN ColorJitter — ver BRIQA 2025
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

---

## Mejora 4 — Modelo por plano (impacto potencial alto para Positioning)

Positioning tiene AUC=0.832 y aparece más en planos específicos. Entrenar modelos separados por plano (axi / cor / sag) puede capturar mejor los patrones espaciales.

```python
# En config.py — añadir
PLANE_SPECIFIC = True  # Si True, entrena 3 modelos (uno por plano)
PLANES = ['axi', 'cor', 'sag']

# En 02_train.py — si PLANE_SPECIFIC:
for plane in PLANES:
    df_train_plane = df_train[df_train['plane'] == plane]
    df_val_plane   = df_val[df_val['plane'] == plane]
    # Entrenar modelo independiente, guardar en checkpoints/best_1a_{plane}.pth
```

**Cuándo activar:** si la varianza de AUC entre planos es > 0.05 para una misma clase.

---

## Mejora 5 — TTA (Test-Time Augmentation) en inferencia

Sin coste de reentrenamiento. Aumenta la estabilidad de las predicciones en casos límite.

```python
# En 05_predict_submission.py

def predict_with_tta(model, img_tensor, thresholds, artifact_cols):
    """Promedia predicciones con y sin flip horizontal/vertical."""
    augmentations = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),   # flip H
        lambda x: torch.flip(x, dims=[-2]),   # flip V
        lambda x: torch.flip(x, dims=[-1, -2]),  # flip HV
    ]
    all_probs = []
    with torch.no_grad():
        for aug in augmentations:
            logits = model(aug(img_tensor))
            probs  = torch.sigmoid(logits).cpu().numpy()[0]
            all_probs.append(probs)
    mean_probs = np.mean(all_probs, axis=0)
    return {col: int(mean_probs[i] >= thresholds[col])
            for i, col in enumerate(artifact_cols)}
```

---

## Roadmap de mejoras

| Prioridad | Mejora | Tiempo estimado | Ganancia F1 esperada |
|---|---|---|---|
| 1 | Gradient Reweighting (v8) | ~14h entrenamiento | +0.05–0.10 |
| 2 | RotatingBatchSampler | ~14h entrenamiento | +0.03–0.05 |
| 3 | Eliminar ColorJitter | Modificación trivial + reentrenamiento | +0.01–0.03 |
| 4 | TTA en inferencia | 0h reentrenamiento | +0.01–0.02 |
| 5 | Modelos por plano | ~42h entrenamiento (3×) | Condicionado al análisis por plano |

**Objetivo:** alcanzar F1 ≥ 0.75 (gap restante con BRIQA: 0.049) con mejoras 1+2+3.

### Versión recomendada (v8)

```
v8 = Gradient Reweighting + RotatingBatchSampler + sin ColorJitter
   + P25/P50/P75 (mantener)
   + challenge_score como monitor (mantener)
```

---

## Archivos a modificar

| Archivo | Cambio |
|---|---|
| [task_1a/losses.py](../task_1a/losses.py) | Crear — `GradientReweightingLoss` |
| [task_1a/samplers.py](../task_1a/samplers.py) | Crear — `RotatingBatchSampler` |
| [task_1a/dataset.py](../task_1a/dataset.py) | Eliminar `ColorJitter` de train transforms |
| [task_1a/02_train.py](../task_1a/02_train.py) | Usar nueva loss y sampler |
| [task_1a/05_predict_submission.py](../task_1a/05_predict_submission.py) | Añadir TTA |
| [task_1a/config.py](../task_1a/config.py) | `VERSION = "v8"`, `USE_GR_LOSS = True` |
