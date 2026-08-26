# Task 1A — Quality Assessment

## Problema

Clasificación multi-etiqueta de artefactos en imágenes RM de ultra-bajo campo (0.064T, escáner Hyperfine SWOOP). Dado un volumen 2D en cualquiera de los tres planos (axial, coronal, sagital), el modelo debe detectar cuáles de los 7 tipos de artefactos están presentes de forma simultánea.

**Contexto clínico:** Las imágenes de bajo campo se adquieren en entornos de bajos recursos (sin sedación, portátil, bajo costo), donde la calidad de imagen es inherentemente inferior a los equipos de alto campo. Evaluar objetivamente esa calidad es el primer paso para determinar si una imagen es diagnósticamente útil.

---

## Clases objetivo

| Columna | Descripción | Escala | Binarización para evaluación |
|---|---|---|---|
| `Noise` | Ruido excesivo | 0–2 | ≥ 1 → positivo |
| `Zipper` | Artefacto tipo zipper (líneas paralelas) | 0–2 | ≥ 1 → positivo |
| `Positioning` | Mal posicionamiento del paciente | 0–2 | ≥ 1 → positivo |
| `Banding` | Artefacto de bandas horizontales | 0–2 | ≥ 1 → positivo |
| `Motion` | Artefacto de movimiento | 0–2 | ≥ 1 → positivo |
| `Contrast` | Problema de contraste | 0–2 | ≥ 1 → positivo |
| `Distortion` | Distorsión geométrica | 0–2 | ≥ 1 → positivo |

> **Escala real (verificada en CSV):** todos los artefactos usan la misma escala 0 = ausente, 1 = leve, 2 = severo. Para las métricas del challenge se binariza: ≥ 1 es positivo.
> **Formato de filenames en CSV:** `LISA_XXXX_LF_{plano}.nii.gz` (LISA y LF en mayúsculas). En disco: `lisa_XXXX_lf_{plano}.nii.gz` (todo minúsculas). Conversión: `.lower()`.

---

## Datos disponibles

**Fuente:** `data/metadata/lisa_task1a_2026.csv`

- **532 imágenes LF** (244 casos × ~2.2 planos promedio)
- Cada imagen es un `.nii.gz` 2D en uno de tres planos: `lf_axi`, `lf_cor`, `lf_sag`
- El 47% de las imágenes tienen 2 o más artefactos simultáneos

**Distribución de clases en train:**

| Artefacto | Imágenes afectadas | % | Clase |
|---|---|---|---|
| Contrast | 197 | 37% | Frecuente |
| Zipper | 179 | 34% | Frecuente |
| Motion | 169 | 32% | Frecuente |
| Distortion | 144 | 27% | Frecuente |
| Noise | 93 | 17% | Media |
| Positioning | 70 | 13% | Media |
| Banding | 21 | 4% | Rara |
| Sin artefactos | 126 | 24% | — |

**Implicación directa:** Banding al 4% requiere tratamiento especial en la función de pérdida.

---

## Métricas de evaluación (definidas por el challenge)

El ranking final se calcula como la **media de las 5 métricas**, una por clase de artefacto evaluada por separado:

| Métrica | Descripción | Observación |
|---|---|---|
| **Accuracy** | (TP + TN) / total | Dominada por la clase mayoritaria — poco informativa aislada |
| **F1 score** | 2·P·R / (P+R) | Equilibra precision y recall |
| **F2 score** | 5·P·R / (4P+R) | Penaliza más los falsos negativos — recall tiene más peso |
| **Precision** | TP / (TP + FP) | Relevante para evitar falsos positivos |
| **Recall** | TP / (TP + FN) | Relevante para no perder artefactos reales |

> El F2 score indica que el challenge **prefiere no perder artefactos** (recall > precision). El umbral de binarización debe calibrarse hacia recall alto.

**Si una predicción falla o falta, todas las métricas se asignan a 0 para ese caso.**

---

## Arquitectura: EfficientNet-B4 + cabeza multi-etiqueta (v3)

### Decisiones de diseño

- 532 imágenes insuficientes para entrenar desde cero → backbone pre-entrenado en ImageNet obligatorio
- EfficientNet-B4: mejor trade-off parámetros/rendimiento en clasificación médica 2D con datos limitados
- Cabeza multi-etiqueta con sigmoid independiente: permite co-ocurrencia de artefactos sin asumir exclusividad mutua

### Historial de revisiones de estrategia

| Versión | Loss | Batch | Input | Monitor | Score @0.5 | Score calibrado |
|---|---|---|---|---|---|---|
| v1 | `BCE + pos_weight` | random | slice×3 | val AUC | 0.684 | — |
| v2 | `FocalLoss(γ=2)` | random | P25/P50/P75 | challenge_score | 0.671 | — |
| v3 | `FocalLoss(γ=2, α/clase)` | random | P25/P50/P75 | challenge_score | 0.690 | **0.724** |
| **v4** | **Gradient Reweighting** | **RotatingBatch** | P25/P50/P75 | challenge_score | *pendiente* | *pendiente* |

**Fuente v4 — BRIQA (ganador LISA 2025, F1=0.799):**

**Gradient Reweighting:** en cada step, calcula la norma L2 del gradiente de la loss por clase w.r.t. los parámetros de la cabeza clasificadora. Pondera inversamente: `α_c = min_c'(||∇L_c'||) / ||∇L_c||`. Las clases "difíciles" (gradiente alto) reciben menos peso; las "fáciles" o "ignoradas" reciben más. No requiere α manual — el balance emerge del propio entrenamiento.

**RotatingBatchSampler:** mantiene ratio ~2:1 neg:pos en cada batch y cicla los positivos deterministamente. Con Banding (21 ejemplos en train), sin rotating batch algunos epochs verían 0 ejemplos de Banding; con rotating batch cada uno aparece en exactamente `ceil(21/n_pos_per_batch)` batches por epoch.

**Sin ColorJitter:** LISA 2025 mostró que augmentación de intensidad (brillo, contraste) degrada rendimiento porque los patrones de intensidad del MRI son diagnósticos para los artefactos.

**Componentes actuales (v4):**

| Componente | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Input | slice×3 | P25/P50/P75 | P25/P50/P75 | P25/P50/P75 |
| Loss | `BCE + pos_weight` | `FocalLoss(γ=2)` | `FocalLoss(γ=2, α/clase)` | **Gradient Reweighting** |
| Batch | random shuffle | random shuffle | random shuffle | **RotatingBatchSampler** |
| ColorJitter | sí | sí | sí | **no** |
| Monitor | val AUC | challenge_score | challenge_score | challenge_score |

### Diagrama del pipeline

```
NIfTI 2D (lf_axi / lf_cor / lf_sag)
        │
        ▼
  Preprocesamiento
  (normalizar, resize 224×224, convertir a 3 canales)
        │
        ▼
  EfficientNet-B4 (pretrained ImageNet)
  ─────────────────────────────────────
  features → GlobalAveragePooling
        │
        ▼
  Dropout(0.4)
        │
        ▼
  Linear(1792 → 512) → ReLU
        │
        ▼
  Dropout(0.3)
        │
        ▼
  Linear(512 → 7)   ← 7 salidas independientes
        │
        ▼
  Sigmoid por clase
        │
        ▼
  Umbral por clase → predicción binaria 0/1
```

---

## Preprocesamiento

```python
import nibabel as nib
import numpy as np
from PIL import Image

def load_lf_slice(path):
    """Carga un NIfTI 2D LF y retorna array normalizado listo para el modelo."""
    nii = nib.load(path)
    vol = nii.get_fdata()

    # Tomar el slice central si tiene dimensión extra
    if vol.ndim == 3:
        mid = vol.shape[2] // 2
        img = vol[:, :, mid]
    else:
        img = vol

    # Normalizar por percentil 1-99 (evita que outliers dominen)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)

    # Convertir a uint8 para usar transforms de ImageNet
    img_u8 = (img * 255).astype(np.uint8)

    # Replicar a 3 canales (ImageNet backbone espera RGB)
    img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)

    return Image.fromarray(img_rgb)
```

**Augmentación de entrenamiento** (conservadora — las imágenes ya tienen artefactos reales):

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

---

## Dataset y split

### Regla de split: estratificar por caso, no por imagen

```
244 casos
 ├── 195 casos (80%) → train  → ~426 imágenes
 └──  49 casos (20%) → val   → ~106 imágenes
```

Los planos `axi`, `cor`, `sag` de un mismo caso van **todos al mismo split**. Mezclarlos produciría data leakage (el modelo vería en val información de un paciente que ya estaba en train).

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/metadata/lisa_task1a_2026.csv')

# Extraer ID de caso del nombre de archivo
df['case_id'] = df['filename'].str.extract(r'lisa_(\d+)_lf')

cases = df['case_id'].unique()
train_cases, val_cases = train_test_split(cases, test_size=0.2, random_state=42)

df_train = df[df['case_id'].isin(train_cases)].reset_index(drop=True)
df_val   = df[df['case_id'].isin(val_cases)].reset_index(drop=True)
```

---

## Función de pérdida: BCEWithLogitsLoss ponderada

```python
import torch
import torch.nn as nn

ARTIFACT_COLS = ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']

def compute_pos_weights(df_train):
    """Calcula pos_weight = neg_count / pos_count para cada clase."""
    labels = (df_train[ARTIFACT_COLS].values >= 1).astype(float)
    n_pos = labels.sum(axis=0)
    n_neg = len(labels) - n_pos
    pos_weight = n_neg / (n_pos + 1e-8)
    return torch.tensor(pos_weight, dtype=torch.float32)

pos_weight = compute_pos_weights(df_train)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda())
```

> Para Banding (4% ≈ 21 imágenes), `pos_weight ≈ 405/21 ≈ 19`. Esto hace que cada ejemplo positivo de Banding pese 19 veces más en la pérdida.

---

## Modelo

```python
import timm
import torch.nn as nn

class ArtifactClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features  # 1792 para EfficientNet-B4

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)  # logits sin sigmoid (BCEWithLogitsLoss los aplica internamente)
```

---

## Entrenamiento

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

model = ArtifactClassifier().cuda()

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

EPOCHS = 50
PATIENCE = 10
best_auc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    # --- train ---
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    scheduler.step()

    # --- val ---
    model.eval()
    val_auc = evaluate_auc(model, val_loader)  # ver sección de evaluación

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'checkpoints/best_1a.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping en época {epoch}")
            break
```

---

## Evaluación y umbral de binarización

### Métrica de monitoreo interna: AUC promedio por clase

```python
from sklearn.metrics import roc_auc_score
import numpy as np

def evaluate_auc(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.cuda())
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    aucs = []
    for i, col in enumerate(ARTIFACT_COLS):
        if labels[:, i].sum() > 0:  # AUC indefinido si no hay positivos
            aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
    return np.mean(aucs)
```

### Calibración del umbral por clase

Dado que F2 penaliza más los falsos negativos, se optimiza el umbral por clase sobre el conjunto de validación para maximizar F2:

```python
from sklearn.metrics import fbeta_score

def find_best_threshold(probs_col, labels_col, beta=2):
    best_thresh, best_f2 = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs_col >= t).astype(int)
        f2 = fbeta_score(labels_col, preds, beta=beta, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thresh = f2, t
    return best_thresh, best_f2

thresholds = {}
for i, col in enumerate(ARTIFACT_COLS):
    t, f2 = find_best_threshold(val_probs[:, i], val_labels[:, i])
    thresholds[col] = t
    print(f"{col:15s} umbral={t:.2f}  F2={f2:.3f}")
```

---

## Criterios de éxito / fallo

| Métrica | Criterio de paso | Acción si falla |
|---|---|---|
| AUC promedio (7 clases) | ≥ 0.80 | Ver tabla de corrección |
| F1 ponderado | ≥ 0.70 | Ver tabla de corrección |
| Ninguna clase con AUC | < 0.65 | Tratar esa clase por separado |

### Tabla de corrección

| Causa probable | Diagnóstico | Corrección |
|---|---|---|
| Clases raras (Banding 4%) no aprendidas | F1 ≪ AUC en Banding | Aumentar `pos_weight`, oversampling de positivos |
| Overfitting | Val-loss sube, train-loss baja | Reducir lr, aumentar dropout, más augmentación |
| Planos inconsistentes | AUC varía mucho entre axi/cor/sag | Entrenar un modelo por plano |
| Backbone inadecuado | Meseta en val-AUC desde época 5 | Cambiar a Swin-T o ConvNeXt-B |
| Umbral mal calibrado | Recall bajo, precision alta | Bajar umbral hacia 0.3, re-evaluar F2 |

> **Regla de desbloqueo:** Task 1b solo comienza cuando Task 1a tiene AUC ≥ 0.80. Las predicciones de 1a se usan para dirigir la mejora en 1b.

---

## Formato de submission

El challenge evalúa sobre un conjunto de test ciego. El modelo debe predecir las 7 etiquetas binarias para cada imagen de validación `val/single_plane/` (casos 1001–1072).

```python
import os
import pandas as pd

results = []
model.load_state_dict(torch.load('checkpoints/best_1a.pth'))
model.eval()

val_dir = 'data/val/single_plane/'
for case_id in sorted(os.listdir(val_dir)):
    for fname in os.listdir(os.path.join(val_dir, case_id)):
        if not fname.endswith('.nii.gz'):
            continue
        path = os.path.join(val_dir, case_id, fname)
        img = val_transforms(load_lf_slice(path)).unsqueeze(0).cuda()
        with torch.no_grad():
            logits = model(img)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        row = {'filename': fname}
        for i, col in enumerate(ARTIFACT_COLS):
            row[col] = int(probs[i] >= thresholds[col])
        results.append(row)

pd.DataFrame(results).to_csv('submission_task1a.csv', index=False)
```

---

## Checklist de implementación

- [ ] Explorar `lisa_task1a_2026.csv` — verificar distribución real de etiquetas y casos por plano
- [ ] Implementar `load_lf_slice()` — validar con 5 imágenes de ejemplo (axi, cor, sag)
- [ ] Construir `TaskDataset` con split por caso
- [ ] Instanciar `ArtifactClassifier` y verificar shapes entrada/salida
- [ ] Calcular `pos_weight` real desde `df_train` y confirmar valor para Banding
- [ ] Entrenar 50 épocas, monitorear `train_loss`, `val_loss`, `val_auc` por época
- [ ] Calibrar umbrales por clase sobre val, reportar F1/F2/precision/recall por clase
- [ ] Generar `submission_task1a.csv` con formato correcto
- [ ] Verificar que ninguna imagen de val falta en el CSV (métricas a 0 si falta)

---

## Dependencias Python

```
torch >= 2.0
torchvision
timm >= 0.9          # EfficientNet-B4 pretrained
nibabel              # lectura NIfTI
scikit-learn         # AUC, F1, F2
numpy
pandas
Pillow
```

---

## Referencias

- Challenge description: LISA 2026 Task 1A — Quality Assessment
- Métrica F2: [Ravi et al., Med Image Anal 2024](https://doi.org/10.1016/j.media.2023.103033)
- EfficientNet: Tan & Le, ICML 2019
- `timm` library: [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- Datos: [zenodo.org/records/15081583](https://zenodo.org/records/15081583)
