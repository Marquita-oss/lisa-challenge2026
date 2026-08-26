# Task 1B — Mejoras: Reducción de Artefactos (Image Enhancement)

> **Nota de contexto (2026-08-19):** este documento propone un roadmap (loss adversarial
> con Task 1A como discriminador, degradación sintética más realista, Restormer) para
> superar el domain gap encontrado en v1. El paper final llegó a una conclusión distinta
> tras evaluar las cuatro variantes entrenadas (incluyendo la adversarial y la de registro
> + loss perceptual descritas aquí): ninguna le gana en el leaderboard oficial a un
> *pass-through* nativo sin parámetros aprendidos, por un techo de registro CISO↔LF
> (~0.5–0.6 de correlación) y un efecto Goodhart en la selección por proxy local. Ver
> [`paper_combined/paper.tex`](../paper_combined/paper.tex) §Methods/Results Task 1B y
> [`docs/DOCKER_SUBMISSION.md`](DOCKER_SUBMISSION.md) para la decisión final de submission.
> Se conserva como registro de las alternativas consideradas y por qué se descartaron.

## Resultados actuales (v1 — ResUNet)

| Métrica | Imagen degradada | Imagen mejorada | Criterio | Estado |
|---|---|---|---|---|
| PSNR (sintético) | 18.40 dB | **31.24 dB** | ≥ 28 dB | ✅ PASS (+3.24 dB) |
| SSIM (sintético) | 0.303 | **0.872** | ≥ 0.85 | ✅ PASS |
| Mejora PSNR | — | **+12.84 dB** | — | — |
| Submission | — | **72/72 archivos** | — | ✅ Generado |

**Fuente del benchmark:** PSNR ≥ 28 dB definido en la especificación oficial del LISA Challenge 2026 ([zenodo.org/records/15081583](https://zenodo.org/records/15081583)). La referencia externa SFNet (LISA 2025) alcanzó mejoras DSC de 61% → 71% en Task 2, lo que valida el pipeline de mejora de imagen como potenciador de la segmentación.

### Hallazgo crítico — Domain gap

El feedback de Task 1A tras aplicar el denoiser muestra que el **score de Noise aumenta** en vez de bajar:

| Partición | Noise antes | Noise después | Δ | Esperado |
|---|---|---|---|---|
| withnoise_nomotion | 0.686 | 0.724 | **+0.038** | ↓ |
| nonoise_withmotion | 0.677 | 0.720 | **+0.043** | = |
| withnoise_withmotion | 0.707 | 0.727 | **+0.020** | ↓ |
| Motion (todos) | ~0.33 | ~0.32 | −0.004 | ↓ |

**Diagnóstico:** El modelo reduce el ruido térmico Riciano sintético (PSNR +12.8 dB) pero introduce un suavizado que Task 1A puede interpretar como patrón diferente de ruido. El scanner Hyperfine SWOOP 0.064T tiene ruido de RF, aliasing, y susceptibilidad magnética que la simulación Riciana no modela.

---

## Mejora 1 — Pérdida adversarial con Task 1A como discriminador (impacto: eliminar domain gap)

Añadir un término de pérdida que penaliza al denoiser cuando Task 1A detecta artefactos en la imagen mejorada.

### Concepto

```
L_total = 0.6 × L1(enhanced, clean)
        + 0.4 × (1 − SSIM(enhanced, clean))
        + λ_adv × L_adv(enhanced)

L_adv = BCE(task1a_preds(enhanced), target=0)
        para las clases Noise y Motion únicamente.
```

### Implementación

```python
# task_1b/losses.py — añadir al CombinedLoss existente

import sys
sys.path.append('..')
import torch
import torch.nn as nn
from task_1a.model import ArtifactClassifier


class CombinedLossWithAdversarial(nn.Module):
    def __init__(self, task1a_ckpt: str, lambda_adv: float = 0.1,
                 lambda_l1: float = 0.6, lambda_ssim: float = 0.4):
        super().__init__()
        self.lambda_adv  = lambda_adv
        self.lambda_l1   = lambda_l1
        self.lambda_ssim = lambda_ssim

        # Task 1A congelado como discriminador
        self.task1a = ArtifactClassifier(num_classes=7)
        self.task1a.load_state_dict(
            torch.load(task1a_ckpt, weights_only=False)
        )
        for p in self.task1a.parameters():
            p.requires_grad = False
        self.task1a.eval()

        # Índices de Noise y Motion en ARTIFACT_COLS
        # ['Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast', 'Distortion']
        self._noise_idx  = 0
        self._motion_idx = 4

    def forward(self, enhanced, clean):
        # Pérdida de reconstrucción
        l1_loss   = torch.mean(torch.abs(enhanced - clean))
        ssim_loss = 1 - self._ssim(enhanced, clean)

        # Pérdida adversarial: Task 1A debe ver 0 artefactos
        enhanced_3ch = enhanced.repeat(1, 3, 1, 1)  # (B,1,H,W) → (B,3,H,W)
        logits = self.task1a(enhanced_3ch)
        probs  = torch.sigmoid(logits)

        # Solo penalizar Noise y Motion
        target = torch.zeros_like(probs[:, [self._noise_idx, self._motion_idx]])
        adv_loss = nn.functional.binary_cross_entropy(
            probs[:, [self._noise_idx, self._motion_idx]], target
        )

        total = (self.lambda_l1   * l1_loss +
                 self.lambda_ssim * ssim_loss +
                 self.lambda_adv  * adv_loss)
        return total

    @staticmethod
    def _ssim(x, y, window_size=11):
        from pytorch_msssim import ssim
        return ssim(x, y, data_range=1.0, size_average=True)
```

### Cómo activar

```python
# En task_1b/02_train.py

from task_1b.losses import CombinedLossWithAdversarial

criterion = CombinedLossWithAdversarial(
    task1a_ckpt='task_1a/checkpoints/best_1a.pth',
    lambda_adv=0.1   # Empezar conservador; aumentar a 0.2 si Noise no baja
).cuda()
```

**Prerequisito:** Task 1A debe estar entrenado y el checkpoint disponible en `task_1a/checkpoints/best_1a.pth`.

**Tiempo estimado:** ~14h adicionales de entrenamiento (misma arquitectura, loss más costosa).

---

## Mejora 2 — Degradación sintética más realista (impacto: +2–4 dB PSNR en condiciones reales)

El ruido Riciano sintético no replica el ruido del Hyperfine SWOOP. Añadir componentes de ruido más realistas.

### Componentes adicionales

```python
# task_1b/augmentations.py

import numpy as np
import torch


def add_rician_noise(img: torch.Tensor, sigma_range=(0.03, 0.18)) -> torch.Tensor:
    """Ruido Riciano estándar (ya implementado en v1)."""
    sigma = np.random.uniform(*sigma_range)
    noise_r = torch.randn_like(img) * sigma
    noise_i = torch.randn_like(img) * sigma
    return torch.sqrt((img + noise_r) ** 2 + noise_i ** 2).clamp(0, 1)


def add_structured_noise(img: torch.Tensor,
                          stripe_prob: float = 0.3,
                          aliasing_prob: float = 0.2) -> torch.Tensor:
    """
    Simula ruido estructurado del scanner de bajo campo:
    - Bandas de RF (stripe noise) — perpendicular al eje de frecuencia
    - Aliasing (wrap-around) — ocurre cuando FOV < objeto
    """
    result = img.clone()
    B, C, H, W = result.shape

    # Bandas de RF horizontales
    if np.random.random() < stripe_prob:
        n_stripes = np.random.randint(1, 4)
        for _ in range(n_stripes):
            row = np.random.randint(H // 4, 3 * H // 4)
            width = np.random.randint(1, 4)
            amp   = np.random.uniform(0.05, 0.20)
            result[:, :, row:row+width, :] += amp

    # Aliasing (ghosting periódico)
    if np.random.random() < aliasing_prob:
        shift = np.random.randint(H // 3, 2 * H // 3)
        alpha = np.random.uniform(0.05, 0.15)
        result += alpha * torch.roll(result, shift, dims=-2)

    return result.clamp(0, 1)


def add_motion_ghosting(img: torch.Tensor,
                         ghost_prob: float = 0.5) -> torch.Tensor:
    """Ghosting de movimiento en k-space (ya en v1 — ampliar rango)."""
    if np.random.random() > ghost_prob:
        return img
    n_ghosts = np.random.randint(2, 8)      # v1: 2–6
    alpha    = np.random.uniform(0.03, 0.30) # v1: 0.04–0.25
    kspace   = torch.fft.fftn(img, dim=(-2, -1))
    for _ in range(n_ghosts):
        shift = np.random.randint(img.shape[-2] // 4, img.shape[-2] * 3 // 4)
        phase = np.random.uniform(0, 2 * np.pi)
        echo  = alpha * torch.exp(torch.tensor(1j * phase)) * torch.roll(kspace, shift, dims=-2)
        kspace = kspace + echo
    corrupted = torch.fft.ifftn(kspace, dim=(-2, -1)).real
    return corrupted.clamp(0, 1)
```

### Integración en dataset

```python
# task_1b/dataset.py — degradación compuesta (v2)

from task_1b.augmentations import add_rician_noise, add_structured_noise, add_motion_ghosting

def degrade_composite(img: torch.Tensor) -> torch.Tensor:
    """Pipeline de degradación v2: Riciano + structured + motion."""
    img = add_rician_noise(img, sigma_range=(0.02, 0.20))     # ampliado
    img = add_structured_noise(img, stripe_prob=0.3, aliasing_prob=0.2)
    img = add_motion_ghosting(img, ghost_prob=0.5)
    return img
```

---

## Mejora 3 — Arquitectura Restormer (impacto: +2–4 dB PSNR, mejor SSIM)

Restormer supera a ResUNet en tareas de denoising con pocos datos gracias a la atención en canales. Es el Nivel 2 planificado en el WORK_PLAN.

### Por qué ahora

- ResUNet v1 cumple el criterio PSNR ≥ 28 dB pero con apenas +3.24 dB de margen.
- Restormer reporta PSNR ≥ 34 dB en benchmarks de denoising de imágenes médicas.
- El domain gap identificado puede compensarse parcialmente con mejor capacidad arquitectónica.

### Instalación y configuración mínima

```bash
pip install einops
```

```python
# task_1b/model_restormer.py

import torch
import torch.nn as nn
from einops import rearrange


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention."""
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, 3,
                                     stride=1, padding=1, groups=channels * 3)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out  = (attn @ v)
        out  = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.proj(out)


class RestormerBlock(nn.Module):
    def __init__(self, channels, num_heads, ffn_expansion=2.66):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden * 2, channels, 1, bias=False),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # Atención
        residual = x
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm1(x_flat)
        x = x_flat.transpose(1, 2).view(b, c, h, w)
        x = self.attn(x) + residual
        # FFN
        residual = x
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm2(x_flat)
        x = x_flat.transpose(1, 2).view(b, c, h, w)
        x = self.ffn(x) + residual
        return x


class RestormerSmall(nn.Module):
    """
    Restormer reducido para training con 79 casos + degradación sintética.
    ~8M parámetros (vs 26M del Restormer original).
    """
    def __init__(self, inp_channels=1, out_channels=1,
                 dim=32, num_blocks=(2, 3, 3, 4), heads=(1, 2, 4, 8)):
        super().__init__()
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=False)

        self.encoder = nn.ModuleList([
            nn.Sequential(*[RestormerBlock(dim * (2**i), heads[i])
                           for _ in range(num_blocks[i])])
            for i in range(len(num_blocks) - 1)
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(dim * (2**i), dim * (2**(i+1)), 2, 2, bias=False)
            for i in range(len(num_blocks) - 1)
        ])
        self.bottleneck = nn.Sequential(
            *[RestormerBlock(dim * (2**(len(num_blocks)-1)), heads[-1])
              for _ in range(num_blocks[-1])]
        )
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(dim * (2**(i+1)), dim * (2**i), 2, 2, bias=False)
            for i in range(len(num_blocks)-2, -1, -1)
        ])
        self.decoder = nn.ModuleList([
            nn.Sequential(*[RestormerBlock(dim * (2**(i+1)), heads[i])
                           for _ in range(num_blocks[i])])
            for i in range(len(num_blocks)-2, -1, -1)
        ])
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)
        skips = []
        for enc, down in zip(self.encoder, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoder, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.output(x)
```

### Configuración de entrenamiento para Restormer

```python
# Cambios en config.py para v2 con Restormer

MODEL = 'restormer'      # 'resunet' | 'restormer'
IMG_SIZE = 256
BATCH_SIZE = 4           # Restormer más pesado en memoria
LR = 1e-4                # AdamW
EPOCHS = 150
PATIENCE = 25
LOSS = 'L1+SSIM'         # Igual que v1
```

---

## Mejora 4 — Procesado pseudo-3D (contexto entre slices)

Los artefactos de movimiento son incoherencias entre slices adyacentes. Un modelo que vea ±2 slices como contexto puede corregirlos mejor.

```python
# task_1b/dataset.py — modo pseudo-3D

class SyntheticDenoiseDataset3D(torch.utils.data.Dataset):
    """
    Input:  5 slices consecutivos degradados → (B, 5, H, W)
    Target: slice central limpio           → (B, 1, H, W)
    """
    def __init__(self, clean_slices, context=2, **kwargs):
        super().__init__(**kwargs)
        self.slices  = clean_slices
        self.context = context
        # Índices válidos (con contexto suficiente en ambos extremos)
        self.valid_idx = list(range(context, len(clean_slices) - context))

    def __getitem__(self, idx):
        real_idx = self.valid_idx[idx]
        clean_center = self.slices[real_idx]
        neighbors = [self.slices[real_idx + d]
                     for d in range(-self.context, self.context + 1)]
        degraded = torch.stack([degrade(s) for s in neighbors], dim=0)  # (5, H, W)
        return degraded, clean_center.unsqueeze(0)
```

**Requiere modificar** la primera capa del modelo: `nn.Conv2d(5, dim, ...)` para aceptar 5 canales de entrada.

---

## Roadmap de mejoras

| Prioridad | Mejora | Tiempo | Impacto en PSNR | Impacto en feedback 1A |
|---|---|---|---|---|
| 1 | Degradación compuesta (stripes + aliasing) | Modificación dataset, ~14h reentrenamiento | +1–3 dB | Medio |
| 2 | Pérdida adversarial con Task 1A | ~14h reentrenamiento | +0–1 dB | **Alto — reduce Noise score** |
| 3 | Restormer (v2) | ~20h reentrenamiento | +2–4 dB | Medio-alto |
| 4 | Pseudo-3D (contexto ±2 slices) | Modificación arquitectura, ~20h | +1–2 dB en Motion | Medio para Motion |

**Objetivo v2:** PSNR ≥ 33 dB + Noise score de Task 1A se reduce (no aumenta) tras el denoising.

### Versión recomendada (v2)

```
v2 = Degradación compuesta (Mejora 2) + Pérdida adversarial Task 1A (Mejora 1)
   + ResUNet v1 como backbone (mantener — no cambiar arquitectura aún)
   + Verificar en feedback Task 1A antes de pasar a Restormer
```

---

## Archivos a modificar

| Archivo | Cambio |
|---|---|
| [task_1b/augmentations.py](../task_1b/augmentations.py) | Crear — `add_structured_noise`, actualizar `add_motion_ghosting` |
| [task_1b/losses.py](../task_1b/losses.py) | Añadir `CombinedLossWithAdversarial` |
| [task_1b/model_restormer.py](../task_1b/model_restormer.py) | Crear — `RestormerSmall` |
| [task_1b/dataset.py](../task_1b/dataset.py) | Usar `degrade_composite`, opción pseudo-3D |
| [task_1b/02_train.py](../task_1b/02_train.py) | `VERSION = "v2"`, nueva loss, nueva augmentación |
| [task_1b/config.py](../task_1b/config.py) | `MODEL`, `LAMBDA_ADV`, `USE_STRUCTURED_NOISE` |
