# Revisión del Paper Task 1B — LISA Challenge 2026

> **Nota de contexto (2026-08-19):** revisión de un `paper.tex` de Task 1B como
> documento standalone, en una ruta (`task_1b/Springer_Lecture_Notes_in_Computer_Science/`)
> que ya no existe — corresponde a una estructura de repositorio anterior a
> `paper_combined/`, el paper único de equipo que cubre las tres tareas y es el que
> efectivamente se envió a revisión. Los hallazgos críticos de fondo (weight decay,
> semillas distintas entre evaluaciones baseline/propuesto, SSIM no reportado) valen la
> pena revisarlos contra `paper_combined/paper.tex` §Methods/Results Task 1B para
> confirmar si siguen aplicando o ya se resolvieron al reescribir esa sección.

**Archivo revisado:** `task_1b/Springer_Lecture_Notes_in_Computer_Science/paper.tex`
**Fecha de revisión:** 2026-05-31
**Archivos cruzados:** `config.py`, `model.py`, `losses.py`, `results/02_training_v2.json`, `results/03_evaluation.json`, `results/03_evaluation_v2.json`

---

## Problemas Críticos

### 1. Discrepancia weight decay — código vs. paper

- **Paper §2.4:** `Adam (η = 2×10⁻⁴, weight decay 10⁻⁵)`
- **`config.py:45`:** `WEIGHT_DECAY = 1e-4`
- **Diferencia:** un orden de magnitud (10⁻⁵ vs 10⁻⁴)
- **Acción:** verificar cuál fue el valor real usado durante el entrenamiento y corregir el paper o el config.

---

### 2. `overall_pass: false` en el modelo propuesto — SSIM no reportado como fallo

`results/03_evaluation_v2.json`:
```json
"overall_pass": false,
"psnr_output": 27.681   // < 28 dB → falla en split Rician
"ssim_output": 0.7955   // < 0.80  → también falla, NO mencionado en el paper
```

El paper §4.1 menciona que PSNR falla en Rician pero pasa en physics-split (30.70 dB). Sin embargo **nunca menciona que SSIM = 0.796 también cae por debajo del umbral 0.80**. La celda "Pass" de la Tabla 1 muestra `$30.70$\,dB†`, lo que es visualmente confuso en una columna binaria.

**Acción:** agregar nota explícita sobre el fallo de SSIM en Rician, o rediseñar la columna Pass con ✓/✗ + footnote que aclare que el modelo propuesto fue evaluado en el split physics-based.

---

### 3. Valores "before" inconsistentes entre evaluaciones v1 y v2

Las evaluaciones de baseline (v1) y propuesto (v2) muestran valores "before" diferentes para las mismas particiones, indicando que se evaluaron sobre subconjuntos aleatorios distintos (n=30 con distinta semilla):

| Partición | `noise_before` v1 | `noise_before` v2 |
|---|---|---|
| withnoise\_nomotion | 0.686 | 0.585 |
| nonoise\_withmotion | 0.677 | 0.203 |
| withnoise\_withmotion | 0.707 | 0.577 |

Para que la comparación baseline vs. propuesto sea válida, ambas evaluaciones deben realizarse sobre exactamente el mismo subconjunto (misma semilla aleatoria). Como está ahora, la reducción porcentual reportada en Tabla 2 se calcula sobre distintos grupos de imágenes, lo cual invalida la comparación directa.

**Acción:** re-ejecutar `03_evaluate.py` fijando una semilla y el mismo índice de imágenes para ambos modelos, y actualizar Tabla 2 con los valores corregidos.

---

### 4. `goodfellow2014` definido en bibliografía pero no citado en el texto

`paper.tex:476`: `\bibitem{goodfellow2014}` existe pero no hay ningún `\cite{goodfellow2014}` en el cuerpo del paper. Springer/LaTeX genera advertencia y algunos sistemas de envío lo rechazan.

**Acción:** eliminar la entrada o citar explícitamente en algún párrafo relevante (e.g., al introducir el concepto de pérdida adversarial en §2.3).

---

## Problemas Moderados

### 5. Cita `ravi2024` — verificación requerida

```bibtex
\bibitem{ravi2024}
Ravi, D., et al.: Physics-informed data augmentation for robust MRI enhancement.
In: MICCAI 2024, LNCS, vol. 15003. Springer, Cham (2024)
```

Esta cita fue generada a partir del concepto descrito en el paper. El título exacto, nombre de autores, volumen LNCS y número de página **no han sido verificados**. El paper puede no existir con estos datos exactos.

**Acción:** buscar en Google Scholar / Springer con términos `"physics informed data augmentation MRI MICCAI 2024"` y reemplazar con los datos reales, o citar una referencia alternativa verificada para este concepto.

---

### 6. Rutas de figuras inconsistentes y frágiles para envío

Las figuras usan dos patrones distintos de ruta:

```latex
% pipeline — ruta relativa al .tex (OK)
\includegraphics[width=\textwidth]{pipeline_figure.pdf}

% resto de figuras — sube un nivel con ../
\includegraphics[width=\textwidth]{../results/paper_figures/fig1_training_curves.pdf}
```

El patrón `../results/...` funciona en compilación local pero rompe en plataformas de envío que normalizan el directorio raíz o comprimen solo el directorio del .tex.

**Acción:** copiar todos los PDFs de figuras a `Springer_Lecture_Notes_in_Computer_Science/figures/` y reemplazar todas las rutas con `{figures/figX_...pdf}`.

---

### 7. Todos los floats usan `[t]` — figuras se acumulan al final

Líneas afectadas: 146, 199, 260, 285, 327, 354, 382 en `paper.tex`. No hay `\usepackage{placeins}` ni `\FloatBarrier`. Con 5 figuras y 2 tablas, LaTeX no puede colocarlas todas en el top de páginas sucesivas y las descarga al final.

**Acción** (idéntica a la aplicada en Task 1A):
```latex
% En el preámbulo:
\usepackage{placeins}

% Cambiar todos [t] a [htbp]:
\begin{figure}[htbp]
\begin{table}[htbp]

% Agregar antes de cada \section:
\FloatBarrier
```

---

## Problemas Menores

### 8. Inconsistencia ortográfica: "artefact" vs "artifact"

El título y varias secciones usan "Artefact" (inglés británico), pero otras partes del mismo paper usan "artifact" (inglés americano). La mezcla es perceptible dentro del abstract.

**Acción:** unificar en una sola variante. Dado que el challenge es del ámbito médico con literatura predominantemente americana, se recomienda "artifact" en todo el documento, o "artefact" consistentemente si se prefiere la variante británica.

---

### 9. Tabla 1 — columna "Pass" visualmente ambigua

La celda `$30.70$\,dB$^{\dagger}$` en la columna "Pass" del modelo propuesto parece un número en una columna binaria. El lector espera ✓ o ✗.

**Sugerencia de rediseño:**
```latex
Proposed  & 27.68  & +9.28  & 0.796  & ✗ (✓ physics†) \\
```
O agregar una columna separada "PSNR@physics" para evitar la nota al pie.

---

## Lo que está correcto y bien fundamentado

| Aspecto | Detalle |
|---|---|
| Números Tabla 2 | Coinciden exactamente con `03_evaluation_v2.json` |
| Curva de entrenamiento | train_adv epoch 11 = 0.308, epoch 100 = 0.132 — exactamente lo descrito |
| Best PSNR propuesto | 30.703 dB en epoch 93 — verificado en `02_training_v2.json` |
| Índices adversariales | Noise=0, Motion=4 coherentes entre `config.py`, `losses.py` y ecuaciones |
| Arquitectura ResUNet | Descripción en paper es fiel a `model.py` (4 niveles, residual global, InstanceNorm) |
| Warmup adversarial | 10 epochs descrito y confirmado en historial de entrenamiento |
| Dominio gap (Baseline) | `03_evaluation.json` confirma que baseline aumenta noise scores en todas las particiones — coherente con la Figura 7 descrita |

---

## Resumen de acciones por prioridad

| Prioridad | Ítem | Archivo |
|---|---|---|
| 🔴 Crítica | Verificar y corregir weight decay (10⁻⁴ vs 10⁻⁵) | `config.py` / `paper.tex §2.4` |
| 🔴 Crítica | Mencionar fallo de SSIM=0.796 en Tabla 1 | `paper.tex Tabla 1` |
| 🔴 Crítica | Re-evaluar ambos modelos con mismo subconjunto fijo (misma semilla) | `03_evaluate.py` |
| 🔴 Crítica | Eliminar o citar `goodfellow2014` | `paper.tex bibliography` |
| 🟡 Moderada | Verificar `ravi2024` en Google Scholar | `paper.tex bibliography` |
| 🟡 Moderada | Mover figuras a `figures/`, eliminar rutas `../` | `paper.tex` + archivos PDF |
| 🟡 Moderada | Cambiar `[t]`→`[htbp]`, agregar `placeins` + `\FloatBarrier` | `paper.tex` |
| 🟢 Menor | Unificar "artifact" vs "artefact" en todo el documento | `paper.tex` |
| 🟢 Menor | Rediseñar columna "Pass" en Tabla 1 | `paper.tex Tabla 1` |
