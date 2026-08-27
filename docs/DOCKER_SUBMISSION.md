# LISA 2026 — Submission Docker (Testing Phase, deadline 20 julio 2026 23:59 UTC)

Este documento cubre Task 1a y Task 1b. **Task 2 ya lo armo y subio Gabriel por
separado** (`task_2/docker/` si esta sincronizado en el repo) — confirmado
enviado.

Proyecto Synapse del equipo: **syn76374434** (un solo proyecto para las 3
imagenes; cada tarea es un repo Docker distinto con su propia evaluation queue).

Reglas clave de la fase de testing (instrucciones oficiales + confirmado por
Gabriel a partir de haber enviado Task 2):

- Solo pueden participar equipos que enviaron short-paper y con Data Agreement
  individual firmado por cada autor.
- Limite de **2 submissions ACCEPTED por equipo por tarea**; para el ranking final
  cuenta la **ultima submission de cada queue** (se puede reenviar si mejoras
  algo). Las que salen `Invalid` **no** cuentan contra el cupo — usalas para
  iterar sin miedo.
- Memoria del contenedor: **24 GB** maximo.
- El sistema corre la imagen asi (confirmado, no inferido):
  ```
  docker run --rm --gpus all --network none -v datos:/input:ro -v salida:/output mi-imagen
  ```
  Implicaciones duras:
  - `ENTRYPOINT` **sin argumentos** — ambos Dockerfiles ya cumplen esto.
  - `--network none`: **cero red en runtime**, no solo "no descargar". Nada de
    `pretrained=True`, `torch.hub`, `from_pretrained`. Ambas imagenes cargan
    arquitectura con `pretrained=False` y pesos locales copiados en build time.
  - Sin `--shm-size`: el `/dev/shm` por defecto de Docker son 64 MB. Un
    `DataLoader` con `num_workers>0` puede morir con memoria compartida
    ("Background workers died"). Ambos scripts usan `num_workers=0`.
  - `/input` es **read-only**, `/output` es donde se lee la submission final —
    solo lo que quede ahi cuenta.
- El contenedor debe escribir `.csv` o `.nii.gz` **directamente dentro de
  `/output`**. **No comprimir en `.zip`**.
- Cada caso se procesa con try/except individual — si un archivo falla, no se
  pierde toda la submission (Task 1a rellena con fila de ceros y loguea el
  error; Task 1b omite ese archivo de salida y loguea el error, pero el
  contenedor sigue y termina con exit code 0 mientras al menos un caso se haya
  procesado bien).

## Convencion de nombres real (confirmada por el equipo, NO inferida)

Entradas — 3 vistas por caso, planas en `/input`, sin subcarpetas:
```
/input/LISA_TESTING_0001_LF_axi.nii.gz
/input/LISA_TESTING_0001_LF_cor.nii.gz
/input/LISA_TESTING_0001_LF_sag.nii.gz
```

Salidas:
- **Task 1a** (114 casos esperados): un unico `/output/LISA_LF_QC_predictions.csv`
  con columnas `patient_id,Noise,Zipper,Positioning,Banding,Motion,Contrast,Distortion`
  (heredadas del pipeline ya usado en fases anteriores — no las reinvente, pero
  si el wiki de la fase de testing especifica otro esquema de columnas hay que
  ajustarlo). `patient_id` se extrae como `LISA_LF_{ID}` tomando el numero que
  precede a `_LF_` en el nombre de archivo, sin asumir la palabra intermedia
  (`TESTING`, `VALIDATION`, etc.).
- **Task 1b** (50 casos esperados): un archivo por input, mismo nombre + sufijo
  `_enhanced` antes de la extension — **sin reformatear nada mas**:
  ```
  LISA_TESTING_0001_LF_axi.nii.gz -> LISA_TESTING_0001_LF_axi_enhanced.nii.gz
  ```
  Geometria (shape/spacing/affine) identica al input — verificado.

---

## 0. Instalar Docker Desktop (Windows 11)

Esta maquina no tiene Docker instalado. Pasos:

1. Descargar el instalador: https://www.docker.com/products/docker-desktop/
2. Ejecutar el instalador. Si pregunta, dejar **WSL 2** como backend (recomendado
   en Windows 11; se instala automaticamente si falta).
3. Reiniciar si lo pide.
4. Abrir Docker Desktop y esperar a que el icono de la ballena quede estable
   (`Engine running`).
5. Verificar en una terminal nueva:
   ```
   docker --version
   docker info
   ```
6. Si vas a usar GPU dentro del contenedor (recomendado para Task 1a — Task 1b no
   necesita GPU), confirmar que Docker Desktop tiene habilitado el soporte GPU
   (Settings → Resources → WSL Integration, y tener drivers NVIDIA actualizados
   con soporte WSL2 CUDA). Probar con:
   ```
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
   Si esto falla, las imagenes igual funcionan por CPU (`torch.cuda.is_available()`
   cae a CPU automaticamente en `run_inference.py` de Task 1a), solo mas lento.

Avisame cuando `docker --version` funcione y sigo con el build/test desde aca.

---

## 1. Build

Desde la **raiz del repo** (importante — el contexto tiene que ser la raiz para
que el `COPY` de checkpoints funcione sin duplicarlos):

```bash
docker build -f task_1a/docker/Dockerfile -t lisa-task1a:v1 .
docker build -f task_1b/docker/Dockerfile -t lisa-task1b:v1 .
```

Un `.dockerignore` en la raiz ya excluye `data/`, `nnunet_workspace/`, zips y
checkpoints no usados para que el build no tarde enviando gigas de contexto
innecesario.

Tamanos esperados aproximadamente:
- `lisa-task1a`: ~3-4 GB (imagen base CUDA + ensemble de 10 checkpoints, ~1.3 GB)
- `lisa-task1b`: ~200-300 MB (Python slim, sin GPU, sin checkpoints)

---

## 2. Smoke test local (antes de gastar submissions)

Usa un subconjunto pequeno de `data/val/single_plane/` como `/input` falso.

### Task 1a

```bash
mkdir -p /tmp/fake_input_1a /tmp/fake_output_1a
cp data/val/single_plane/1001/*.nii.gz data/val/single_plane/1002/*.nii.gz /tmp/fake_input_1a/

docker run --rm --gpus all \
  -v /tmp/fake_input_1a:/input:ro \
  -v /tmp/fake_output_1a:/output:rw \
  lisa-task1a:v1

cat /tmp/fake_output_1a/LISA_LF_QC_predictions.csv
```

Verificar: el CSV tiene columnas `patient_id,Noise,Zipper,Positioning,Banding,Motion,Contrast,Distortion`,
valores en `{0,1,2}`, una fila por archivo de entrada.

### Task 1b

```bash
mkdir -p /tmp/fake_input_1b /tmp/fake_output_1b
cp data/val/single_plane/1001/*.nii.gz data/val/single_plane/1002/*.nii.gz /tmp/fake_input_1b/

docker run --rm \
  -v /tmp/fake_input_1b:/input:ro \
  -v /tmp/fake_output_1b:/output:rw \
  lisa-task1b:v1

ls -la /tmp/fake_output_1b/
```

Verificar: aparecen archivos `LISA_VALIDATION_{ID}_{plano}_enhanced.nii.gz` (SIN
zip), mismo shape/affine que el input (podes chequear con
`python -c "import nibabel as nib; print(nib.load('...').shape, nib.load('...').affine)"`
comparando input vs output).

Si algo falla, revisa el log (`docker run` imprime todo por stdout — ambos
`run_inference.py` loguean INPUT_DIR, conteo de archivos, progreso y errores por
archivo). En PowerShell reemplaza `/tmp/...` por una ruta tipo
`C:\Users\rmarcar\AppData\Local\Temp\fake_input_1a` y usa `-v`
con la ruta de Windows (Docker Desktop la traduce).

---

## 3. Verificar limites de memoria (24 GB)

```bash
docker run --rm --gpus all --memory=24g --memory-swap=24g \
  -v /tmp/fake_input_1a:/input:ro -v /tmp/fake_output_1a:/output:rw \
  lisa-task1a:v1
```

Repetir para Task 1b (deberia usar muchisimo menos, es CPU-only sin batching de
modelos).

---

## 4. Push a Synapse Docker Registry y submission

Sustituye `synXXXXXXX` por el Synapse Project ID del equipo (ver
`docs/` o memoria del proyecto: `syn75277286`, proyecto `prosis-team`) y elige un
nombre de repo por tarea, por ejemplo `lisa2026-task1a` / `lisa2026-task1b`.

```bash
docker login docker.synapse.org
# usuario: tu username de Synapse
# password: tu Synapse Personal Access Token (no la password de la cuenta)

docker tag lisa-task1a:v1 docker.synapse.org/syn75277286/lisa2026-task1a:v1
docker push docker.synapse.org/syn75277286/lisa2026-task1a:v1

docker tag lisa-task1b:v1 docker.synapse.org/syn75277286/lisa2026-task1b:v1
docker push docker.synapse.org/syn75277286/lisa2026-task1b:v1
```

Luego, desde la pagina del proyecto en Synapse (pestana **Docker**), el
repositorio subido deberia aparecer listado. Desde ahi (o desde la pagina de la
evaluation queue de cada tarea) usar **"Submit Docker Repository"** apuntando al
tag recien subido, seleccionando la queue correspondiente a Task 1a / Task 1b.

**Importante sobre el cupo de 2 ACCEPTED**: cada `docker push` de un nuevo tag
NO consume submissions — solo consume cupo cuando la sometes a evaluacion via
"Submit" y el sistema la ejecuta y la marca `Accepted`. Si sale `Invalid`, revisa
`docs/DOCKER_SUBMISSION.md` seccion 2 (smoke test) antes de volver a intentar; las
`Invalid` no cuentan contra el limite, pero cada intento consume tiempo hasta el
deadline de hoy.

---

## 5. Notas sobre las decisiones tomadas

- **Task 1a**: ensemble EfficientNet-B4 + ConvNeXt-Small (5 folds cada uno) + TTA
  x8 + umbrales calibrados por F2 (`task_1a/results/thresholds.json`). Es el mismo
  pipeline que ya genero `LISA_LF_QC_predictions.csv` en la raiz del repo (score
  interno ~0.82-0.85 segun memoria del proyecto).
- **Task 1b**: pipeline "native_clean" (clip por-slice a percentiles [p1,p99] +
  empaquetado uint16). **No usa ninguno de los denoisers entrenados** (v1/v2/v3,
  GAN, perceptual) porque localmente todos fueron rechazados: alucinan detalle y
  empeoran FID/BRISQUE frente al passthrough simple (ver memoria del proyecto,
  `docs/REVIEW_TASK_1B.md`, `docs/IMPROVEMENTS_TASK_1B.md`). Si preferis probar
  con un modelo entrenado igual, decime y adapto `task_1b/docker/run_inference.py`
  para cargar `best_1b.pth` (mismo patron que `task_1a/docker`), pero no es lo que
  recomiendo dado el historial ya documentado.
