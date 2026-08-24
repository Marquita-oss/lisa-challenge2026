# LISA Challenge 2026 — Task 1a (Quality Control) — imagen de submission.
#
# IMPORTANTE: construir desde la RAIZ del repo (contexto = raiz), no desde task_1a/docker/:
#   docker build -f task_1a/docker/Dockerfile -t lisa-task1a:v1 .
#
# El contenedor NO descarga nada en tiempo de ejecucion (entorno offline en el
# sistema de evaluacion): todos los pesos (pretrained=False) y checkpoints ya
# estan copiados dentro de la imagen en build time.

FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg-dev zlib1g-dev libpng-dev ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /solution

COPY task_1a/docker/requirements.txt /solution/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /solution/requirements.txt

# Pesos primero (capa pesada, ~1.3 GB): asi editar el codigo mas abajo no invalida
# esta capa en rebuilds incrementales.
# Umbrales calibrados (OOF)
COPY task_1a/results/thresholds.json /solution/thresholds.json

# Checkpoints del ensemble (EfficientNet-B4 + ConvNeXt-Small, 5 folds c/u)
RUN mkdir -p /solution/checkpoints
COPY task_1a/checkpoints/best_effb4_fold0.pth \
     task_1a/checkpoints/best_effb4_fold1.pth \
     task_1a/checkpoints/best_effb4_fold2.pth \
     task_1a/checkpoints/best_effb4_fold3.pth \
     task_1a/checkpoints/best_effb4_fold4.pth \
     task_1a/checkpoints/best_convnexts_fold0.pth \
     task_1a/checkpoints/best_convnexts_fold1.pth \
     task_1a/checkpoints/best_convnexts_fold2.pth \
     task_1a/checkpoints/best_convnexts_fold3.pth \
     task_1a/checkpoints/best_convnexts_fold4.pth \
     /solution/checkpoints/

# Codigo (capa liviana, cambia mas seguido)
COPY task_1a/docker/model.py /solution/model.py
COPY task_1a/docker/run_inference.py /solution/run_inference.py

ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/solution/run_inference.py"]
