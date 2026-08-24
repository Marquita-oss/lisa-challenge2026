# LISA Challenge 2026 — Task 1b (Quality Improvement) — imagen de submission.
#
# IMPORTANTE: construir desde la RAIZ del repo (contexto = raiz):
#   docker build -f task_1b/docker/Dockerfile -t lisa-task1b:v1 .
#
# Pipeline liviano, CPU-only (sin PyTorch/GPU): replica la mejor submission local
# conocida ("native_clean"). No descarga nada en tiempo de ejecucion.

FROM python:3.11-slim

WORKDIR /solution

COPY task_1b/docker/requirements.txt /solution/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /solution/requirements.txt

COPY task_1b/docker/run_inference.py /solution/run_inference.py

ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/solution/run_inference.py"]
