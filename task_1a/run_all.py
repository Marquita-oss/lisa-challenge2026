"""
run_all.py — Orquestador del pipeline limpio Task 1A (end-to-end).

Ejecuta: entrenar cada backbone (k-fold) -> calibrar umbrales sobre OOF del ensemble
-> generar submission de 114 filas.

  python run_all.py                 # entrena B4 + ConvNeXt, calibra, predice
  python run_all.py --skip-train    # sólo calibrar + predecir (usa OOF/checkpoints ya guardados)
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BACKBONES

PY = sys.executable
HERE = Path(__file__).resolve().parent


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()

    labels = [lab for _, lab in BACKBONES]

    if not args.skip_train:
        for backbone, lab in BACKBONES:
            run([PY, 'train.py', '--label', lab, '--backbone', backbone])

    run([PY, 'calibrate.py', '--labels', *labels])

    specs = []
    for backbone, lab in BACKBONES:
        specs += ['--spec', f'{backbone}:best_{lab}_fold{{k}}.pth:256']
    run([PY, 'predict.py', *specs])

    print("\nPipeline completo. Subir LISA_LF_QC_predictions.csv a Synapse.")


if __name__ == '__main__':
    main()
