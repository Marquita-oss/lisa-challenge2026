"""Task 1b — Inferencia con el modelo LF->CISO (canvas nativo 160, sin uint8).

Mirror exacto del entrenamiento (02_train_lf2ciso.py):
  - normalizacion per-slice [0,1]
  - padding a 160x160, modelo, crop a (H,W) nativo
  - TTA flips (4 vistas) promediadas
Salida float32 en escala de la LF (las metricas normalizan igual).

Uso:
  python task_1b/04c_predict_lf2ciso.py --ckpt best_1b_lf2ciso.pth
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    'predict_orig', str(Path(__file__).resolve().parent / '04_predict_submission.py'))
predict_orig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_orig)
build_output_filename = predict_orig.build_output_filename
create_zip = predict_orig.create_zip

from task_1b.config import VAL_SINGLE_DIR, VAL_COMPLETE_DIR, SUBMISSION_DIR, CHECKPOINTS_DIR
from task_1b.model import build_model

CANVAS = 160
FLIPS = [
    (lambda a: a,            lambda a: a),
    (lambda a: np.fliplr(a), lambda a: np.fliplr(a)),
    (lambda a: np.flipud(a), lambda a: np.flipud(a)),
    (lambda a: np.flipud(np.fliplr(a)), lambda a: np.fliplr(np.flipud(a))),
]


def pad_to(a, size=CANVAS):
    H, W = a.shape
    out = np.zeros((size, size), np.float32)
    out[:H, :W] = a
    return out


def load_model(device, ckpt_name):
    ckpt_path = CHECKPOINTS_DIR / ckpt_name
    model = build_model(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state'])
    model.eval()
    print(f"Loaded {ckpt_name} (epoch {ck.get('epoch','?')}, val PSNR={ck.get('val_psnr',float('nan')):.2f})")
    return model


@torch.no_grad()
def enhance_volume(model, nii_path, device, use_tta=True):
    img_obj = nib.load(str(nii_path))
    vol = img_obj.get_fdata(dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[:, :, None]; thin = 2
    else:
        thin = int(np.argmin(vol.shape))
    flips = FLIPS if use_tta else FLIPS[:1]
    out_slices = []
    for i in range(vol.shape[thin]):
        idx = [slice(None)] * 3; idx[thin] = i
        s_raw = vol[tuple(idx)].copy()
        p1 = float(np.percentile(s_raw, 1)); p99 = float(np.percentile(s_raw, 99))
        denom = p99 - p1
        if denom < 1e-8:
            out_slices.append(s_raw.astype(np.float32)); continue
        s_norm = np.clip((s_raw - p1) / denom, 0, 1).astype(np.float32)
        H, W = s_norm.shape
        variants = [pad_to(np.ascontiguousarray(f(s_norm))) for f, _ in flips]
        batch = torch.from_numpy(np.stack(variants)[:, None]).to(device)
        outs = model(batch).squeeze(1).cpu().numpy()
        acc = np.zeros((H, W), np.float32)
        for k, (_, inv) in enumerate(flips):
            acc += np.ascontiguousarray(inv(outs[k][:H, :W]))
        out_norm = np.clip(acc / len(flips), 0, 1)
        out_slices.append((out_norm * denom + p1).astype(np.float32))
    enhanced = np.stack(out_slices, axis=thin)
    return enhanced, img_obj


def process_dir(model, src, device, staging, skip_ciso, use_tta=True):
    files = [f for f in sorted(src.rglob('*.nii.gz'))
             if (not skip_ciso) or '_ciso' not in f.name.lower()]
    staging.mkdir(parents=True, exist_ok=True)
    saved = []
    for k, p in enumerate(files, 1):
        name = build_output_filename(p.name)
        enh, img = enhance_volume(model, p, device, use_tta=use_tta)
        out = nib.Nifti1Image(enh, img.affine, img.header)
        out.header.set_data_dtype(np.float32)
        nib.save(out, str(staging / name))
        saved.append(staging / name)
        if k % 10 == 0 or k == len(files):
            print(f"  [{k}/{len(files)}] {p.name} -> {name}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='best_1b_lf2ciso.pth')
    ap.add_argument('--label', default='lf2ciso')
    ap.add_argument('--tta', action='store_true', help='activar TTA flips (default off)')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device, args.ckpt)
    staging = SUBMISSION_DIR / f'submission_ready_{args.label}'
    t0 = time.time()
    saved = []
    if VAL_SINGLE_DIR.exists():
        print("\n-- single_plane --")
        saved += process_dir(model, VAL_SINGLE_DIR, device, staging, skip_ciso=False, use_tta=args.tta)
    if VAL_COMPLETE_DIR.exists():
        print("\n-- complete --")
        saved += process_dir(model, VAL_COMPLETE_DIR, device, staging, skip_ciso=True, use_tta=args.tta)
    print(f"\nTotal: {len(saved)}  ({time.time()-t0:.0f}s)")
    root = Path(__file__).resolve().parent.parent
    zp = root / f'LISA_enhanced_predictions_{args.label}.zip'
    create_zip(saved, zp)
    print(f"Staging: {staging}")
    print(f"ZIP (float32, convertir a uint16 con pack_uint16.py antes de subir): {zp}")


if __name__ == '__main__':
    main()
