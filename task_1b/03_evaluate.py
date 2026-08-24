"""Task 1b — Evaluation.

Three evaluation modes:
  A) Synthetic PSNR — apply the trained denoiser to synthetic noisy val images
     (uses the same SyntheticDenoiseDataset in eval mode with fixed seed).

  B) Real-data quality check — run the trained Task 1a classifier on the enhanced
     outputs of withnoise/withmotion images and report Noise + Motion scores.

  C) Visual spot-check — save a small PDF/grid of (degraded | enhanced | clean)
     triples to results/03_spot_check/ for manual inspection.

Results written to results/03_evaluation.json (or 03_evaluation_v2.json for v2).

Usage:
  python task_1b/03_evaluate.py                   # v1: best_1b.pth
  python task_1b/03_evaluate.py --ckpt best_1b_v2.pth  # v2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_1b.config import (
    CSV_NONOISE_NOMOTION, CSV_WITHNOISE_NOMOTION,
    CSV_NONOISE_WITHMOTION, CSV_WITHNOISE_WITHMOTION,
    CHECKPOINTS_DIR, RESULTS_DIR, BATCH_SIZE, NUM_WORKERS, RANDOM_SEED,
    MIN_PSNR, MIN_SSIM, TRAIN_DIR,
)
from task_1b.dataset import SyntheticDenoiseDataset, build_split
from task_1b.model import build_model
from task_1b.utils.metrics import psnr_torch, ssim_torch

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPOT_DIR = RESULTS_DIR / '03_spot_check'
SPOT_DIR.mkdir(exist_ok=True)


def load_model(device, ckpt_name: str = 'best_1b.pth'):
    ckpt_path = CHECKPOINTS_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    model = build_model(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val PSNR={ckpt['val_psnr']:.2f} dB)")
    return model


# ── Mode A: Synthetic PSNR ────────────────────────────────────────────────────

@torch.no_grad()
def eval_synthetic(model, device) -> dict:
    print("\n── Mode A: Synthetic PSNR evaluation ───────────────────")
    if not CSV_NONOISE_NOMOTION.exists():
        print("  CSV not found — skipping.")
        return {}

    _, val_ds = build_split(CSV_NONOISE_NOMOTION, val_split=0.20, seed=RANDOM_SEED)
    if len(val_ds) == 0:
        print("  No val samples — skipping.")
        return {}

    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)

    psnr_input_list, psnr_output_list = [], []
    ssim_input_list, ssim_output_list = [], []

    for degraded, clean in loader:
        degraded, clean = degraded.to(device), clean.to(device)
        enhanced = model(degraded)
        if enhanced.shape != clean.shape:
            enhanced = enhanced[:, :, :clean.shape[2], :clean.shape[3]]
            degraded = degraded[:, :, :clean.shape[2], :clean.shape[3]]

        psnr_input_list.append(psnr_torch(degraded, clean))
        psnr_output_list.append(psnr_torch(enhanced, clean))
        ssim_input_list.append(ssim_torch(degraded, clean).item())
        ssim_output_list.append(ssim_torch(enhanced, clean).item())

    avg_psnr_in  = float(np.mean(psnr_input_list))
    avg_psnr_out = float(np.mean(psnr_output_list))
    avg_ssim_in  = float(np.mean(ssim_input_list))
    avg_ssim_out = float(np.mean(ssim_output_list))

    print(f"  Input  PSNR: {avg_psnr_in:.2f} dB  | SSIM: {avg_ssim_in:.4f}")
    print(f"  Output PSNR: {avg_psnr_out:.2f} dB  | SSIM: {avg_ssim_out:.4f}")
    print(f"  PSNR improvement: +{avg_psnr_out - avg_psnr_in:.2f} dB")

    psnr_pass = avg_psnr_out >= MIN_PSNR
    ssim_pass = avg_ssim_out >= MIN_SSIM
    print(f"  PSNR criterion (>={MIN_PSNR} dB): {'PASS' if psnr_pass else 'FAIL'}")
    print(f"  SSIM criterion (>={MIN_SSIM}):    {'PASS' if ssim_pass else 'FAIL'}")

    return {
        'psnr_input': avg_psnr_in,
        'psnr_output': avg_psnr_out,
        'psnr_improvement': avg_psnr_out - avg_psnr_in,
        'ssim_input': avg_ssim_in,
        'ssim_output': avg_ssim_out,
        'psnr_pass': psnr_pass,
        'ssim_pass': ssim_pass,
        'n_batches': len(loader),
    }


# ── Mode B: Task 1a quality feedback on real noisy images ─────────────────────

@torch.no_grad()
def eval_task1a_quality(model, device) -> dict:
    print("\n── Mode B: Task 1a quality feedback ─────────────────────")
    from task_1b.config import TASK_1A_CHECKPOINT, TASK_1A_THRESHOLDS, ARTIFACT_COLS

    if not TASK_1A_CHECKPOINT.exists():
        print("  Task 1a checkpoint not found — skipping.")
        return {}

    # Load Task 1a model
    try:
        import timm
        import torch.nn as nn

        class _Clf(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model('densenet169', pretrained=False, num_classes=0)
                feat_dim = self.backbone.num_features
                self.head = nn.Sequential(
                    nn.Dropout(0.4), nn.Linear(feat_dim, 512),
                    nn.ReLU(inplace=True), nn.Dropout(0.3),
                    nn.Linear(512, 7),
                )
            def forward(self, x):
                return self.head(self.backbone(x))

        clf = _Clf().to(device)
        ckpt = torch.load(TASK_1A_CHECKPOINT, map_location=device, weights_only=False)
        state = ckpt.get('model_state', ckpt)
        clf.load_state_dict(state)
        clf.eval()
    except Exception as e:
        print(f"  Could not load Task 1a model: {e} — skipping.")
        return {}

    from task_1b.utils.io import load_normalized_slices, find_image_path
    from task_1b.dataset import SyntheticDenoiseDataset
    import pandas as pd
    import torchvision.transforms as T
    from PIL import Image as PILImage

    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results_per_partition = {}
    noisy_csvs = {
        'withnoise_nomotion': CSV_WITHNOISE_NOMOTION,
        'nonoise_withmotion': CSV_NONOISE_WITHMOTION,
        'withnoise_withmotion': CSV_WITHNOISE_WITHMOTION,
    }

    for label, csv_path in noisy_csvs.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        noise_scores_before, noise_scores_after = [], []
        motion_scores_before, motion_scores_after = [], []

        for fn in df['filename'][:30]:  # sample up to 30 per partition
            p = find_image_path(fn, TRAIN_DIR)
            if p is None:
                continue
            try:
                norm_slices, _, _ = load_normalized_slices(p)
                mid = len(norm_slices) // 2
                s = norm_slices[mid]  # native-resolution noisy slice

                # Enhance: resize to IMG_SIZE, run model, result stays at IMG_SIZE
                resized = SyntheticDenoiseDataset._resize(s)
                inp = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device)
                enh_np = model(inp).squeeze(0).squeeze(0).cpu().numpy()

                for arr, score_list_n, score_list_m in [
                    (s,      noise_scores_before, motion_scores_before),
                    (enh_np, noise_scores_after,  motion_scores_after),
                ]:
                    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                    rgb = np.stack([arr_u8, arr_u8, arr_u8], axis=-1)
                    pil = PILImage.fromarray(rgb).resize((224, 224))
                    tensor = to_tensor(pil).unsqueeze(0).to(device)
                    logits = clf(tensor)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    score_list_n.append(float(probs[0]))   # Noise idx=0
                    score_list_m.append(float(probs[4]))   # Motion idx=4
            except Exception:
                continue

        if noise_scores_before:
            avg_nb = float(np.mean(noise_scores_before))
            avg_na = float(np.mean(noise_scores_after))
            avg_mb = float(np.mean(motion_scores_before))
            avg_ma = float(np.mean(motion_scores_after))
            print(f"  {label}:")
            print(f"    Noise  score: {avg_nb:.3f} → {avg_na:.3f}")
            print(f"    Motion score: {avg_mb:.3f} → {avg_ma:.3f}")
            results_per_partition[label] = {
                'noise_before': avg_nb, 'noise_after': avg_na,
                'motion_before': avg_mb, 'motion_after': avg_ma,
                'n_images': len(noise_scores_before),
            }

    return results_per_partition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='best_1b.pth',
                        help='Checkpoint filename inside checkpoints/ (default: best_1b.pth)')
    args = parser.parse_args()

    # Derive output filename from checkpoint name: best_1b.pth → 03_evaluation.json
    # best_1b_v2.pth → 03_evaluation_v2.json
    suffix = args.ckpt.replace('best_1b', '').replace('.pth', '')
    out_name = f'03_evaluation{suffix}.json'

    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    model = load_model(device, ckpt_name=args.ckpt)

    synth_results = eval_synthetic(model, device)
    qual_results  = eval_task1a_quality(model, device)

    overall_pass = synth_results.get('psnr_pass', False) and synth_results.get('ssim_pass', False)

    results = {
        'checkpoint': args.ckpt,
        'overall_pass': overall_pass,
        'synthetic_eval': synth_results,
        'task1a_quality_feedback': qual_results,
    }
    out = RESULTS_DIR / out_name
    out.write_text(json.dumps(results, indent=2))
    status = 'PASS' if overall_pass else 'FAIL'
    print(f"\n{status} -- Results saved to {out}")


if __name__ == '__main__':
    main()
