"""Task 1B — GradCAM analysis: v1 vs v2 artifact removal comparison.

For each partition (withnoise_nomotion, nonoise_withmotion, withnoise_withmotion):
  1. Sample N_SAMPLES image slices.
  2. Enhance with Task 1B v1 (best_1b.pth) and v2 (best_1b_v2.pth).
  3. Apply GradCAM on the frozen Task 1A classifier (DenseNet169) for the
     Noise class and the Motion class — before and after each enhancement.
  4. Generate side-by-side comparison figures:
       Rows:    Original | v1 enhanced | v2 enhanced
       Columns: Image | GradCAM Noise | GradCAM Motion
  5. Save aggregate score table to results/gradcam/summary.json.

GradCAM target layer: clf.backbone.features.norm5
  (final BatchNorm after denseblock4 — last spatial feature map before global pool)

Key question answered:
  v1 enhanced → GradCAM activations stay high (Noise/Motion still present)
  v2 enhanced → GradCAM activations drop (adversarial loss removed the features
                that Task 1A uses to detect artifacts)

Usage:
  python task_1b/05_gradcam_analysis.py
  python task_1b/05_gradcam_analysis.py --samples 10
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # non-interactive — safe on Windows without display
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_1b.config import (
    CSV_WITHNOISE_NOMOTION, CSV_NONOISE_WITHMOTION, CSV_WITHNOISE_WITHMOTION,
    CHECKPOINTS_DIR, RESULTS_DIR, TASK_1A_CHECKPOINT, TRAIN_DIR,
)
from task_1b.model import build_model
from task_1b.dataset import SyntheticDenoiseDataset
from task_1b.utils.io import load_normalized_slices, find_image_path

# ── Constants ─────────────────────────────────────────────────────────────────

IMG_CLF = 224   # Task 1A classifier input size
IMG_DEN = 256   # Task 1B denoiser input size (= IMG_SIZE in config)

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

NOISE_IDX  = 0   # index in Task 1A output vector
MOTION_IDX = 4

GRADCAM_DIR = RESULTS_DIR / 'gradcam'
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


# ── GradCAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    Hooks into `target_layer` to capture:
      - forward:  spatial activation maps  A  (B, C, h, w)
      - backward: gradients of the loss w.r.t. those maps  dL/dA

    CAM = ReLU( sum_c( mean_{h,w}(dL/dA_c) · A_c ) )
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations = None
        self._gradients   = None
        self._fwd = target_layer.register_forward_hook(self._hook_fwd)
        self._bwd = target_layer.register_full_backward_hook(self._hook_bwd)

    def _hook_fwd(self, _module, _inp, out):
        self._activations = out

    def _hook_bwd(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0]

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """Compute GradCAM map.

        x: (1, 3, 224, 224) on device, requires_grad=True.
        Returns: (h, w) numpy array in [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(x)                         # forward → _hook_fwd fires
        logits[0, class_idx].backward()                # backward → _hook_bwd fires

        # Global-average-pool gradients over spatial dims → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation channels
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)

        c = cam.squeeze().cpu().detach().float().numpy()
        c_min, c_max = float(c.min()), float(c.max())
        return (c - c_min) / (c_max - c_min + 1e-8)

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def prep_clf(arr: np.ndarray, device: torch.device,
             requires_grad: bool = False) -> torch.Tensor:
    """arr (H,W) float32 [0,1] → (1,3,224,224) ImageNet-normalised tensor."""
    pil = PILImage.fromarray((arr * 255).astype(np.uint8)).resize(
        (IMG_CLF, IMG_CLF), PILImage.BILINEAR)
    t = torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    t = (t - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)
    if requires_grad:
        t = t.requires_grad_(True)
    return t


@torch.no_grad()
def enhance(arr: np.ndarray, denoiser: nn.Module,
            device: torch.device) -> np.ndarray:
    """arr (H,W) → enhanced (H,W) at native resolution."""
    H, W = arr.shape
    resized = SyntheticDenoiseDataset._resize(arr)          # (256,256)
    inp = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device)
    out256 = denoiser(inp).squeeze().cpu().numpy()
    out_pil = PILImage.fromarray((out256 * 255).astype(np.uint8)).resize(
        (W, H), PILImage.BILINEAR)
    return np.array(out_pil, dtype=np.float32) / 255.0


# ── Visualisation helpers ─────────────────────────────────────────────────────

def cam_overlay(img: np.ndarray, cam: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
    """Overlay GradCAM heatmap (hot colormap) on a greyscale image → RGB uint8."""
    H, W = img.shape
    # Upsample cam to image size
    cam_pil = PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
        (W, H), PILImage.BILINEAR)
    cam_up = np.array(cam_pil, dtype=np.float32) / 255.0

    # Apply hot colormap → (H,W,3)
    heatmap = mcm.hot(cam_up)[..., :3].astype(np.float32)

    # Greyscale image as RGB
    img_rgb = np.stack([img, img, img], axis=-1)

    overlay = (1 - alpha) * img_rgb + alpha * heatmap
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def save_figure(
    orig: np.ndarray,
    enh_v1: np.ndarray,
    enh_v2: np.ndarray,
    cams: dict,    # {ver: {'noise': cam_arr, 'motion': cam_arr}}
    scores: dict,  # {ver: {'noise': float, 'motion': float}}
    title: str,
    out_path: Path,
) -> None:
    """3-row × 3-col figure: (orig/v1/v2) × (image/Noise CAM/Motion CAM)."""
    rows = [('Original',    'orig', orig),
            ('v1 enhanced', 'v1',   enh_v1),
            ('v2 enhanced', 'v2',   enh_v2)]
    col_heads = ['Image', 'GradCAM — Noise', 'GradCAM — Motion']
    cam_keys  = ['noise', 'motion']

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), dpi=110)
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.99)

    for r, (ver_label, vk, img) in enumerate(rows):
        sn = scores[vk]['noise']
        sm = scores[vk]['motion']

        # Col 0: raw image
        ax = axes[r, 0]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{ver_label}\nNoise={sn:.3f}  Motion={sm:.3f}', fontsize=8)
        ax.axis('off')

        # Cols 1-2: GradCAM overlays
        for c, ck in enumerate(cam_keys):
            ax = axes[r, c + 1]
            overlay = cam_overlay(img, cams[vk][ck])
            ax.imshow(overlay)

            # Arrow annotation: show score change vs original
            if r > 0:
                orig_s = scores['orig'][ck]
                delta  = scores[vk][ck] - orig_s
                arrow  = 'v' if delta < 0 else '^'
                color  = 'lime' if delta < 0 else 'tomato'
                ax.set_title(f'{col_heads[c+1]}\n{arrow} {delta:+.3f}',
                             fontsize=8, color=color)
            else:
                ax.set_title(col_heads[c + 1], fontsize=8)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(str(out_path), bbox_inches='tight')
    plt.close(fig)


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_clf(device: torch.device) -> nn.Module:
    from task_1a.model import ArtifactClassifier
    clf = ArtifactClassifier(backbone='densenet169', num_classes=7)
    ckpt = torch.load(TASK_1A_CHECKPOINT, map_location=device, weights_only=False)
    # Task 1A saves a raw state_dict (not wrapped in a dict)
    clf.load_state_dict(ckpt)
    clf.eval()
    # Keep parameters in the computation graph for GradCAM backward pass
    for p in clf.parameters():
        p.requires_grad_(True)
    return clf.to(device)


def load_denoiser(ckpt_name: str, device: torch.device) -> nn.Module:
    path = CHECKPOINTS_DIR / ckpt_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = build_model(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


# ── Per-partition analysis ────────────────────────────────────────────────────

def analyze_partition(
    label: str,
    csv_path: Path,
    clf: nn.Module,
    gradcam: GradCAM,
    den_v1: nn.Module,
    den_v2: nn.Module,
    device: torch.device,
    n_samples: int,
) -> list:
    print(f"\n── {label} ─────────────────────────────────────────")
    out_dir = GRADCAM_DIR / label
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)
    records = []

    for fn in df['filename']:
        if len(records) >= n_samples:
            break
        p = find_image_path(fn, TRAIN_DIR)
        if p is None:
            continue
        try:
            norm_slices, _, _ = load_normalized_slices(p)
        except Exception as e:
            print(f"  [SKIP] {fn}: {e}")
            continue
        if not norm_slices:
            continue

        mid  = len(norm_slices) // 2
        orig = norm_slices[mid]

        # Enhance with both models
        enh_v1 = enhance(orig, den_v1, device)
        enh_v2 = enhance(orig, den_v2, device)

        images = [('orig', orig), ('v1', enh_v1), ('v2', enh_v2)]

        # Get Task 1A scores (no-grad) and GradCAM maps (with-grad)
        scores = {}
        cams   = {}

        for vk, img in images:
            # Scores via single no-grad forward pass
            with torch.no_grad():
                x_score = prep_clf(img, device, requires_grad=False)
                probs = torch.sigmoid(clf(x_score))[0].cpu().numpy()
            scores[vk] = {
                'noise':  float(probs[NOISE_IDX]),
                'motion': float(probs[MOTION_IDX]),
            }

            # GradCAM for Noise class
            x_grad = prep_clf(img, device, requires_grad=True)
            cam_noise = gradcam(x_grad, NOISE_IDX)
            clf.zero_grad()

            # GradCAM for Motion class (needs fresh forward pass)
            x_grad2 = prep_clf(img, device, requires_grad=True)
            cam_motion = gradcam(x_grad2, MOTION_IDX)
            clf.zero_grad()

            cams[vk] = {'noise': cam_noise, 'motion': cam_motion}

        # Print score row
        case_id = Path(str(fn)).stem[:22]
        sn = {vk: scores[vk]['noise']  for vk in ('orig', 'v1', 'v2')}
        sm = {vk: scores[vk]['motion'] for vk in ('orig', 'v1', 'v2')}
        print(f"  [{len(records)+1:2d}] {case_id[:18]:<18} | "
              f"Noise  orig={sn['orig']:.3f} v1={sn['v1']:.3f} v2={sn['v2']:.3f} | "
              f"Motion orig={sm['orig']:.3f} v1={sm['v1']:.3f} v2={sm['v2']:.3f}")

        # Save figure
        case_tag = f"{Path(str(fn)).stem[:16]}_sl{mid}"
        out_path = out_dir / f"{case_tag}.png"
        save_figure(orig, enh_v1, enh_v2, cams, scores,
                    title=f"{label}  |  {case_tag}  (slice {mid})",
                    out_path=out_path)

        records.append({'case': str(fn), 'slice': mid, 'scores': scores})

    print(f"  {len(records)} figures -> {out_dir}")
    return records


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(aggregate: dict) -> None:
    print("\n" + "=" * 76)
    print(f"{'Partition':<26} {'Class':<8} {'Original':>10} {'v1':>10} {'v2':>10} "
          f"{'v1 delta':>10} {'v2 delta':>10}")
    print("=" * 76)
    for label, stats in aggregate.items():
        for cls_name, ck in [('Noise', 'noise_mean'), ('Motion', 'motion_mean')]:
            orig = stats['orig'][ck]
            v1   = stats['v1'][ck]
            v2   = stats['v2'][ck]
            d1   = v1 - orig
            d2   = v2 - orig
            tag1 = 'v' if d1 < 0 else '^'
            tag2 = 'v' if d2 < 0 else '^'
            print(f"{label:<26} {cls_name:<8} {orig:>10.3f} {v1:>10.3f} {v2:>10.3f} "
                  f"{tag1}{d1:>+9.3f} {tag2}{d2:>+9.3f}")
    print("=" * 76)
    print("v = score decreased (artifact reduced)   ^ = score increased (artifact worsened)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=5,
                        help='Images to sample per partition (default: 5)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Samples per partition: {args.samples}")

    # Task 1A classifier + GradCAM setup
    print("\nLoading Task 1A classifier (DenseNet169)...")
    clf = load_clf(device)
    # Target layer: norm5 = final BN after denseblock4, last spatial feature map
    target_layer = clf.backbone.features.norm5
    gradcam = GradCAM(clf, target_layer)
    print(f"  GradCAM target: clf.backbone.features.norm5")

    # Task 1B denoisers
    print("Loading Task 1B denoisers...")
    den_v1 = load_denoiser('best_1b.pth',    device)
    den_v2 = load_denoiser('best_1b_v2.pth', device)
    print("  v1 (best_1b.pth) and v2 (best_1b_v2.pth) loaded.")

    partitions = [
        ('withnoise_nomotion',   CSV_WITHNOISE_NOMOTION),
        ('nonoise_withmotion',   CSV_NONOISE_WITHMOTION),
        ('withnoise_withmotion', CSV_WITHNOISE_WITHMOTION),
    ]

    all_records = {}
    for label, csv_path in partitions:
        if not csv_path.exists():
            print(f"\n[SKIP] {csv_path} not found.")
            continue
        all_records[label] = analyze_partition(
            label, csv_path, clf, gradcam, den_v1, den_v2, device, args.samples)

    gradcam.remove()

    # Aggregate statistics
    aggregate = {}
    for label, records in all_records.items():
        if not records:
            continue
        agg = {}
        for vk in ('orig', 'v1', 'v2'):
            noises  = [r['scores'][vk]['noise']  for r in records]
            motions = [r['scores'][vk]['motion'] for r in records]
            agg[vk] = {
                'noise_mean':  float(np.mean(noises)),
                'noise_std':   float(np.std(noises)),
                'motion_mean': float(np.mean(motions)),
                'motion_std':  float(np.std(motions)),
            }
        aggregate[label] = agg

    print_summary(aggregate)

    out = GRADCAM_DIR / 'summary.json'
    out.write_text(json.dumps({'per_case': all_records, 'aggregate': aggregate}, indent=2))
    print(f"\nSummary -> {out}")
    print(f"Figures  -> {GRADCAM_DIR}")


if __name__ == '__main__':
    main()
