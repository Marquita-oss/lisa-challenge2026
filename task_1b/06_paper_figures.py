"""Task 1B — Paper-quality figure generation for MICCAI / Springer LNCS.

Generates all figures needed for the paper using:
  - Okabe-Ito colorblind-safe palette for categorical comparisons
  - Viridis for GradCAM heatmaps (perceptually uniform, colorblind-safe)
  - 300 DPI, PDF output (vector, LNCS-compatible)
  - Single-column LNCS text width: 12.2 cm

Figures produced:
  fig1_training_curves.pdf      Loss decomposition during v2 training
  fig2_scores_comparison.pdf    Noise/Motion score reduction: orig vs v1 vs v2
  fig3_gradcam_panel.pdf        Best-case GradCAM qualitative comparison
  fig4_psnr_ssim.pdf            PSNR / SSIM improvement bar chart

Usage:
  python task_1b/06_paper_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task_1b.config import (
    RESULTS_DIR, CHECKPOINTS_DIR, TASK_1A_CHECKPOINT, TRAIN_DIR,
    CSV_WITHNOISE_NOMOTION, CSV_NONOISE_WITHMOTION, CSV_WITHNOISE_WITHMOTION,
)

FIGURES_DIR = RESULTS_DIR / 'paper_figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Okabe-Ito colorblind-safe palette ─────────────────────────────────────────
# Source: Okabe & Ito (2008), colorblind-friendly, distinguishable in greyscale.
C_ORIG = '#888888'    # grey    — original / input
C_V1   = '#E69F00'    # amber   — v1 (Rician + simple ghosting)
C_V2   = '#0072B2'    # blue    — v2 (physics-based + adversarial)
C_GOOD = '#009E73'    # green   — score decreases (good)
C_BAD  = '#D55E00'    # vermillion — score increases (bad)
C_NEUTRAL = '#56B4E9' # sky-blue — neutral accent

# LNCS single-column text width in inches
LNCS_W = 4.80   # 12.2 cm ≈ 4.80 in
LNCS_H = 3.60   # standard height

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':        8,
    'axes.titlesize':   8,
    'axes.labelsize':   8,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'legend.fontsize':  7,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'pdf.fonttype':     42,   # embeds fonts (required by Springer)
    'ps.fonttype':      42,
})


# ── Helper: save figure ────────────────────────────────────────────────────────

def save(fig, name: str) -> None:
    stem = name.replace('.pdf', '')
    for ext in ['pdf', 'png']:
        path = FIGURES_DIR / f'{stem}.{ext}'
        fig.savefig(str(path), bbox_inches='tight', dpi=300)
        print(f"  Saved: {path.name}")
    plt.close(fig)


# ── Fig 1: Training curves ────────────────────────────────────────────────────

def fig_training_curves() -> None:
    """Val PSNR + loss decomposition (L1, SSIM, Adversarial) over 100 epochs."""
    history_path = RESULTS_DIR / '02_training_v2.json'
    if not history_path.exists():
        print("  [SKIP] 02_training_v2.json not found")
        return

    data = json.loads(history_path.read_text())
    hist = data['history']
    epochs  = [h['epoch']      for h in hist]
    psnr    = [h['val_psnr']   for h in hist]
    l1      = [h['train_l1']   for h in hist]
    ssim    = [h['train_ssim'] for h in hist]
    adv     = [h['train_adv']  for h in hist]
    adv_on  = data['adv_warmup_epochs'] + 1  # epoch when adv activates

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(LNCS_W, 2.8))

    # Left: Val PSNR
    ax1.plot(epochs, psnr, color=C_V2, linewidth=1.2, label='Val PSNR')
    ax1.axvline(adv_on, color=C_BAD, linewidth=0.8, linestyle='--', alpha=0.7)
    ax1.axhline(28.0, color='k', linewidth=0.6, linestyle=':', alpha=0.5)
    ax1.text(adv_on + 1, min(psnr) + 0.3, 'Adv. ON', fontsize=6, color=C_BAD)
    ax1.text(2, 28.15, '28 dB criterion', fontsize=5.5, color='k', alpha=0.6)
    best_ep = epochs[np.argmax(psnr)]
    best_psnr = max(psnr)
    ax1.scatter([best_ep], [best_psnr], color=C_GOOD, s=20, zorder=5)
    ax1.annotate(f'{best_psnr:.2f} dB', xy=(best_ep, best_psnr),
                 xytext=(best_ep - 18, best_psnr - 0.5),
                 fontsize=6, color=C_GOOD,
                 arrowprops=dict(arrowstyle='->', color=C_GOOD, lw=0.7))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Validation PSNR')
    ax1.set_xlim(1, 100)
    ax1.grid(True, alpha=0.25, linewidth=0.5)

    # Right: Loss decomposition (smoothed)
    def smooth(x, w=5):
        return np.convolve(x, np.ones(w) / w, mode='same')

    ax2.plot(epochs, smooth(l1),   color=C_V1,      linewidth=1.2, label='L1')
    ax2.plot(epochs, smooth(ssim), color=C_NEUTRAL,  linewidth=1.2, label='1−SSIM')
    ax2.plot(epochs, smooth(adv),  color=C_BAD,      linewidth=1.2,
             linestyle='--', label='Adversarial')
    ax2.axvline(adv_on, color=C_BAD, linewidth=0.8, linestyle=':', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss component')
    ax2.set_title('Training loss decomposition')
    ax2.set_xlim(1, 100)
    ax2.legend(loc='upper right', framealpha=0.85)
    ax2.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout(pad=0.8)
    save(fig, 'fig1_training_curves.pdf')


# ── Fig 2: Score comparison bar chart ────────────────────────────────────────

def fig_scores_comparison() -> None:
    """Grouped bars: Noise and Motion scores for orig / v1 / v2, per partition."""
    eval_v2 = json.loads((RESULTS_DIR / '03_evaluation_v2.json').read_text())
    fb = eval_v2['task1a_quality_feedback']

    partitions_short = {
        'withnoise_nomotion':   'Noise\nonly',
        'nonoise_withmotion':   'Motion\nonly',
        'withnoise_withmotion': 'Noise +\nMotion',
    }

    n_parts = 3
    x = np.arange(n_parts)
    w = 0.22   # bar width

    # Build arrays — use v2 before scores as "original", compare v1 and v2
    # Note: v1 03_evaluation.json has stale normalization; use GradCAM aggregate for v1
    gradcam = json.loads((RESULTS_DIR / 'gradcam' / 'summary.json').read_text())['aggregate']

    labels = list(partitions_short.keys())
    short  = list(partitions_short.values())

    def get_arr(vk, cls):
        return [gradcam[p][vk][f'{cls}_mean'] for p in labels]

    orig_n  = get_arr('orig', 'noise')
    v1_n    = get_arr('v1',   'noise')
    v2_n    = get_arr('v2',   'noise')
    orig_m  = get_arr('orig', 'motion')
    v1_m    = get_arr('v1',   'motion')
    v2_m    = get_arr('v2',   'motion')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(LNCS_W, 2.8), sharey=False)

    for ax, orig, v1, v2, title in [
        (ax1, orig_n, v1_n, v2_n, 'Noise score (↓ better)'),
        (ax2, orig_m, v1_m, v2_m, 'Motion score (↓ better)'),
    ]:
        ax.bar(x - w, orig, w, color=C_ORIG,    label='Input',    zorder=3)
        ax.bar(x,     v1,   w, color=C_V1,     label='Baseline', zorder=3)
        ax.bar(x + w, v2,   w, color=C_V2,     label='Proposed', zorder=3)

        # Annotate v1 delta
        for i, (o, v) in enumerate(zip(orig, v1)):
            d = v - o
            col = C_GOOD if d < 0 else C_BAD
            ax.text(i,       v + 0.008, f'{d:+.2f}', ha='center',
                    fontsize=5.5, color=col, fontweight='bold')
        # Annotate v2 delta
        for i, (o, v) in enumerate(zip(orig, v2)):
            d = v - o
            col = C_GOOD if d < 0 else C_BAD
            ax.text(i + w,   v + 0.008, f'{d:+.2f}', ha='center',
                    fontsize=5.5, color=col, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=7)
        ax.set_ylabel('Score (sigmoid output)')
        ax.set_title(title)
        ax.set_ylim(0, max(max(orig), 0.55) + 0.12)
        ax.legend(loc='upper right', framealpha=0.85)
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    save(fig, 'fig2_scores_comparison.pdf')


# ── Fig 3: GradCAM panel ──────────────────────────────────────────────────────

def cam_overlay_viridis(img: np.ndarray, cam: np.ndarray,
                         alpha: float = 0.50) -> np.ndarray:
    """Overlay GradCAM with viridis colormap on greyscale image → RGB uint8."""
    H, W = img.shape
    cam_up = np.array(
        PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
            (W, H), PILImage.BILINEAR), dtype=np.float32) / 255.0
    heatmap = mcm.viridis(cam_up)[..., :3].astype(np.float32)
    img_rgb  = np.stack([img, img, img], axis=-1)
    overlay  = (1 - alpha) * img_rgb + alpha * heatmap
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def fig_gradcam_panel() -> None:
    """Best-case GradCAM panel for the paper.

    Uses LISA_0013_LF_sag (withnoise_nomotion): the case with the largest
    Noise reduction (0.700 → v1: 0.398 → v2: 0.295).
    Layout: 2 rows × 4 cols
      Row 0: Image orig | GradCAM Noise orig | GradCAM Noise v1 | GradCAM Noise v2
      Row 1: (empty)   | GradCAM Motion orig | GradCAM Motion v1 | GradCAM Motion v2
    With score annotations and delta arrows.
    """
    import torch
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Load models --
    from task_1b.model import build_model
    from task_1b.dataset import SyntheticDenoiseDataset
    from task_1b.utils.io import load_normalized_slices, find_image_path

    def load_den(name):
        m = build_model(device)
        ck = torch.load(CHECKPOINTS_DIR / name, map_location=device, weights_only=False)
        m.load_state_dict(ck['model_state'])
        m.eval()
        return m

    den_v1 = load_den('best_1b.pth')
    den_v2 = load_den('best_1b_v2.pth')

    from task_1a.model import ArtifactClassifier
    clf = ArtifactClassifier(backbone='densenet169', num_classes=7)
    ck = torch.load(TASK_1A_CHECKPOINT, map_location=device, weights_only=False)
    clf.load_state_dict(ck)
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(True)
    clf = clf.to(device)

    # GradCAM hooks
    acts, grads = [None], [None]
    def fwd_hook(_m, _i, out): acts[0] = out
    def bwd_hook(_m, _gi, go): grads[0] = go[0]
    h1 = clf.backbone.features.norm5.register_forward_hook(fwd_hook)
    h2 = clf.backbone.features.norm5.register_full_backward_hook(bwd_hook)

    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    _STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def prep(arr):
        pil = PILImage.fromarray((arr*255).astype(np.uint8)).resize((224,224), PILImage.BILINEAR)
        t = torch.from_numpy(np.array(pil,dtype=np.float32)/255.0)
        t = t.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)
        return ((t - _MEAN) / _STD).requires_grad_(True)

    def gradcam(arr, cls_idx):
        x = prep(arr)
        clf.zero_grad()
        logits = clf(x)
        logits[0, cls_idx].backward()
        w = grads[0].mean(dim=(2,3), keepdim=True)
        cam = F.relu((w * acts[0]).sum(dim=1, keepdim=True))
        c = cam.squeeze().cpu().detach().float().numpy()
        mn, mx = c.min(), c.max()
        return (c - mn) / (mx - mn + 1e-8)

    @torch.no_grad()
    def score(arr, cls_idx):
        x = prep(arr).detach().requires_grad_(False)
        return float(torch.sigmoid(clf(x))[0, cls_idx].item())

    @torch.no_grad()
    def enhance(arr):
        H, W = arr.shape
        r = SyntheticDenoiseDataset._resize(arr)
        inp = torch.from_numpy(r).unsqueeze(0).unsqueeze(0).to(device)
        return den_v1(inp).squeeze().cpu().numpy(), \
               den_v2(inp).squeeze().cpu().numpy()

    # -- Load best case: LISA_0013 withnoise_nomotion --
    import pandas as pd
    df = pd.read_csv(CSV_WITHNOISE_NOMOTION)
    # Find LISA_0013
    target = [fn for fn in df['filename'] if '0013' in fn]
    if not target:
        target = [df['filename'].iloc[0]]
    fn = target[0]
    p = find_image_path(fn, TRAIN_DIR)
    if p is None:
        print("  [SKIP] LISA_0013 not found for GradCAM panel")
        h1.remove(); h2.remove()
        return

    slices, _, _ = load_normalized_slices(p)
    sl = slices[len(slices) // 2]
    enh1, enh2 = enhance(sl)

    # Compute cams + scores
    NOISE, MOTION = 0, 4
    data = {}
    for vk, arr in [('orig', sl), ('v1', enh1), ('v2', enh2)]:
        data[vk] = {
            'img':    arr,
            'cam_n':  gradcam(arr, NOISE),
            'cam_m':  gradcam(arr, MOTION),
            'score_n': score(arr, NOISE),
            'score_m': score(arr, MOTION),
        }

    h1.remove(); h2.remove()

    # -- Build figure: 2 rows × 4 cols --
    # Row 0: Original image | Noise CAMs (orig, v1, v2)
    # Row 1: blank          | Motion CAMs (orig, v1, v2)
    ver_labels = ['Input', 'Baseline', 'Proposed']
    ver_keys   = ['orig', 'v1', 'v2']
    ver_colors = [C_ORIG, C_V1, C_V2]

    fig = plt.figure(figsize=(LNCS_W, 3.4))
    gs  = gridspec.GridSpec(2, 4, figure=fig, wspace=0.04, hspace=0.35)

    # Col 0, row 0: original image (spans 2 rows)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(sl, cmap='gray', vmin=0, vmax=1)
    ax_img.set_title('Input\nimage', fontsize=7, fontweight='bold')
    ax_img.axis('off')

    for row, (cls_name, ck_key, cls_idx) in enumerate([
        ('Noise', 'cam_n', NOISE),
        ('Motion', 'cam_m', MOTION),
    ]):
        for col, vk in enumerate(ver_keys):
            ax = fig.add_subplot(gs[row, col + 1])
            overlay = cam_overlay_viridis(data[vk]['img'], data[vk][ck_key])
            ax.imshow(overlay)
            ax.axis('off')

            s = data[vk][f'score_{ck_key[-1]}']
            d = s - data['orig'][f'score_{ck_key[-1]}']

            if vk == 'orig':
                ax.set_title(f'{cls_name}\n{s:.3f}', fontsize=6.5,
                             fontweight='bold', color=ver_colors[col])
            else:
                arrow = '▼' if d < 0 else '▲'
                ax.set_title(f'{ver_labels[col]}\n{s:.3f}  {arrow}{d:+.3f}',
                             fontsize=6.5, fontweight='bold',
                             color=ver_colors[col])

    # Colorbar (viridis)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label('Activation\nintensity', fontsize=6, labelpad=4)
    cb.ax.tick_params(labelsize=5.5)

    # Compact legend note — suptitle goes in LaTeX \caption{}
    #fig.text(0.50, 0.01, '▼ score reduced  |  ▲ score worsened',
    #         ha='center', fontsize=5.5, color='#555555')

    save(fig, 'fig3_gradcam_panel.pdf')


# ── Fig 4: PSNR / SSIM bar chart ─────────────────────────────────────────────

def fig_psnr_ssim() -> None:
    """PSNR and SSIM improvement: input vs v1-output vs v2-output."""
    e1 = json.loads((RESULTS_DIR / '03_evaluation.json').read_text())['synthetic_eval']
    e2 = json.loads((RESULTS_DIR / '03_evaluation_v2.json').read_text())['synthetic_eval']

    metrics = ['PSNR (dB)', 'SSIM']
    input_vals = [e1['psnr_input'],  e1['ssim_input']]
    v1_vals    = [e1['psnr_output'], e1['ssim_output']]
    v2_vals    = [e2['psnr_output'], e2['ssim_output']]

    fig, axes = plt.subplots(1, 2, figsize=(LNCS_W, 2.6))

    for ax, metric, inp, v1, v2 in zip(axes, metrics, input_vals, v1_vals, v2_vals):
        bars = ax.bar([0, 1, 2], [inp, v1, v2],
                      color=[C_ORIG, C_V1, C_V2],
                      width=0.55, zorder=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Input', 'Baseline', 'Proposed'], fontsize=7)
        ax.set_title(metric)
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        # Value labels on bars
        for bar, val in zip(bars, [inp, v1, v2]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if metric == 'PSNR (dB)' else 0.005),
                    f'{val:.2f}' if metric == 'PSNR (dB)' else f'{val:.3f}',
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')

        # Criterion line
        if metric == 'PSNR (dB)':
            ax.axhline(28.0, color='k', linewidth=0.7, linestyle=':', alpha=0.6)
            ax.text(2.35, 28.15, '≥28 dB', fontsize=5.5, color='k', alpha=0.7)
        elif metric == 'SSIM':
            ax.axhline(0.80, color='k', linewidth=0.7, linestyle=':', alpha=0.6)
            ax.text(2.35, 0.806, '≥0.80', fontsize=5.5, color='k', alpha=0.7)

        ymin = min(inp, v1, v2) * 0.92
        ax.set_ylim(ymin, max(inp, v1, v2) * 1.10)

    legend_patches = [
        mpatches.Patch(color=C_ORIG, label='Input (degraded)'),
        mpatches.Patch(color=C_V1,   label='Baseline (Rician+Ghosting)'),
        mpatches.Patch(color=C_V2,   label='Proposed (Physics+Adv.)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=6.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout(pad=0.8, rect=[0, 0.06, 1, 1])
    save(fig, 'fig4_psnr_ssim.pdf')


# ── Fig 5: Adversarial loss effect summary ───────────────────────────────────

def fig_adv_effect_summary() -> None:
    """Score reduction heatmap: 3 partitions × 2 classes × 3 versions.

    Uses a diverging colormap (RdYlGn) centred at the original score,
    showing how v1 and v2 push scores up or down.
    """
    gradcam = json.loads((RESULTS_DIR / 'gradcam' / 'summary.json').read_text())['aggregate']

    part_labels = ['Noise\nonly', 'Motion\nonly', 'Noise+\nMotion']
    part_keys   = ['withnoise_nomotion', 'nonoise_withmotion', 'withnoise_withmotion']
    cls_labels  = ['Noise class', 'Motion class']
    cls_keys    = ['noise_mean', 'motion_mean']
    ver_labels  = ['Input', 'Baseline', 'Proposed']
    ver_keys    = ['orig', 'v1', 'v2']

    # Build 3-panel (one per version) or a grouped bar
    # Use a compact grouped horizontal bar chart
    fig, axes = plt.subplots(1, 2, figsize=(LNCS_W, 2.8))

    for ax, ck, cl in zip(axes, cls_keys, cls_labels):
        data = {vk: [gradcam[pk][vk][ck] for pk in part_keys] for vk in ver_keys}
        x = np.arange(len(part_keys))
        w = 0.22
        ax.barh(x - w, data['orig'], w, color=C_ORIG, label='Input',    zorder=3)
        ax.barh(x,     data['v1'],   w, color=C_V1,   label='Baseline', zorder=3)
        ax.barh(x + w, data['v2'],   w, color=C_V2,   label='Proposed', zorder=3)

        # Delta annotations for v2 only (most important)
        for i, (o, v) in enumerate(zip(data['orig'], data['v2'])):
            d = v - o
            col = C_GOOD if d < 0 else C_BAD
            ax.text(v + 0.005, i + w, f'{d:+.2f}', va='center',
                    fontsize=5.5, color=col, fontweight='bold')

        ax.set_yticks(x)
        ax.set_yticklabels(part_labels, fontsize=7)
        ax.set_xlabel('Score')
        ax.set_title(cl)
        ax.set_xlim(0, max(max(data['orig']), 0.55) + 0.15)
        ax.legend(loc='lower right', fontsize=6, framealpha=0.85)
        ax.grid(axis='x', alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    for ax in axes:
        ax.title.set_gid(10)
    fig.tight_layout(pad=0.8)
    save(fig, 'fig5_adv_effect_summary.pdf')


# ── Fig 6: Qualitative panel ──────────────────────────────────────────────────

def fig_qualitative_panel() -> None:
    """3 × 3 image panel: Input | Baseline | Proposed for one case per partition.

    Cases chosen by highest original artifact score per partition (from GradCAM).
    Score annotations (N:noise  M:motion) overlaid bottom-left of each cell.
    """
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from task_1b.model import build_model
    from task_1b.dataset import SyntheticDenoiseDataset
    from task_1b.utils.io import load_normalized_slices, find_image_path

    def load_den(name):
        m = build_model(device)
        ck = torch.load(CHECKPOINTS_DIR / name, map_location=device, weights_only=False)
        m.load_state_dict(ck['model_state'])
        m.eval()
        return m

    den_v1 = load_den('best_1b.pth')
    den_v2 = load_den('best_1b_v2.pth')

    @torch.no_grad()
    def enhance(arr):
        r = SyntheticDenoiseDataset._resize(arr)
        inp = torch.from_numpy(r).unsqueeze(0).unsqueeze(0).to(device)
        out1 = den_v1(inp).squeeze().cpu().numpy()
        out2 = den_v2(inp).squeeze().cpu().numpy()
        return r, out1, out2

    # Best case per partition — highest original artifact score (from GradCAM summary)
    cases = [
        ('withnoise_nomotion',   'LISA_0013_LF_sag.nii.gz', 'Noise\nonly'),
        ('nonoise_withmotion',   'LISA_0004_LF_cor.nii.gz', 'Motion\nonly'),
        ('withnoise_withmotion', 'LISA_0008_LF_sag.nii.gz', 'Noise +\nMotion'),
    ]
    gcam = json.loads((RESULTS_DIR / 'gradcam' / 'summary.json').read_text())

    fig, axes = plt.subplots(3, 3, figsize=(LNCS_W, LNCS_W),
                              gridspec_kw=dict(wspace=0.03, hspace=0.14))
    col_titles = ['Input', 'Baseline', 'Proposed']
    col_colors = [C_ORIG, C_V1, C_V2]
    ver_map    = ['orig', 'v1', 'v2']

    for r, (partition, fn, row_label) in enumerate(cases):
        p = find_image_path(fn, TRAIN_DIR)
        if p is None:
            print(f"  [SKIP] {fn} not found")
            for ax in axes[r]:
                ax.axis('off')
            continue

        slices, _, _ = load_normalized_slices(p)
        sl = slices[len(slices) // 2]
        inp_r, out1, out2 = enhance(sl)

        case_data = next(
            (c for c in gcam['per_case'][partition] if fn in c['case']), None)

        for c, (arr, vk) in enumerate(zip([inp_r, out1, out2], ver_map)):
            ax = axes[r, c]
            ax.imshow(arr, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
            ax.axis('off')
            if case_data:
                s = case_data['scores'][vk]
                ax.text(0.03, 0.04, f"N:{s['noise']:.2f}  M:{s['motion']:.2f}",
                        transform=ax.transAxes, fontsize=4.8, color='white',
                        bbox=dict(facecolor='black', alpha=0.55, pad=1.5,
                                  boxstyle='round,pad=0.2'))

        axes[r, 0].text(-0.05, 0.5, row_label,
                         transform=axes[r, 0].transAxes,
                         rotation=90, va='center', ha='right',
                         fontsize=6.5, fontweight='bold')

    for c, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, c].set_title(title, fontsize=7, fontweight='bold',
                              color=color, pad=4)

    fig.tight_layout(pad=0.3)
    save(fig, 'fig6_qualitative_panel.pdf')


# ── Fig 7: Domain gap slope chart ─────────────────────────────────────────────

def fig_domain_gap() -> None:
    """Slope chart exposing Baseline domain gap on motion-only images.

    Rician-trained Baseline (v1) fails to generalise to motion artefacts,
    increasing the Task-1A Noise score in 4/5 motion-only cases.
    The Proposed method reduces both scores across all cases.
    """
    from matplotlib.lines import Line2D

    gcam   = json.loads((RESULTS_DIR / 'gradcam' / 'summary.json').read_text())
    cases  = gcam['per_case']['nonoise_withmotion']
    x      = [0, 1, 2]
    x_lbl  = ['Input', 'Baseline', 'Proposed']

    fig, axes = plt.subplots(1, 2, figsize=(LNCS_W, 2.6))

    for ax, cls_label, skey in [
        (axes[0], 'Noise class',  'noise'),
        (axes[1], 'Motion class', 'motion'),
    ]:
        all_ys, n_bad = [], 0

        for case in cases:
            s0 = case['scores']['orig'][skey]
            s1 = case['scores']['v1'][skey]
            s2 = case['scores']['v2'][skey]
            ys = [s0, s1, s2]
            all_ys.append(ys)
            col = C_BAD if s1 > s0 else C_GOOD
            if s1 > s0:
                n_bad += 1
            ax.plot(x, ys, color=col, linewidth=1.1, alpha=0.72,
                    marker='o', markersize=3.5)

        means = np.mean(all_ys, axis=0)
        ax.plot(x, means, color='black', linewidth=2.0, linestyle='--',
                marker='D', markersize=5, zorder=5)

        span = float(np.max(all_ys) - np.min(all_ys))
        ax.annotate(
            f'{n_bad}/{len(cases)} cases\nBaseline ↑',
            xy=(1, means[1]),
            xytext=(1.22, means[1] + span * 0.28),
            fontsize=5.5, color=C_BAD, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_BAD, lw=0.7),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(x_lbl, fontsize=7)
        ax.set_ylabel('Score (sigmoid output)', fontsize=7)
        ax.set_title(f'{cls_label}\n(motion-only partition)', fontsize=7)
        ax.set_xlim(-0.35, 2.55)
        pad = span * 0.15
        ax.set_ylim(float(np.min(all_ys)) - pad,
                    float(np.max(all_ys)) + span * 0.55)
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        legend_els = [
            Line2D([0], [0], color=C_BAD,   lw=1.2, marker='o', ms=3.5,
                   label='Baseline worsens ▲'),
            Line2D([0], [0], color=C_GOOD,  lw=1.2, marker='o', ms=3.5,
                   label='Baseline improves ▼'),
            Line2D([0], [0], color='black', lw=2.0, ls='--', marker='D', ms=4,
                   label='Mean'),
        ]
        ax.legend(handles=legend_els, fontsize=5.5, loc='upper left',
                  framealpha=0.85)

    fig.tight_layout(pad=0.8)
    save(fig, 'fig7_domain_gap.pdf')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Output directory: {FIGURES_DIR}\n")
    print("Generating figures...")

    print("\n[1/7] Training curves (v2)...")
    fig_training_curves()

    print("\n[2/7] Score comparison (grouped bars)...")
    fig_scores_comparison()

    print("\n[3/7] GradCAM panel (best case)...")
    fig_gradcam_panel()

    print("\n[4/7] PSNR / SSIM bar chart...")
    fig_psnr_ssim()

    print("\n[5/7] Adversarial effect summary (horizontal bars)...")
    fig_adv_effect_summary()

    print("\n[6/7] Qualitative panel (Input | Baseline | Proposed)...")
    fig_qualitative_panel()

    print("\n[7/7] Domain gap slope chart...")
    fig_domain_gap()

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("\nFigure index for paper:")
    print("  fig1_training_curves.pdf    → Sec 3  Method / Training")
    print("  fig2_scores_comparison.pdf  → Sec 4  Results (Task-1A feedback)")
    print("  fig3_gradcam_panel.pdf      → Sec 4  Qualitative / Explainability")
    print("  fig4_psnr_ssim.pdf          → Sec 4  Quantitative metrics")
    print("  fig5_adv_effect_summary.pdf → Sec 4  Adversarial effect")
    print("  fig6_qualitative_panel.pdf  → Sec 4  Visual comparison")
    print("  fig7_domain_gap.pdf         → Sec 4  Domain gap / motivation")


if __name__ == '__main__':
    main()
