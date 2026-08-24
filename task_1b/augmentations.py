"""Physics-based MRI artifact generators for Task 1B v2.

Inspired by Ravi et al. (2024) "Learning to Enhance Ultra-low-field MRI
through Physics-informed Degradation..."  Medical Image Analysis.

All functions operate on 2-D float32 numpy arrays in [0, 1].
"""
import numpy as np
from PIL import Image as PILImage


# ── Artifact generators ───────────────────────────────────────────────────────

def add_kspace_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian noise in k-space — more physically accurate than image-domain Rician.

    MRI thermal noise arises in the receiver coil and is i.i.d. complex Gaussian
    in k-space.  sigma is relative to the mean k-space magnitude.
    """
    kspace = np.fft.fft2(img)
    scale = sigma * float(np.abs(kspace).mean())
    noise = np.random.normal(0, scale, kspace.shape) + \
            1j * np.random.normal(0, scale, kspace.shape)
    result = np.abs(np.fft.ifft2(kspace + noise)).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def add_gibbs_ringing(img: np.ndarray, pct_h: float, pct_v: float) -> np.ndarray:
    """Gibbs ringing via random k-space line undersampling.

    Zeros pct_h fraction of outer horizontal k-space rows and pct_v fraction
    of outer vertical columns (DC region is preserved).
    pct_h, pct_v: fraction of lines to zero out (range 0.01 – 0.25).
    """
    kspace = np.fft.fftshift(np.fft.fft2(img))
    H, W = img.shape

    if pct_h > 0:
        n = max(1, int(H * pct_h))
        outer = list(range(0, H // 4)) + list(range(3 * H // 4, H))
        rows = np.random.choice(outer, size=min(n, len(outer)), replace=False)
        kspace[rows, :] = 0

    if pct_v > 0:
        n = max(1, int(W * pct_v))
        outer = list(range(0, W // 4)) + list(range(3 * W // 4, W))
        cols = np.random.choice(outer, size=min(n, len(outer)), replace=False)
        kspace[:, cols] = 0

    result = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace))).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def add_banding(img: np.ndarray, amplitude: float, n_spikes: int,
                dist_frac: float) -> np.ndarray:
    """Banding artifacts via symmetric k-space spikes.

    Adds n_spikes pairs of high-amplitude conjugate-symmetric k-space points
    at distance dist_frac * min(H,W)/2 from DC.  Manifests as stripe patterns.
    amplitude: spike amplitude relative to DC (0.05 – 0.35).
    dist_frac: spike distance as fraction of half k-space size (0.10 – 0.45).
    """
    kspace = np.fft.fftshift(np.fft.fft2(img))
    H, W = img.shape
    cy, cx = H // 2, W // 2
    dc_amp = float(np.abs(kspace[cy, cx]))
    half = min(H, W) / 2.0

    for _ in range(n_spikes):
        angle = np.random.uniform(0, 2 * np.pi)
        dist_px = int(dist_frac * half)
        dy = int(dist_px * np.sin(angle))
        dx = int(dist_px * np.cos(angle))
        r1 = np.clip(cy + dy, 0, H - 1)
        c1 = np.clip(cx + dx, 0, W - 1)
        r2 = np.clip(cy - dy, 0, H - 1)
        c2 = np.clip(cx - dx, 0, W - 1)
        phase = np.random.uniform(0, 2 * np.pi)
        val = amplitude * dc_amp * np.exp(1j * phase)
        kspace[r1, c1] += val
        kspace[r2, c2] += np.conj(val)

    result = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace))).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def add_zipper(img: np.ndarray, n_lines: int, amplitude: float) -> np.ndarray:
    """Zipper artifacts via periodic k-space column corruption.

    Replaces k-space columns at regular spacing with random complex noise,
    producing vertical stripe patterns in image space.
    n_lines: number of corrupted columns (1 – 4).
    amplitude: noise amplitude relative to max k-space magnitude (0.05 – 0.25).
    """
    kspace = np.fft.fft2(img)
    H, W = kspace.shape
    spacing = W // (n_lines + 1)
    scale = amplitude * float(np.abs(kspace).max())
    for k in range(1, n_lines + 1):
        col = k * spacing
        if 0 <= col < W:
            kspace[:, col] = (np.random.normal(0, scale, H) +
                               1j * np.random.normal(0, scale, H))
    result = np.abs(np.fft.ifft2(kspace)).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def add_bias_field(img: np.ndarray, order: int = 2,
                   strength: float = 0.25) -> np.ndarray:
    """Smooth multiplicative bias field simulating RF/B1 inhomogeneity.

    Models the B1+ field as a low-order 2-D polynomial multiplied into the image.
    order: polynomial degree (1 = linear, 2 = quadratic, 3 = cubic).
    strength: maximum fractional intensity variation (0.10 – 0.35).
    """
    H, W = img.shape
    y = np.linspace(-1, 1, H, dtype=np.float32)
    x = np.linspace(-1, 1, W, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')

    field = np.ones((H, W), dtype=np.float32)
    cap = strength / max(order * 2, 1)
    for n in range(1, order + 1):
        for k in range(n + 1):
            coef = np.random.uniform(-cap, cap)
            field += coef * (yy ** (n - k)) * (xx ** k)

    field = np.clip(field, 1.0 - strength, 1.0 + strength)
    return np.clip((img * field).astype(np.float32), 0.0, 1.0)


def add_motion_ghosting_v2(img: np.ndarray, rotation_deg: float,
                           translation_px: int,
                           pct_lines: float) -> np.ndarray:
    """Physics-based motion ghosting — 3-parameter model (Ravi et al.).

    Swaps a random subset of k-space phase-encode lines with lines from a
    slightly rotated+translated copy of the image (simulating patient motion
    during the phase-encode acquisition window).
    rotation_deg: ghost rotation (0.5 – 12 deg).
    translation_px: ghost x-translation in pixels (1 – 10).
    pct_lines: fraction of k-space rows swapped (0.05 – 0.30).
    """
    H, W = img.shape
    pil = PILImage.fromarray((img * 255).astype(np.uint8))
    moved = pil.rotate(rotation_deg, translate=(translation_px, 0), fillcolor=0)
    moved_arr = np.array(moved, dtype=np.float32) / 255.0

    kspace_orig  = np.fft.fft2(img)
    kspace_moved = np.fft.fft2(moved_arr)
    n_swap = max(1, int(H * pct_lines))
    rows = np.random.choice(H, size=n_swap, replace=False)
    mixed = kspace_orig.copy()
    mixed[rows, :] = kspace_moved[rows, :]

    result = np.abs(np.fft.ifft2(mixed)).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


# ── Composite degradation pipeline ───────────────────────────────────────────

def composite_degrade(img: np.ndarray) -> np.ndarray:
    """Apply a random subset of physics-based artifacts.

    Each artifact is independently sampled; probabilities reflect the
    artifact distribution observed in Hyperfine SWOOP 0.064T images.
    """
    deg = img.copy()

    # 1. k-space noise (always applied — thermal noise is always present)
    sigma = np.random.uniform(0.05, 0.20)
    deg = add_kspace_noise(deg, sigma)

    # 2. Gibbs ringing (40 % of samples)
    if np.random.random() < 0.40:
        pct_h = np.random.uniform(0.01, 0.20)
        pct_v = np.random.uniform(0.01, 0.20)
        deg = add_gibbs_ringing(deg, pct_h, pct_v)

    # 3. Banding (30 % of samples)
    if np.random.random() < 0.30:
        amp      = np.random.uniform(0.05, 0.30)
        n_spikes = np.random.randint(1, 4)
        dist     = np.random.uniform(0.10, 0.40)
        deg = add_banding(deg, amp, n_spikes, dist)

    # 4. Zipper (20 % of samples)
    if np.random.random() < 0.20:
        n_lines = np.random.randint(1, 4)
        amp     = np.random.uniform(0.05, 0.25)
        deg = add_zipper(deg, n_lines, amp)

    # 5. Bias field (50 % of samples)
    if np.random.random() < 0.50:
        order    = np.random.randint(1, 4)
        strength = np.random.uniform(0.10, 0.35)
        deg = add_bias_field(deg, order, strength)

    # 6. Motion ghosting v2 (50 % of samples)
    if np.random.random() < 0.50:
        rot    = np.random.uniform(0.5, 10.0)
        transl = np.random.randint(1, 8)
        pct    = np.random.uniform(0.05, 0.25)
        deg = add_motion_ghosting_v2(deg, rot, transl, pct)

    return deg
