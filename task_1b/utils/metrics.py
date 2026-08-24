"""PSNR and SSIM metrics for image quality evaluation."""
import numpy as np
import torch
import torch.nn.functional as F


# ── Torch (batch) versions — used during training ─────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


def ssim_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """SSIM between two (B, 1, H, W) batches. Returns scalar tensor."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel(window_size, sigma).to(pred.device)
    kernel = kernel.view(1, 1, window_size, window_size)

    pad = window_size // 2
    mu1 = F.conv2d(pred,   kernel, padding=pad)
    mu2 = F.conv2d(target, kernel, padding=pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sig1_sq = F.conv2d(pred ** 2,         kernel, padding=pad) - mu1_sq
    sig2_sq = F.conv2d(target ** 2,       kernel, padding=pad) - mu2_sq
    sig12   = F.conv2d(pred * target,     kernel, padding=pad) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))
    return ssim_map.mean()


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(data_range ** 2 / mse)


# ── NumPy versions — used for per-image reporting ─────────────────────────────

def psnr_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    data_range = float(target.max() - target.min())
    if data_range < 1e-8:
        return 100.0
    return 10.0 * np.log10(data_range ** 2 / mse)


def ssim_numpy(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Simple SSIM for 2D numpy arrays."""
    from scipy.ndimage import uniform_filter
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    win = 11

    pred   = pred.astype(np.float64)
    target = target.astype(np.float64)

    mu1 = uniform_filter(pred,   size=win)
    mu2 = uniform_filter(target, size=win)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sig1_sq = uniform_filter(pred ** 2,       size=win) - mu1_sq
    sig2_sq = uniform_filter(target ** 2,     size=win) - mu2_sq
    sig12   = uniform_filter(pred * target,   size=win) - mu12

    num = (2 * mu12 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    return float(np.mean(num / (den + 1e-10)))
