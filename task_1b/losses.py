"""Loss functions for Task 1B.

v1: CombinedLoss — λ_L1 · L1 + λ_SSIM · (1 − SSIM)
v2: CombinedLossWithAdversarial — same reconstruction loss + adversarial BCE
    using a frozen Task 1A classifier as discriminator.
"""
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from task_1b.utils.metrics import ssim_torch
from task_1b.config import LAMBDA_L1, LAMBDA_SSIM, LAMBDA_ADV


# ── v1 ────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """λ_L1 · L1 + λ_SSIM · (1 − SSIM)."""

    def __init__(self, lambda_l1: float = LAMBDA_L1, lambda_ssim: float = LAMBDA_SSIM):
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss   = self.l1(pred, target)
        ssim_val  = ssim_torch(pred, target)
        return self.lambda_l1 * l1_loss + self.lambda_ssim * (1.0 - ssim_val)


# ── v2: frozen Task 1A discriminator ─────────────────────────────────────────

# ImageNet stats used by Task 1A's DenseNet169
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Class indices in the Task 1A output vector
_NOISE_IDX  = 0   # 'Noise'
_MOTION_IDX = 4   # 'Motion'


class Task1aDiscriminator:
    """Frozen Task 1A ArtifactClassifier used to guide Task 1B denoising.

    Converts single-channel enhanced slices → 3-channel ImageNet-normalised
    224×224 tensors, runs them through the frozen classifier, and returns the
    Noise and Motion logits so the adversarial loss can push them toward 0.

    This class is NOT an nn.Module so it is never accidentally trained.
    """

    def __init__(self, ckpt_path: Path, device: torch.device):
        import sys
        from pathlib import Path as _Path
        root = _Path(ckpt_path).resolve().parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from task_1a.model import ArtifactClassifier
        self.device = device

        model = ArtifactClassifier(backbone='densenet169', num_classes=7,
                                   dropout1=0.0, dropout2=0.0)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Task 1A saves a raw state_dict (no wrapper dict)
        state = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.to(device)

        self._mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std  = torch.tensor(_IMAGENET_STD,  device=device).view(1, 3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) float32 in [0,1].  Returns (B, 2) logits [Noise, Motion].

        No @torch.no_grad() here: discriminator params are frozen (requires_grad=False)
        so they are not updated, but gradients flow back to x (= the enhanced image).
        """
        x3 = x.repeat(1, 3, 1, 1)                                     # (B,3,H,W)
        x3 = F.interpolate(x3, size=(224, 224), mode='bilinear',
                            align_corners=False)
        x3 = (x3 - self._mean) / self._std
        logits = self.model(x3)                                         # (B,7)
        return logits[:, [_NOISE_IDX, _MOTION_IDX]]                    # (B,2)


class CombinedLossWithAdversarial(nn.Module):
    """Reconstruction loss + adversarial BCE via frozen Task 1A classifier.

    L_total = λ_L1·L1 + λ_SSIM·(1−SSIM) + λ_adv·BCE(task1a(pred), 0)

    The adversarial term is gated by `adv_active`: set it to False during the
    first ADV_WARMUP_EPOCHS epochs so the model first learns basic denoising.
    """

    def __init__(
        self,
        discriminator: Task1aDiscriminator,
        lambda_l1:   float = LAMBDA_L1,
        lambda_ssim: float = LAMBDA_SSIM,
        lambda_adv:  float = LAMBDA_ADV,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_l1   = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_adv  = lambda_adv
        self.l1  = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.adv_active = False

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        l1_loss   = self.l1(pred, target)
        ssim_val  = ssim_torch(pred, target)
        ssim_loss = 1.0 - ssim_val

        adv_loss = torch.tensor(0.0, device=pred.device)
        if self.adv_active and self.lambda_adv > 0:
            # Discriminator params are frozen; gradients flow back to pred only.
            logits = self.discriminator(pred)
            targets_zero = torch.zeros_like(logits)
            adv_loss = self.bce(logits, targets_zero)

        total = (self.lambda_l1 * l1_loss
                 + self.lambda_ssim * ssim_loss
                 + self.lambda_adv * adv_loss)

        return total, {
            'l1':   float(l1_loss.item()),
            'ssim': float(ssim_loss.item()),
            'adv':  float(adv_loss.item()),
        }
