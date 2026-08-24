"""
losses.py — Focal loss ordinal con regularización de monotonicidad.

Movido desde train_v9_module.py (sin cambios de comportamiento). Dos cabezas
cumulativas: P(score>=1) y P(score>=2). El término de monotonicidad penaliza
P(>=2) > P(>=1), que violaría la definición ordinal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0,
                 alpha_ge1: torch.Tensor = None,
                 alpha_ge2: torch.Tensor = None,
                 w2: float = 4.0, w_mono: float = 0.5):
        super().__init__()
        self.gamma, self.w2, self.w_mono = gamma, w2, w_mono
        self.register_buffer('alpha_ge1', alpha_ge1) if alpha_ge1 is not None else setattr(self, 'alpha_ge1', None)
        self.register_buffer('alpha_ge2', alpha_ge2) if alpha_ge2 is not None else setattr(self, 'alpha_ge2', None)

    def _focal(self, logits, targets, alpha):
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t   = torch.exp(-bce)
        focal = ((1 - p_t) ** self.gamma) * bce
        if alpha is not None:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal   = alpha_t * focal
        return focal.mean()

    def forward(self, logits_ge1, logits_ge2, labels_ge1, labels_ge2):
        l1 = self._focal(logits_ge1, labels_ge1, self.alpha_ge1)
        l2 = self._focal(logits_ge2, labels_ge2, self.alpha_ge2)
        mono = F.relu(torch.sigmoid(logits_ge2) - torch.sigmoid(logits_ge1)).mean()
        total = l1 + self.w2 * l2 + self.w_mono * mono
        logs = {'l_ge1': float(l1.detach()), 'l_ge2': float(l2.detach()), 'l_mono': float(mono.detach())}
        return total, logs
