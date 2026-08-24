"""ResUNet for MRI artifact removal (denoising + motion reduction).

Architecture:
  - 4-level encoder with residual blocks and MaxPool
  - 512-channel bottleneck
  - 4-level decoder with bilinear upsample + skip connections
  - Global residual: output = input + net(input)
    → network predicts the artifact / noise to subtract
  - Instance Normalization (tolerates small batches, variable resolutions)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from task_1b.config import BASE_CH, N_LEVELS, BOTTLE_CH


class ResBlock(nn.Module):
    """Two-conv residual block with Instance Normalization."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.block(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )
        self.block = ResBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch from odd-dimension inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    """Residual U-Net for single-channel MRI denoising.

    Input:  (B, 1, H, W)  — normalized [0, 1] MRI slice
    Output: (B, 1, H, W)  — enhanced slice (clamped to [0, 1])

    The network learns a residual: output = clamp(input + residual, 0, 1).
    """

    def __init__(
        self,
        base_ch: int  = BASE_CH,
        n_levels: int = N_LEVELS,
        bottle_ch: int = BOTTLE_CH,
        in_ch: int = 1,
    ):
        super().__init__()
        channels = [base_ch * (2 ** i) for i in range(n_levels)]  # [32,64,128,256]

        # Encoder
        self.encoders = nn.ModuleList()
        prev = in_ch
        for ch in channels:
            self.encoders.append(EncoderBlock(prev, ch))
            prev = ch

        # Bottleneck
        self.bottleneck = ResBlock(prev, bottle_ch)

        # Decoder
        self.decoders = nn.ModuleList()
        dec_in = bottle_ch
        for ch in reversed(channels):
            self.decoders.append(DecoderBlock(dec_in, ch, ch))
            dec_in = ch

        # Output head — predicts residual
        self.out = nn.Conv2d(channels[0], in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        feat = x
        for enc in self.encoders:
            feat, skip = enc(feat)
            skips.append(skip)

        feat = self.bottleneck(feat)

        for dec, skip in zip(self.decoders, reversed(skips)):
            feat = dec(feat, skip)

        residual = self.out(feat)
        return torch.clamp(x + residual, 0.0, 1.0)


def build_model(device: torch.device) -> ResUNet:
    model = ResUNet()
    return model.to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
