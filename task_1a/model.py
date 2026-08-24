"""
model.py — Clasificador ordinal con backbone intercambiable (EfficientNet-B4, ConvNeXt, ...).

Arquitectura IDÉNTICA a la OrdinalClassifierV9 original (shared: Dropout->Linear->ReLU->BN1d;
dos cabezas Dropout->Linear), de modo que los checkpoints best_1a_v9_fold*.pth existentes
cargan sin conversión. Genérico para cualquier backbone de timm (num_features se infiere).
"""
import timm
import torch.nn as nn


class OrdinalClassifier(nn.Module):
    def __init__(self, backbone: str = 'efficientnet_b4', n_artifacts: int = 7,
                 dropout1: float = 0.4, dropout2: float = 0.3, hidden_dim: int = 512,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.shared = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
        )
        self.head_ge1 = nn.Sequential(nn.Dropout(dropout2), nn.Linear(hidden_dim, n_artifacts))
        self.head_ge2 = nn.Sequential(nn.Dropout(dropout2), nn.Linear(hidden_dim, n_artifacts))

    def forward(self, x):
        shared = self.shared(self.backbone(x))
        return self.head_ge1(shared), self.head_ge2(shared)


# Alias de compatibilidad con checkpoints v9 (mismo módulo, mismos nombres de params)
OrdinalClassifierV9 = OrdinalClassifier
