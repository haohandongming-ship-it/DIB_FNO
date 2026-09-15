"""
频谱掩码生成器 (Spectral Mask Generator)
CNN Encoder-Decoder 架构，生成频域掩码 M(ω)
"""

import torch
import torch.nn as nn
from typing import Tuple


class SpectralMaskGenerator(nn.Module):
    """
    频谱掩码生成器
    输入: 空间域气象场 x ∈ R^(B×C×H×W)
    输出: 频域掩码 M(ω) ∈ R^(B×1×H×(W//2+1))
    """

    def __init__(
        self,
        in_channels: int,
        freq_shape: Tuple[int, int],
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.freq_shape = freq_shape
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 3, 1, 1),
        )

        self.upsample = nn.Upsample(
            size=freq_shape, mode='bilinear', align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 空间域输入
        Returns:
            mask: (B, 1, H, W//2+1) 频域掩码 [0, 1]
        """
        feat = self.encoder(x)
        mask_map = self.decoder(feat)
        mask_map = self.upsample(mask_map)
        # 温度退火：温度越低，sigmoid 越陡峭，掩码越接近 0/1
        return torch.sigmoid(mask_map / self.temperature)