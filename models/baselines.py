"""
基线模型: FourCastNet / AdaptFNO
用于与 DIB-FNO 对比
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .afno import AFNOBlockFixed


class FourCastNetBaseline(nn.Module):
    """FourCastNet 基线: 固定模式 AFNO 块，无频谱掩码"""

    def __init__(
        self,
        in_channels: int,
        img_shape: Tuple[int, int],
        hidden_dim: int = 256,
        n_blocks: int = 4,
        modes: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            AFNOBlockFixed(hidden_dim, modes=modes)
            for _ in range(n_blocks)
        ])
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, 1)

    def forward(
        self, x: torch.Tensor, return_mask: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = self.input_proj(x)
        for block in self.blocks:
            z = block(z)
        pred = self.output_proj(z)
        return (pred, None) if return_mask else pred


class AdaptFNOBaseline(nn.Module):
    """AdaptFNO 基线: 可学习频域权重，但无掩码生成器"""

    def __init__(
        self,
        in_channels: int,
        img_shape: Tuple[int, int],
        hidden_dim: int = 256,
        n_blocks: int = 4,
        modes: int = 16,
    ):
        super().__init__()
        H, W = img_shape
        self.in_channels = in_channels
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            AFNOBlockFixed(hidden_dim, modes=modes)
            for _ in range(n_blocks)
        ])
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, 1)
        # 可学习的频域权重
        self.mode_weights = nn.Parameter(torch.ones(1, 1, H, W // 2 + 1))

    def forward(
        self, x: torch.Tensor, return_mask: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = self.input_proj(x)
        for block in self.blocks:
            z = block(z)
        pred = self.output_proj(z)
        return (pred, self.mode_weights) if return_mask else pred