"""
DIB-FNO 主模型
Dynamic Information Bottleneck Fourier Neural Operator
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .spectral_mask import SpectralMaskGenerator
from .afno import SpectralFilter, AFNOBlock


class DIBFNO(nn.Module):
    """
    DIB-FNO: 动态信息瓶颈傅里叶神经算子

    架构:
        Input → MaskGen(M) → SpectralFilter(X*M) → InputProj → AFNO×N → OutputProj → Pred

    核心创新:
        1. 可学习的频谱掩码 M(ω)，实现动态信息瓶颈
        2. 频域滤波替代传统空间域注意力
        3. 温度退火控制掩码稀疏度
    """

    def __init__(
        self,
        in_channels: int,
        img_shape: Tuple[int, int],
        hidden_dim: int = 256,
        n_blocks: int = 4,
        modes: int = 16,
        mask_hidden_dim: int = 64,
        use_checkpoint: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        H, W = img_shape
        self.in_channels = in_channels
        self.img_shape = img_shape
        self.freq_shape = (H, W // 2 + 1)

        # 频谱掩码生成器
        self.mask_gen = SpectralMaskGenerator(
            in_channels, self.freq_shape, hidden_dim=mask_hidden_dim,
        )

        # 频谱滤波器
        self.spectral_filter = SpectralFilter()

        # 输入/输出投影
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, 1)

        # AFNO 变换块
        self.blocks = nn.ModuleList([
            AFNOBlock(hidden_dim, modes=modes, use_checkpoint=use_checkpoint,
                      dropout=dropout)
            for _ in range(n_blocks)
        ])

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        return_mask: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) 输入气象场
            return_mask: 是否返回掩码
        Returns:
            pred: (B, C, H, W) 预测场
            mask: (B, 1, H, W//2+1) 频谱掩码 (if return_mask=True)
        """
        # 1. 生成频谱掩码
        mask = self.mask_gen(x)

        # 2. 频域滤波
        x_filtered = self.spectral_filter(x, mask)

        # 3. 输入投影
        z = self.input_proj(x_filtered)

        # 4. AFNO 变换
        for block in self.blocks:
            z = block(z)

        # 5. 输出投影
        pred = self.output_proj(z)

        return (pred, mask) if return_mask else pred

    def set_temperature(self, temperature: float):
        """设置掩码生成器的温度参数"""
        self.mask_gen.temperature = temperature

    def get_temperature(self) -> float:
        return self.mask_gen.temperature