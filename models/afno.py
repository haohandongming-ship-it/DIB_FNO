"""
AFNO (Adaptive Fourier Neural Operator) 块
包含频域滤波和 AFNO 变换块
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class SpectralFilter(nn.Module):
    """频域滤波器：应用掩码 M(ω) 到频域"""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 空间域输入
            mask: (B, 1, H, W//2+1) 频域掩码
        Returns:
            x_filtered: (B, C, H, W) 滤波后的空间域
        """
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        mask_expanded = mask.expand(-1, C, -1, -1).to(x_fft.dtype)
        return torch.fft.irfft2(x_fft * mask_expanded, s=(H, W), norm='ortho')


class AFNOBlock(nn.Module):
    """
    AFNO 变换块
    核心操作: LayerNorm → FFT → 频域Conv → IFFT → MLP → 残差连接
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        modes: int = 16,
        use_checkpoint: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.fft_conv = nn.Conv2d(dim, dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
        )
        self.modes = modes
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # LayerNorm + 转置
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        # FFT → 频域裁剪 → Conv → IFFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        fft_crop = x_fft[:, :, :self.modes, :self.modes]
        conv_out = self.fft_conv(fft_crop.real)
        x_fft_mod = x_fft.clone()
        x_fft_mod[:, :, :self.modes, :self.modes] = conv_out.to(dtype=x_fft.dtype)
        x = torch.fft.irfft2(x_fft_mod, s=(H, W), norm='ortho')

        # MLP
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x + residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return grad_checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class AFNOBlockFixed(nn.Module):
    """固定模式 AFNO 块（用于基线模型）"""

    def __init__(self, dim: int, modes: int = 64, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.fft_conv = nn.Conv2d(dim, dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.modes = modes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x_fft = torch.fft.rfft2(x, norm='ortho')
        fft_crop = x_fft[:, :, :self.modes, :self.modes]
        conv_out = self.fft_conv(fft_crop.real)
        x_fft_mod = x_fft.clone()
        x_fft_mod[:, :, :self.modes, :self.modes] = conv_out.to(dtype=x_fft.dtype)
        x = torch.fft.irfft2(x_fft_mod, s=(H, W), norm='ortho')

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x + residual