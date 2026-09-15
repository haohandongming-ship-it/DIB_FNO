"""
损失函数模块
DIB 损失: 预测损失 + 稀疏性正则化
温度退火调度器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DIBLoss(nn.Module):
    """
    DIB 损失函数
    L = L_pred + β * L_sparsity
    其中 L_sparsity 是掩码保留率的平均值
    """

    def __init__(
        self,
        sparsity_weight: float = 0.001,
        warmup_epochs: int = 20,
        use_l1: bool = False,
    ):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.warmup_epochs = warmup_epochs
        self.use_l1 = use_l1
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_beta(self) -> float:
        """线性预热 β"""
        if self.current_epoch < self.warmup_epochs:
            return self.sparsity_weight * self.current_epoch / max(self.warmup_epochs, 1)
        return self.sparsity_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Returns:
            total_loss, pred_loss, sparsity
        """
        pred_loss = F.mse_loss(pred, target)

        if mask is not None:
            if self.use_l1:
                sparsity = mask.abs().mean()
            else:
                sparsity = mask.mean()
        else:
            sparsity = torch.tensor(0.0, device=pred.device)

        beta = self.get_beta()
        total_loss = pred_loss + beta * sparsity

        return total_loss, pred_loss, sparsity


class TemperatureScheduler:
    """
    温度退火调度器
    τ(t) = τ_init * (τ_final/τ_init)^(t/T)
    """

    def __init__(
        self,
        init_temp: float = 1.0,
        final_temp: float = 0.1,
        total_epochs: int = 50,
    ):
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs
        self.decay_rate = (final_temp / init_temp) ** (1.0 / max(total_epochs, 1))

    def get_temperature(self, epoch: int) -> float:
        return self.init_temp * (self.decay_rate ** epoch)

    def get_temperature_linear(self, epoch: int) -> float:
        """线性退火替代方案"""
        progress = min(epoch / max(self.total_epochs, 1), 1.0)
        return self.init_temp + (self.final_temp - self.init_temp) * progress