"""
消融实验模块
评估不同 λ (sparsity weight) 对模型性能的影响
"""

import torch
import numpy as np
from typing import Dict, List, Optional

from models import DIBFNO
from training import DIBLoss, Trainer
from training.metrics import lat_weighted_rmse


def run_ablation_study(
    base_model_config: dict,
    train_loader,
    val_loader,
    device: torch.device,
    clim_mean: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
    lambda_values: List[float] = None,
    epochs_per: int = 10,
) -> List[Dict]:
    """
    消融实验: 扫描不同 λ 值

    Returns:
        [{"lambda": float, "rmse": float, "acc": float, "retention_ratio": float}, ...]
    """
    if lambda_values is None:
        lambda_values = [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    results = []
    for lam in lambda_values:
        print(f"\n--- Ablation: λ={lam:.0e} ---")

        model = DIBFNO(**base_model_config).to(device)
        trainer = Trainer(model, device, config={
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "sparsity_weight": lam,
            "warmup_epochs": max(epochs_per // 2, 1),
            "epochs": epochs_per,
            "use_amp": (device.type == "cuda"),
        })

        for ep in range(1, epochs_per + 1):
            trainer.train_epoch(train_loader, ep)

        val_loss, val_rmse, val_acc = trainer.validate(
            val_loader, clim_mean, lat_weights,
        )

        # 计算掩码保留率
        retention = 0.0
        n_ret = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                _, mask = model(x, return_mask=True)
                retention += mask.mean().item()
                n_ret += 1
                if n_ret >= 5:
                    break

        result = {
            "lambda": lam,
            "rmse": val_rmse,
            "acc": val_acc,
            "retention_ratio": retention / max(n_ret, 1),
        }
        results.append(result)
        print(f"  λ={lam:.0e}: RMSE={val_rmse:.4f}, ACC={val_acc:.4f}, "
              f"Retention={result['retention_ratio']:.3f}")

    return results