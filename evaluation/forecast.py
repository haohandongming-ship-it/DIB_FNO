"""
多步自回归预报评估模块
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from training.metrics import lat_weighted_rmse, lat_weighted_acc


@torch.no_grad()
def autoregressive_forecast(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    n_steps: int = 40,
) -> List[torch.Tensor]:
    """
    自回归多步预报
    Args:
        model: 训练好的模型
        x_init: (B, C, H, W) 初始场
        n_steps: 预报步数
    Returns:
        predictions: 包含初始场和每一步预报的列表
    """
    model.eval()
    predictions = [x_init]
    x = x_init.clone()
    for _ in range(n_steps):
        pred = model(x)
        predictions.append(pred)
        x = pred
    return predictions


@torch.no_grad()
def evaluate_multi_step(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    n_steps_list: List[int],
    lat_weights: Optional[torch.Tensor] = None,
    clim_mean: Optional[torch.Tensor] = None,
    max_batches: int = 10,
) -> Dict[int, Dict]:
    """
    多步预报评估
    Returns:
        {step: {"rmse": float, "acc": float, "hours": int}}
    """
    model.eval()
    rmse_by_step = {s: [] for s in n_steps_list}
    acc_by_step = {s: [] for s in n_steps_list}

    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        x = x.to(device)
        preds = autoregressive_forecast(model, x, n_steps=max(n_steps_list))

        for step in n_steps_list:
            pred_s = preds[step]
            rmse = lat_weighted_rmse(pred_s, y.to(device), lat_weights)
            rmse_by_step[step].append(rmse)
            if clim_mean is not None:
                acc = lat_weighted_acc(pred_s, y.to(device), clim_mean, lat_weights)
                acc_by_step[step].append(acc)

    result = {}
    for step in n_steps_list:
        result[step] = {
            "rmse": np.mean(rmse_by_step[step]) if rmse_by_step[step] else 0,
            "acc": np.mean(acc_by_step[step]) if acc_by_step[step] else 0,
            "hours": step * 6,
        }
    return result


@torch.no_grad()
def evaluate_all_models(
    models: Dict[str, torch.nn.Module],
    dataloader,
    device: torch.device,
    n_steps_list: List[int],
    lat_weights: Optional[torch.Tensor] = None,
    clim_mean: Optional[torch.Tensor] = None,
    max_batches: int = 10,
) -> Dict[str, Dict]:
    """评估所有模型的多步预报性能"""
    results = {}
    for name, model in models.items():
        model.eval()
        result = evaluate_multi_step(
            model, dataloader, device, n_steps_list,
            lat_weights, clim_mean, max_batches,
        )
        results[name] = result
        for step, v in result.items():
            print(f"  {name} @ {v['hours']}h: RMSE={v['rmse']:.4f}, ACC={v['acc']:.4f}")
    return results