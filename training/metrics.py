"""
评估指标模块
包含: 纬度加权 RMSE, ACC, SSIM, 每变量指标分解, 涡度计算
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_lat_weights(latitudes: np.ndarray, H: int) -> torch.Tensor:
    """计算纬度加权权重（cos(lat) 归一化）"""
    lat_rad = np.deg2rad(latitudes)
    weights = np.cos(lat_rad)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def lat_weighted_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
) -> float:
    """纬度加权 RMSE"""
    diff = (pred - target) ** 2
    if lat_weights is not None:
        w = lat_weights.to(pred.device).view(1, 1, -1, 1)
        diff = diff * w
    return torch.sqrt(diff.mean()).item()


def lat_weighted_acc(
    pred: torch.Tensor,
    target: torch.Tensor,
    clim_mean: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
) -> float:
    """纬度加权 ACC (异常相关系数)"""
    pa = pred - clim_mean.to(pred.device)
    ta = target - clim_mean.to(pred.device)
    if lat_weights is not None:
        w = lat_weights.to(pred.device).view(1, 1, -1, 1)
        pa, ta = pa * w, ta * w
    num = (pa * ta).sum()
    den = torch.sqrt((pa ** 2).sum() * (ta ** 2).sum())
    return (num / (den + 1e-8)).item()


def per_variable_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    var_names: List[str],
    clim_mean: Optional[torch.Tensor] = None,
    lat_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """每变量指标分解"""
    results = {}
    for i, var in enumerate(var_names):
        p = pred[:, i:i + 1]
        t = target[:, i:i + 1]
        rmse = lat_weighted_rmse(p, t, lat_weights)
        if clim_mean is not None:
            cm = clim_mean[:, i:i + 1].to(pred.device)
            acc = lat_weighted_acc(p, t, cm, lat_weights)
        else:
            acc = 0.0
        results[var] = {"rmse": rmse, "acc": acc}
    return results


def compute_clim_mean(dataset, in_channels: int, device: torch.device) -> torch.Tensor:
    """从数据集计算气候态均值"""
    n_samples = min(50, len(dataset))
    samples = [dataset[i][0] for i in range(n_samples)]
    clim = torch.stack(samples).mean(dim=(0, 2, 3), keepdim=True)
    return clim.to(device)


# ==============================================================
# 涡度与 SSIM
# ==============================================================

def compute_vorticity(
    field_u: np.ndarray,
    field_v: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """从 u, v 风场计算涡度 (∂v/∂x - ∂u/∂y)"""
    dv_dx = (field_v[:, 2:] - field_v[:, :-2]) / (2 * dx)
    du_dy = (field_u[2:, :] - field_u[:-2, :]) / (2 * dy)
    return dv_dx[:, 1:-1] - du_dy[1:, :-1]


def compute_ssim(
    field1: np.ndarray,
    field2: np.ndarray,
    window_size: int = 11,
) -> float:
    """结构相似性指数 (SSIM)"""
    if window_size > min(field1.shape):
        window_size = max(3, min(field1.shape) // 2 * 2 + 1)

    data_range = field1.max() - field1.min()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    f1 = torch.tensor(field1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    f2 = torch.tensor(field2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)

    mu1 = F.conv2d(f1, kernel, padding=window_size // 2).squeeze().numpy()
    mu2 = F.conv2d(f2, kernel, padding=window_size // 2).squeeze().numpy()
    sigma1_sq = F.conv2d(f1 ** 2, kernel, padding=window_size // 2).squeeze().numpy() - mu1 ** 2
    sigma2_sq = F.conv2d(f2 ** 2, kernel, padding=window_size // 2).squeeze().numpy() - mu2 ** 2
    sigma12 = F.conv2d(f1 * f2, kernel, padding=window_size // 2).squeeze().numpy() - mu1 * mu2

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(np.mean(num / (den + 1e-8)))


def compute_ssim_for_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
    var_idx_u: int = 2,
    var_idx_v: int = 3,
    window_size: int = 11,
) -> float:
    """计算预报和真值的涡度 SSIM"""
    try:
        pred_u = pred[0, var_idx_u].cpu().numpy()
        pred_v = pred[0, var_idx_v].cpu().numpy()
        tgt_u = target[0, var_idx_u].cpu().numpy()
        tgt_v = target[0, var_idx_v].cpu().numpy()
        vort_pred = compute_vorticity(pred_u, pred_v)
        vort_tgt = compute_vorticity(tgt_u, tgt_v)
        return compute_ssim(vort_pred, vort_tgt, window_size)
    except Exception:
        return 0.0


# ==============================================================
# 验证指标汇总
# ==============================================================

@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    clim_mean: Optional[torch.Tensor] = None,
    lat_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """验证一个 epoch，返回 (loss, rmse, acc)"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0
    acc_num_sum = acc_den1_sum = acc_den2_sum = 0.0

    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred, mask = model(x, return_mask=True)
        loss, _, _ = criterion(pred, y, mask)
        total_loss += loss.item()
        total_mse += F.mse_loss(pred, y, reduction='sum').item()
        total_samples += x.size(0)

        if clim_mean is not None:
            pa = pred - clim_mean.to(device)
            ta = y - clim_mean.to(device)
            if lat_weights is not None:
                w = lat_weights.to(device).view(1, 1, -1, 1)
                pa, ta = pa * w, ta * w
            acc_num_sum += (pa * ta).sum().item()
            acc_den1_sum += (pa ** 2).sum().item()
            acc_den2_sum += (ta ** 2).sum().item()

    avg_loss = total_loss / len(dataloader)
    n_elements = total_samples * pred.size(1) * pred.size(2) * pred.size(3)
    rmse = (total_mse / n_elements) ** 0.5 if n_elements > 0 else 0.0
    acc = acc_num_sum / ((acc_den1_sum * acc_den2_sum) ** 0.5 + 1e-8) if clim_mean is not None else 0.0

    return avg_loss, rmse, acc