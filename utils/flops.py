"""
FLOPs 和参数量估算工具
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计参数量：总参数 / 可训练参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 4, 36, 72),
) -> Dict:
    """估算模型 FLOPs 和参数量"""
    params, trainable = count_parameters(model)
    C, H, W = input_shape[1], input_shape[2], input_shape[3]

    n_blocks = _get_n_blocks(model)
    hidden = _get_hidden_dim(model, 256)
    modes = _get_modes(model, 16)

    # FFT: O(C * H * W * log(HW))
    fft_flops = C * H * W * (math.log2(H) + math.log2(W)) * n_blocks
    # Conv: O(C * C * modes * modes)
    conv_flops = C * C * modes * modes * n_blocks
    # MLP: O(C * hidden * H * W * 2)
    mlp_flops = C * hidden * H * W * 2 * n_blocks

    total = fft_flops + conv_flops + mlp_flops

    return {
        "params": params,
        "trainable": trainable,
        "estimated_flops": total,
        "flops_readable": f"{total / 1e9:.2f}G",
        "params_readable": f"{params / 1e6:.2f}M",
        "breakdown": {
            "fft": fft_flops,
            "conv": conv_flops,
            "mlp": mlp_flops,
        },
    }


def _get_n_blocks(model: nn.Module) -> int:
    if hasattr(model, 'blocks'):
        return len(model.blocks)
    if hasattr(model, 'module') and hasattr(model.module, 'blocks'):
        return len(model.module.blocks)
    return 0


def _get_hidden_dim(model: nn.Module, default: int = 256) -> int:
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        block = model.blocks[0]
        if hasattr(block, 'mlp') and len(block.mlp) > 0:
            if hasattr(block.mlp[0], 'out_features'):
                return block.mlp[0].out_features
    if hasattr(model, 'module') and hasattr(model.module, 'blocks'):
        return _get_hidden_dim(model.module, default)
    return default


def _get_modes(model: nn.Module, default: int = 16) -> int:
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        return getattr(model.blocks[0], 'modes', default)
    if hasattr(model, 'module') and hasattr(model.module, 'blocks'):
        return _get_modes(model.module, default)
    return default


def compare_models_flops(
    models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...],
) -> Dict[str, Dict]:
    """对比多个模型的复杂度"""
    results = {}
    for name, model in models.items():
        results[name] = estimate_model_flops(model, input_shape)
    return results