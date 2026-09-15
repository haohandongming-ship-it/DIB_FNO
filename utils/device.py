"""
设备管理工具：GPU/CPU 自适应，多GPU支持，混合精度配置
"""

import os
import torch
import multiprocessing
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto") -> torch.device:
    """获取最优可用设备"""
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA 不可用，回退到 CPU")
        return torch.device("cpu")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def configure_device(device: torch.device) -> dict:
    """配置设备相关参数，返回运行配置"""
    config = {
        "device": device,
        "use_amp": False,
        "num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": 2,
    }

    if device.type == "cuda":
        config["use_amp"] = True
        config["pin_memory"] = True
        config["num_workers"] = min(4, multiprocessing.cpu_count())
        _configure_cuda()

        # 多GPU检测
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            logger.info(f"检测到 {n_gpus} 个 GPU，可使用 DataParallel 或 DistributedDataParallel")
            config["multi_gpu"] = True
            config["n_gpus"] = n_gpus
        else:
            config["multi_gpu"] = False
    else:
        cpu_count = multiprocessing.cpu_count()
        num_threads = min(cpu_count, 8)
        torch.set_num_threads(num_threads)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        config["num_workers"] = min(4, cpu_count)

    return config


def _configure_cuda():
    """CUDA 优化配置"""
    if torch.cuda.is_available():
        # 启用 cuDNN benchmark 和 TF32
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True

        # 打印 GPU 信息
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU[{i}]: {props.name}, "
                        f"显存: {props.total_memory / 1024**3:.1f} GB, "
                        f"SM: {props.multi_processor_count}")


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    start_batch_size: int = 32,
    max_memory_ratio: float = 0.8,
) -> int:
    """自动搜索最优 batch size（基于 GPU 显存）"""
    if device.type != "cuda":
        return start_batch_size

    total_mem = torch.cuda.get_device_properties(device).total_memory
    available_mem = total_mem * max_memory_ratio

    try:
        # 估算单样本内存
        x = torch.randn(1, *input_shape[1:], device=device)
        model.eval()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.empty_cache()
        mem_per_sample = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        if mem_per_sample > 0:
            optimal = int(available_mem / (mem_per_sample * 2))  # 预留2x空间
            return max(1, min(optimal, start_batch_size * 2))
    except Exception:
        pass

    return start_batch_size


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """尝试使用 torch.compile 加速（PyTorch 2.0+）"""
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            return torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            logger.warning(f"torch.compile 失败: {e}")
    return model


def wrap_model_for_multi_gpu(
    model: torch.nn.Module,
    device: torch.device,
    strategy: str = "auto",
) -> Tuple[torch.nn.Module, bool]:
    """多GPU包装策略"""
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0

    if n_gpus <= 1:
        return model, False

    if strategy == "auto":
        strategy = "dp"  # DataParallel 最简单

    if strategy == "dp":
        model = torch.nn.DataParallel(model)
        logger.info(f"使用 DataParallel 包装 ({n_gpus} GPUs)")
        return model, True

    if strategy == "ddp":
        logger.info("DistributedDataParallel 需要 torchrun 启动，跳过自动包装")
        return model, False

    return model, False