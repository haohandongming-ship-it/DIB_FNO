"""
PyTorch Dataset 类
支持 GRIB 缓存、ERA5 缓存、合成数据三种数据源
"""

import os
import logging
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GribCacheDataset(Dataset):
    """基于 GRIB 数据 .npy 缓存的 PyTorch Dataset"""

    def __init__(
        self,
        cache_dir: str,
        variables: List[str],
        lead_steps: int = 1,
        transform=None,
    ):
        self.cache_dir = cache_dir
        self.variables = variables
        self.lead_steps = lead_steps
        self.transform = transform

        # 加载元数据
        meta_path = os.path.join(cache_dir, "meta.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"缓存元数据不存在: {meta_path}。请先运行 GribReader.cache_to_npy()"
            )
        meta = np.load(meta_path, allow_pickle=True)
        self._n_samples = int(meta["n_samples"])
        self._img_shape = tuple(meta["img_shape"])
        self._latitudes = meta["latitudes"]

        # 验证缓存文件存在
        self._cached_files = []
        for t in range(self._n_samples + self.lead_steps):
            fpath = os.path.join(cache_dir, f"t_{t:06d}.npy")
            if os.path.exists(fpath):
                self._cached_files.append(fpath)

        logger.info(f"GRIB缓存数据集: {self._n_samples} 样本, "
                    f"网格: {self._img_shape}, 变量: {self.variables}")

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = np.load(self._cached_files[idx])
        y = np.load(self._cached_files[idx + self.lead_steps])

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    @property
    def img_shape(self) -> Tuple[int, int]:
        return self._img_shape

    @property
    def latitudes(self) -> np.ndarray:
        return self._latitudes


class ERA5CacheDataset(Dataset):
    """基于 ERA5 .npy 缓存的 PyTorch Dataset"""

    def __init__(
        self,
        cache_dir: str,
        lead_steps: int = 1,
        transform=None,
    ):
        self.cache_dir = cache_dir
        self.lead_steps = lead_steps
        self.transform = transform

        meta_path = os.path.join(cache_dir, "meta.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"缓存元数据不存在: {meta_path}")

        meta = np.load(meta_path, allow_pickle=True)
        self._n_samples = int(meta["n_samples"])
        self._img_shape = tuple(meta["img_shape"])
        self._latitudes = meta["latitudes"]

        logger.info(f"ERA5缓存数据集: {self._n_samples} 样本, 网格: {self._img_shape}")

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_path = os.path.join(self.cache_dir, f"t_{idx:06d}.npy")
        y_path = os.path.join(self.cache_dir, f"t_{idx + self.lead_steps:06d}.npy")
        x = np.load(x_path)
        y = np.load(y_path)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    @property
    def img_shape(self) -> Tuple[int, int]:
        return self._img_shape

    @property
    def latitudes(self) -> np.ndarray:
        return self._latitudes


class SyntheticDataset(Dataset):
    """合成数据 Dataset（用于快速测试）"""

    def __init__(
        self,
        img_shape: Tuple[int, int],
        n_variables: int = 4,
        n_samples: int = 200,
        noise_scale: float = 0.1,
        seed: int = 42,
    ):
        self._img_shape = img_shape
        self._n_samples = n_samples - 1
        self._latitudes = np.linspace(90, -90, img_shape[0])

        rng = np.random.RandomState(seed)
        self.data = (
            rng.randn(n_samples, n_variables, *img_shape).astype(np.float32)
            * noise_scale + 1.0
        )

        logger.info(f"合成数据集: {self._n_samples} 样本, 网格: {img_shape}, "
                    f"变量: {n_variables}")

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.data[idx].copy()),
            torch.from_numpy(self.data[idx + 1].copy()),
        )

    @property
    def img_shape(self) -> Tuple[int, int]:
        return self._img_shape

    @property
    def latitudes(self) -> np.ndarray:
        return self._latitudes


def create_dataloaders(
    dataset: Dataset,
    batch_size: int,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 42,
):
    """创建训练/验证 DataLoader"""
    from torch.utils.data import DataLoader, random_split

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_loader, val_loader


def create_dataset_from_config(data_config) -> Dataset:
    """根据配置创建数据集"""
    source = data_config.source

    if source == "grib":
        # 首先检查 GRIB 缓存
        cache_dir = "./grib_cache"
        meta_path = os.path.join(cache_dir, "meta.npz")
        if not os.path.exists(meta_path):
            # 需要先缓存 GRIB 数据
            from .grib_reader import GribReader
            reader = GribReader(
                grib_path=data_config.grib_path,
                variables=data_config.variables,
                cache_dir=cache_dir,
            )
            logger.info("GRIB 缓存不存在，开始转换...")
            reader.cache_to_npy()
            reader.close()

        return GribCacheDataset(
            cache_dir=cache_dir,
            variables=data_config.variables,
            lead_steps=data_config.lead_time // data_config.time_resolution,
        )

    elif source == "zarr":
        from .era5_reader import ERA5CacheReader
        reader = ERA5CacheReader(
            zarr_path=data_config.zarr_path,
            variables=data_config.variables,
            years=data_config.years,
            lead_time=data_config.lead_time,
            time_resolution=data_config.time_resolution,
            cache_dir=data_config.cache_dir,
        )
        return ERA5CacheDataset(
            cache_dir=data_config.cache_dir,
            lead_steps=data_config.lead_time // data_config.time_resolution,
        )

    elif source == "cache":
        cache_dir = data_config.cache_dir
        meta_path = os.path.join(cache_dir, "meta.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"缓存不存在: {meta_path}")
        return ERA5CacheDataset(
            cache_dir=cache_dir,
            lead_steps=data_config.lead_time // data_config.time_resolution,
        )

    elif source == "synthetic":
        return SyntheticDataset(
            img_shape=data_config.img_shape,
            n_variables=len(data_config.variables),
            n_samples=200,
        )

    else:
        raise ValueError(f"未知数据源: {source}")