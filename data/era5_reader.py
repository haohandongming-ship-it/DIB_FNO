"""
ERA5 本地缓存数据读取器
支持从 Google Cloud ARCO-ERA5 Zarr 下载并缓存到本地 .npy
"""

import os
import warnings
import logging
from typing import List, Optional, Tuple, Dict
import numpy as np
import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ERA5CacheReader:
    """ERA5 本地缓存版数据集读取器"""

    # 可能的气压层变量到 ARCO-ERA5 变量名映射
    PRESSURE_LEVEL_VARS = {
        'z': 'geopotential',
        't': 'temperature',
        'u': 'u_component_of_wind',
        'v': 'v_component_of_wind',
        'r': 'relative_humidity',
        'q': 'specific_humidity',
    }

    def __init__(
        self,
        zarr_path: str = "gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr",
        variables: Optional[List[str]] = None,
        years: Tuple[int, int] = (2020, 2020),
        lead_time: int = 6,
        time_resolution: int = 6,
        cache_dir: str = "./era5_cache",
        storage_options: Optional[Dict] = None,
    ):
        if variables is None:
            variables = ["msl", "u10", "v10", "t2m"]
        self.variables = variables
        self.lead_steps = lead_time // time_resolution
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self._img_shape = None
        self._latitudes = None
        self._n_samples = 0

        # 尝试加载已有缓存
        meta_path = os.path.join(cache_dir, "meta.npz")
        if os.path.exists(meta_path):
            logger.info(f"从本地缓存加载: {cache_dir}")
            try:
                meta = np.load(meta_path, allow_pickle=True)
                self._n_samples = int(meta['n_samples'])
                self._img_shape = tuple(meta['img_shape'])
                self._latitudes = meta['latitudes']
                return
            except Exception as e:
                logger.warning(f"缓存元数据损坏: {e}")

        # 首次运行，从 GCS 下载
        logger.info(f"首次运行，从 GCS 下载到 {cache_dir} ...")
        if storage_options is None:
            storage_options = {"token": "anon"}

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*default credentials.*")
                warnings.filterwarnings("ignore", message=".*bucket type.*")
                ds = xr.open_zarr(
                    zarr_path, consolidated=True, storage_options=storage_options
                )
        except Exception as e:
            raise RuntimeError(f"无法打开 Zarr: {e}")

        time_slice = slice(f"{years[0]}-01-01", f"{years[1]}-12-31")
        ds = ds.sel(time=time_slice)[variables]
        stride = max(1, time_resolution // 1)
        ds = ds.isel(time=slice(0, None, stride))

        first_var = variables[0]
        n_values = ds.sizes.get('values', 0)
        self._img_shape = self._infer_grid_shape(ds, first_var, n_values)
        logger.info(f"网格形状: {self._img_shape}")

        if 'latitude' in ds[first_var].coords:
            self._latitudes = np.unique(ds[first_var].coords['latitude'].values)
        else:
            self._latitudes = np.linspace(90, -90, self._img_shape[0])

        n_time = len(ds.time)
        logger.info(f"转换 {n_time} 个时间步到 .npy ...")
        for t_idx in tqdm(range(n_time), desc="缓存中"):
            out_path = os.path.join(cache_dir, f"t_{t_idx:06d}.npy")
            if not os.path.exists(out_path):
                t_data = ds.isel(time=t_idx)
                arrays = [t_data[var].values.reshape(self._img_shape) for var in variables]
                arr = np.stack(arrays, axis=0).astype(np.float32)
                np.save(out_path, arr)

        self._n_samples = n_time - self.lead_steps
        np.savez(
            meta_path,
            n_samples=self._n_samples,
            img_shape=np.array(self._img_shape),
            latitudes=self._latitudes,
        )
        logger.info(f"缓存完成: {self._n_samples} 个样本")

    @staticmethod
    def _infer_grid_shape(ds, first_var, n_values):
        if 'latitude' in ds[first_var].coords and 'longitude' in ds[first_var].coords:
            lat = ds[first_var].coords['latitude'].values
            lon = ds[first_var].coords['longitude'].values
            n_lat, n_lon = len(np.unique(lat)), len(np.unique(lon))
            if n_lat * n_lon == n_values:
                return (n_lat, n_lon)
        for h, w in [(721, 752), (721, 1440), (361, 720), (181, 360), (36, 72)]:
            if h * w == n_values:
                return (h, w)
        h = int(n_values ** 0.5)
        return (h, n_values // h)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def img_shape(self) -> Tuple[int, int]:
        return self._img_shape

    @property
    def latitudes(self) -> np.ndarray:
        return self._latitudes