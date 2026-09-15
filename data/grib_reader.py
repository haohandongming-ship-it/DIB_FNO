"""
GRIB 格式气象数据读取器
支持 cfgrib 引擎读取 GRIB1/GRIB2 格式数据
"""

import os
import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class GribReader:
    """GRIB 数据读取器，支持大规模 GRIB 文件的高效读取与缓存"""

    def __init__(
        self,
        grib_path: str,
        variables: Optional[List[str]] = None,
        filter_keys: Optional[Dict] = None,
        cache_dir: str = "./grib_cache",
    ):
        """
        Args:
            grib_path: GRIB 文件路径
            variables: 需要读取的变量名列表
            filter_keys: 额外的过滤条件，如 {'typeOfLevel': 'surface'}
            cache_dir: 缓存目录
        """
        self.grib_path = grib_path
        self.variables = variables
        self.filter_keys = filter_keys or {}
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self._ds = None
        self._metadata = None

    @property
    def dataset(self) -> xr.Dataset:
        """延迟加载 GRIB 数据集"""
        if self._ds is None:
            self._ds = self._open_grib()
        return self._ds

    def _open_grib(self) -> xr.Dataset:
        """打开 GRIB 文件，支持后端选择"""
        logger.info(f"正在打开 GRIB 文件: {self.grib_path}")

        try:
            import cfgrib
            # 尝试用 cfgrib 打开
            ds = xr.open_dataset(
                self.grib_path,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": self.filter_keys if self.filter_keys else None,
                    "indexpath": os.path.join(self.cache_dir, "grib.idx"),
                },
                chunks="auto",  # 自动分块以支持大文件
            )
            logger.info(f"cfgrib 成功打开，数据集维度: {dict(ds.dims)}")
            return ds

        except (ImportError, ValueError) as e:
            logger.warning(f"cfgrib 打开失败 ({e})，尝试其他引擎")

        try:
            import pygrib
            ds = xr.open_dataset(self.grib_path, engine="pynio")
            return ds
        except ImportError:
            pass

        try:
            # 最后尝试：使用 eccodes 直接读取
            ds = xr.open_dataset(
                self.grib_path,
                engine="cfgrib",
                backend_kwargs={
                    "errors": "ignore",
                    "indexpath": os.path.join(self.cache_dir, "grib.idx"),
                },
            )
            return ds
        except Exception as e:
            raise RuntimeError(
                f"无法打开 GRIB 文件 {self.grib_path}。"
                f"请安装 cfgrib: pip install cfgrib\n"
                f"原始错误: {e}"
            )

    def get_metadata(self) -> Dict:
        """获取 GRIB 文件元数据"""
        if self._metadata is not None:
            return self._metadata

        try:
            ds = self.dataset
            metadata = {
                "variables": list(ds.data_vars),
                "dims": dict(ds.dims),
                "coords": list(ds.coords),
                "grid_shape": None,
                "n_timesteps": 0,
                "file_size_gb": os.path.getsize(self.grib_path) / 1e9,
            }

            # 推断网格形状
            if "latitude" in ds.coords and "longitude" in ds.coords:
                lat = np.unique(ds.coords["latitude"].values)
                lon = np.unique(ds.coords["longitude"].values)
                metadata["grid_shape"] = (len(lat), len(lon))
                metadata["lat_range"] = (float(lat.min()), float(lat.max()))
                metadata["lon_range"] = (float(lon.min()), float(lon.max()))

            if "time" in ds.coords:
                metadata["n_timesteps"] = len(ds.coords["time"])
                metadata["time_range"] = (
                    str(ds.coords["time"].values[0]),
                    str(ds.coords["time"].values[-1]),
                )

            self._metadata = metadata
            return metadata

        except Exception as e:
            logger.error(f"读取元数据失败: {e}")
            return {"error": str(e)}

    def read_variable(self, var_name: str, time_index: int = 0) -> np.ndarray:
        """读取单个变量在指定时间步的数据"""
        ds = self.dataset
        if var_name not in ds.data_vars:
            raise ValueError(f"变量 '{var_name}' 不在数据集中。可用变量: {list(ds.data_vars)}")

        if "time" in ds[var_name].dims:
            data = ds[var_name].isel(time=time_index).values
        else:
            data = ds[var_name].values

        return data.astype(np.float32)

    def read_timestep(self, time_index: int) -> np.ndarray:
        """读取指定时间步的所有变量数据，返回 (C, H, W) 数组"""
        ds = self.dataset
        var_list = self.variables if self.variables else list(ds.data_vars)
        if not var_list:
            var_list = list(ds.data_vars)

        arrays = []
        for var in var_list:
            if var in ds.data_vars:
                data = self.read_variable(var, time_index)
                arrays.append(data)
            else:
                logger.warning(f"变量 '{var}' 不在数据集中，跳过")

        if not arrays:
            raise RuntimeError(f"时间步 {time_index} 没有可读取的变量")

        return np.stack(arrays, axis=0).astype(np.float32)

    def get_n_timesteps(self) -> int:
        """获取总时间步数"""
        try:
            if "time" in self.dataset.coords:
                return len(self.dataset.coords["time"])
            return 1
        except Exception:
            return 1

    def get_grid_shape(self) -> Tuple[int, int]:
        """获取网格形状 (H, W)"""
        meta = self.get_metadata()
        if meta.get("grid_shape"):
            return meta["grid_shape"]
        # 从数据推断
        first_var = list(self.dataset.data_vars)[0]
        data = self.dataset[first_var].values
        if data.ndim >= 2:
            return data.shape[-2], data.shape[-1]
        return (720, 1440)  # 默认 0.25°

    def get_latitudes(self) -> np.ndarray:
        """获取纬度数组"""
        ds = self.dataset
        if "latitude" in ds.coords:
            return np.unique(ds.coords["latitude"].values)
        if "lat" in ds.coords:
            return np.unique(ds.coords["lat"].values)
        # 默认
        return np.linspace(90, -90, self.get_grid_shape()[0])

    def cache_to_npy(self, max_timesteps: Optional[int] = None) -> int:
        """将 GRIB 数据转换为 .npy 缓存（加速后续读取）"""
        n_timesteps = self.get_n_timesteps()
        if max_timesteps:
            n_timesteps = min(n_timesteps, max_timesteps)

        logger.info(f"正在缓存 {n_timesteps} 个时间步到 {self.cache_dir}...")

        count = 0
        for t in range(n_timesteps):
            out_path = os.path.join(self.cache_dir, f"t_{t:06d}.npy")
            if not os.path.exists(out_path):
                try:
                    arr = self.read_timestep(t)
                    np.save(out_path, arr)
                    count += 1
                except Exception as e:
                    logger.warning(f"时间步 {t} 缓存失败: {e}")

        logger.info(f"缓存完成: {count} 个新文件")
        return count

    def close(self):
        """关闭数据集，释放资源"""
        if self._ds is not None:
            self._ds.close()
            self._ds = None


def probe_grib_file(grib_path: str) -> Dict:
    """探测 GRIB 文件结构（不完整加载）"""
    reader = GribReader(grib_path)
    return reader.get_metadata()