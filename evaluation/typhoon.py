"""
台风路径预报评估模块
包含: 大圆距离计算、台风中心定位、合成台风路径生成
"""

import numpy as np
from typing import Dict, List


def compute_track_error(
    pred_lat: float, pred_lon: float,
    true_lat: float, true_lon: float,
) -> float:
    """大圆距离计算台风路径误差 (km)"""
    R = 6371.0
    lat1, lon1 = np.deg2rad(pred_lat), np.deg2rad(pred_lon)
    lat2, lon2 = np.deg2rad(true_lat), np.deg2rad(true_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def find_storm_center(
    field: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> tuple:
    """从 MSLP 场中定位台风中心（最小海平面气压）"""
    min_idx = np.argmin(field)
    lat_idx, lon_idx = np.unravel_index(min_idx, field.shape)
    return lat_grid[lat_idx], lon_grid[lon_idx]


def generate_synthetic_typhoon_tracks(
    n_steps: int = 40,
    seed: int = 42,
) -> Dict:
    """生成合成台风路径数据（用于快速验证）"""
    rng = np.random.RandomState(seed)
    tracks = {}
    for name in ["Rai", "Chanthu", "Noru"]:
        start_lat = rng.uniform(5, 25)
        start_lon = rng.uniform(120, 150)
        lats = [start_lat]
        lons = [start_lon]
        max_winds = [rng.uniform(30, 65)]
        min_pressures = [rng.uniform(930, 1000)]

        for _ in range(n_steps - 1):
            lats.append(lats[-1] + rng.uniform(-0.5, 1.0))
            lons.append(lons[-1] + rng.uniform(-0.5, 0.8))
            max_winds.append(max_winds[-1] + rng.uniform(-3, 3))
            min_pressures.append(min_pressures[-1] + rng.uniform(-5, 5))

        tracks[name] = {
            "time": np.arange(n_steps) * 6,
            "lat": np.array(lats),
            "lon": np.array(lons),
            "max_wind": np.clip(np.array(max_winds), 15, 80),
            "min_pressure": np.clip(np.array(min_pressures), 900, 1010),
        }
    return tracks


def simulate_typhoon_forecast_tracks(
    true_tracks: Dict,
    model_rmse: float,
    seed: int = 123,
) -> Dict:
    """模拟各模型预报的台风路径（基于真实路径 + 随机误差）"""
    rng = np.random.RandomState(seed)
    models = {
        "DIB-FNO": 0.3,
        "FourCastNet": 0.6,
        "AdaptFNO": 0.5,
    }
    pred_tracks = {}
    for model_name, noise_scale in models.items():
        pred_tracks[model_name] = {}
        for name, track in true_tracks.items():
            n = len(track["lat"])
            pred_tracks[model_name][name] = {
                "lat": track["lat"] + rng.randn(n) * noise_scale * 2,
                "lon": track["lon"] + rng.randn(n) * noise_scale * 2,
                "max_wind": track["max_wind"] + rng.randn(n) * noise_scale * 5,
                "min_pressure": track["min_pressure"] + rng.randn(n) * noise_scale * 3,
            }
    return pred_tracks


def evaluate_typhoon_tracks(
    true_tracks: Dict,
    pred_tracks: Dict,
    typhoon_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """评估台风路径预报误差"""
    results = {}
    for name in typhoon_names:
        results[name] = {}
        for model_name, pred in pred_tracks.items():
            errors = [
                compute_track_error(
                    pred[name]["lat"][i], pred[name]["lon"][i],
                    true_tracks[name]["lat"][i], true_tracks[name]["lon"][i],
                )
                for i in range(len(true_tracks[name]["lat"]))
            ]
            results[name][model_name] = np.mean(errors)
    return results