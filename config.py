"""
DIB-FNO 全局配置模块
支持：快速验证 / 论文完整 / 自定义 三种预设，以及 GRIB 数据源配置
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import os

# ==============================================================
# 预设配置
# ==============================================================

@dataclass
class ModelConfig:
    """模型架构配置"""
    hidden_dim: int = 256
    n_blocks: int = 4
    modes: int = 16
    use_checkpoint: bool = False
    mask_hidden_dim: int = 64


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    accumulation_steps: int = 1
    sparsity_weight: float = 0.001
    warmup_epochs: int = 16
    init_temperature: float = 1.0
    final_temperature: float = 0.1
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class DataConfig:
    """数据配置

    数据源优先级（见 README「数据来源」表）:
      1. 本地 GRIB/NetCDF 文件   —— 设 DIBFNO_GRIB_PATH 或 --grib-path
      2. ARCO-ERA5 Zarr（GCS 匿名访问）—— 无需凭证
         gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr
      3. 合成数据（source='synthetic'）—— 仅用于跑通流程
    """
    # 数据源: 'grib' | 'zarr' | 'synthetic' | 'cache'
    source: str = "grib"
    # 不再硬编码本机绝对路径：可用环境变量 DIBFNO_GRIB_PATH 覆盖，
    # 或用命令行 --grib-path 指定。
    grib_path: str = os.environ.get("DIBFNO_GRIB_PATH", "./data.grib")
    zarr_path: str = os.environ.get(
        "DIBFNO_ZARR_PATH",
        "gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr",
    )
    cache_dir: str = os.environ.get("DIBFNO_CACHE_DIR", "./era5_cache")
    resolution: str = "0.25°"
    img_shape: Tuple[int, int] = (720, 1440)
    variables: List[str] = field(default_factory=lambda: [
        "msl", "u10", "v10", "t2m"
    ])
    # 全量20变量
    full_variables: List[str] = field(default_factory=lambda: [
        "z500", "z700", "z850", "z1000",
        "t500", "t700", "t850",
        "u850", "u1000",
        "v850", "v1000",
        "r500", "r700", "r850",
        "q500", "q700", "q850",
        "msl", "t2m", "u10", "v10",
    ])
    years: Tuple[int, int] = (2020, 2020)
    lead_time: int = 6
    time_resolution: int = 6
    train_ratio: float = 0.8
    # GRIB 过滤参数
    grib_filter_keys: Optional[Dict] = None


@dataclass
class EvalConfig:
    """评估配置"""
    n_steps_eval: List[int] = field(default_factory=lambda: [4, 8, 16, 24, 40])
    max_batches_eval: int = 10
    ablation_lambda_values: List[float] = field(default_factory=lambda: [
        0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
    ])
    ablation_epochs: int = 10
    ssim_window_size: int = 11
    paper_var_names: List[str] = field(default_factory=lambda: [
        'Z500', 'T850', 'U850', 'R500'
    ])


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # 运行时
    device: str = "auto"  # auto | cuda | cpu
    seed: int = 42
    output_dir: str = "./figures/demo_zh"   # 出图目录（避免污染仓库根目录）
    run_name: str = "dibfno_run"


# ==============================================================
# 预设工厂
# ==============================================================

def get_quick_config() -> Config:
    """快速验证配置（小分辨率、少epoch）"""
    return Config(
        model=ModelConfig(hidden_dim=64, n_blocks=2, modes=16),
        training=TrainingConfig(
            batch_size=32, epochs=20, learning_rate=1e-3,
            accumulation_steps=2, use_amp=False,
        ),
        data=DataConfig(
            source="synthetic",
            resolution="5.0°",
            img_shape=(36, 72),
            variables=["msl", "u10", "v10", "t2m"],
        ),
        eval=EvalConfig(
            n_steps_eval=[1, 4, 8],
            max_batches_eval=5,
            ablation_epochs=5,
        ),
    )


def get_paper_config() -> Config:
    """论文级完整配置"""
    return Config(
        model=ModelConfig(hidden_dim=256, n_blocks=4, modes=16, use_checkpoint=True),
        training=TrainingConfig(
            batch_size=32, epochs=80, learning_rate=5e-4,
            accumulation_steps=1,
        ),
        data=DataConfig(
            source="grib",
            resolution="0.25°",
            img_shape=(720, 1440),
            variables=["msl", "u10", "v10", "t2m"],
        ),
        eval=EvalConfig(
            n_steps_eval=[4, 8, 16, 24, 40],
            max_batches_eval=10,
            ablation_epochs=10,
        ),
    )


def get_grib_config(grib_path: str = None, variables: List[str] = None) -> Config:
    """基于 GRIB 数据的配置"""
    cfg = get_paper_config()
    cfg.data.source = "grib"
    if grib_path:
        cfg.data.grib_path = grib_path
    if variables:
        cfg.data.variables = variables
    return cfg