# DIB-FNO

**Dynamic Information Bottleneck for Fourier Neural Operators**
面向中期天气预报的动态信息瓶颈傅里叶神经算子

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#-环境与安装)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org/)
[![Data](https://img.shields.io/badge/data-ARCO--ERA5-1f6feb.svg)](#-数据来源)
[![License](https://img.shields.io/badge/license-未指定-lightgrey.svg)](#-许可证)

> 一句话：在 AFNO / FNO 主干前插入一个**输入相关的谱掩码生成器**，用信息瓶颈原则
> 动态决定保留哪些傅里叶模态，从而在大气场预报中自适应地区分有用信号与噪声频段。

---

## 📖 目录

- [核心方法](#-核心方法)
- [仓库结构](#-仓库结构)
- [环境与安装](#-环境与安装)
- [快速开始](#-快速开始)
- [数据来源](#-数据来源)
- [模型与基线](#-模型与基线)
- [评估指标](#-评估指标)
- [结果性质（务必先读）](#-结果性质务必先读)
- [论文图形重建包](#-论文图形重建包figures)
- [引用与参考](#-引用与参考)
- [许可证](#-许可证)

---

## 🧠 核心方法

传统 FNO 在频域保留固定的截断模态，无法区分「天气尺度有用信号」与「小尺度噪声」。
DIB-FNO 让掩码**随输入样本变化**：

| 组件 | 位置 | 作用 |
| --- | --- | --- |
| `SpectralMaskGenerator` | `models/spectral_mask.py` | 由大气场编码出逐样本、逐频段的软掩码 $M(\omega)$ |
| `SpectralFilter` / `AFNOBlock` | `models/afno.py` | 在谱域执行 $Z=\mathcal{F}^{-1}[\mathcal{F}[X]\odot M(\omega)]$ 并做 AFNO 结构化 MLP |
| `DIBFNO` | `models/dibfno.py` | 组合谱掩码 + N×AFNO + 输出投影，`forward(x, return_mask=True)` 返回 `(pred, mask)` |
| `DIBLoss` | `training/loss.py` | 预报损失 + 稀疏正则 $\lambda$（带 warmup） |
| `TemperatureScheduler` | `training/loss.py` | Gumbel-Softmax 温度 $\tau: 1.0 \to 0.1$ 退火，训练后期趋于硬掩码 |

**任务设定**：给定当前时刻大气场 $X(t)$，预报 **6 小时后**的场 $X(t+6h)$，并以自回归
滚动方式评估 1–10 天预报技巧。

---

## 📂 仓库结构

```
DIB_FNO/
├── main.py                     # 主入口（CLI：probe / train / eval / full / quick）
├── run.py                      # 便捷包装，等价于 python main.py --mode quick
├── config.py                   # 配置：ModelConfig / TrainingConfig / DataConfig / EvalConfig
│
├── models/                     # 模型定义
│   ├── spectral_mask.py        #   谱掩码生成器（DIB 核心）
│   ├── afno.py                 #   SpectralFilter / AFNOBlock
│   ├── dibfno.py               #   DIB-FNO 主体
│   └── baselines.py            #   FourCastNet / AdaptFNO 基线
├── data/                       # 数据管道
│   ├── grib_reader.py          #   本地 GRIB/NetCDF（cfgrib 引擎）
│   ├── era5_reader.py          #   ARCO-ERA5 Zarr 读取 + 本地 .npy 缓存
│   └── dataset.py              #   GribCacheDataset / ERA5CacheDataset / SyntheticDataset
├── training/                   # 训练
│   ├── trainer.py              #   Trainer（AMP、梯度累加、梯度检查点）
│   ├── loss.py                 #   DIBLoss + TemperatureScheduler
│   └── metrics.py              #   纬度加权 RMSE/ACC、SSIM、气候态
├── evaluation/                 # 评估
│   ├── forecast.py             #   多步自回归预报评估
│   ├── ablation.py             #   λ 消融实验
│   └── typhoon.py              #   台风路径 / 强度误差
├── visualization/              # 出图（22 个 plot_* 函数）
│   ├── plots.py
│   └── style.py
├── utils/                      # 设备选择、torch.compile、FLOPs/参数量统计
│
├── figures/                    # ★ 全部图形资产（见下方说明）
│   ├── demo_en/                #   演示图（英文文件名，旧单文件流程产出）
│   ├── demo_zh/                #   演示图（中文文件名，当前模块化流程产出）
│   ├── paper/                  #   论文图形重建包：24 幅图（PNG + 矢量 PDF）
│   ├── paper_tables/           #   6 张表格重排图
│   ├── paper_data/             #   论文表格锚点 CSV + 台风最佳路径 CSV
│   ├── paper_scripts/          #   重建包的出图脚本（固定种子，可复现）
│   ├── paper_README.md         #   重建报告：10 条原文矛盾裁决 + 数据来源分级
│   └── paper_PROTOCOL.md       #   技术协议：两个命名实验规格、公式、锚点
│
├── docs/                       # 实现计划与设计文档
│
├── dibfno_best.pth             # 训练产出的最佳权重（演示用）
├── requirements.txt            # 核心依赖
└── requirements-figures.txt    # 图形重建包依赖
```

> **单一实现**：仓库只保留模块化包这一份实现（`main.py` + `models/` `data/` `training/`
> `evaluation/` `visualization/` `utils/`）。早期重定义了同名模型的单文件原型
> （`dibfno_validation.py`，1476/1707 行）已删除，避免同一套模型存在多份实现导致分叉。

---

## ⚙ 环境与安装

```bash
git clone https://github.com/haohandongming-ship-it/DIB_FNO.git
cd DIB_FNO

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 1) 先按 CUDA 版本安装 PyTorch（见 https://pytorch.org/get-started/locally/）
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1

# 2) 其余依赖
pip install -r requirements.txt
```

要求 Python 3.9+；CPU 可运行（自动降级：关闭 AMP、限制线程数）。

---

## 🚀 快速开始

```bash
# 最快跑通全流程：合成数据 + 5° 小分辨率 + 20 epoch（CPU 可跑）
python main.py --mode quick
# 或
python run.py
```

产物：`dibfno_best.pth`，以及 `figures/demo_zh/` 下的 22 张图（可用 `--output-dir` 改路径，
默认不会在仓库根目录散落 PNG）。

### 命令行参数

```bash
python main.py --mode {probe,train,eval,full,quick}   # 运行模式（默认 full）
                --config {quick,paper,grib}           # 配置预设（默认 quick）
                --grib-path PATH                      # 本地 GRIB（默认读 DIBFNO_GRIB_PATH）
                --device {auto,cuda,cpu}              # 计算设备（默认 auto）
                --epochs N                            # 覆盖训练轮数
                --batch-size N                        # 覆盖批次大小
                --output-dir DIR                      # 出图目录（默认 figures/demo_zh）
                --compile                             # 启用 torch.compile
```

| 模式 | 说明 |
| --- | --- |
| `probe` | 只探测 GRIB 文件结构（变量、网格、时次），不训练 |
| `train` | 只训练 |
| `eval` | 只评估（复用已有配置） |
| `full` | 数据探测 → 训练 → 完整评估（论文级流程） |
| `quick` | 合成数据小分辨率端到端验证 |

| 预设 | 数据源 | 分辨率 | 网格 | hidden / blocks | epochs |
| --- | --- | --- | --- | --- | --- |
| `quick` | 合成数据 | 5.0° | 36 × 72 | 64 / 2 | 20 |
| `paper` | GRIB | 0.25° | 720 × 1440 | 256 / 4 | 80 |
| `grib` | GRIB | 0.25° | 720 × 1440 | 256 / 4 | 80 |

### 用真实数据训练

```bash
# 方式 A：本地 GRIB 文件（推荐，可先用 --mode probe 查看结构）
python main.py --mode full --config grib --grib-path /path/to/data.grib

# 方式 B：直接用 ARCO-ERA5 Zarr（GCS 匿名访问，首次会缓存到本地 .npy）
python -c "
from data import ERA5CacheReader
ds = ERA5CacheReader(variables=['msl','u10','v10','t2m'], years=(2020,2020))
print(ds.img_shape, len(ds))
"
```

数据源路径均可通过环境变量覆盖，无需改动代码：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DIBFNO_GRIB_PATH` | `./data.grib` | 本地 GRIB 文件 |
| `DIBFNO_ZARR_PATH` | GCS ARCO-ERA5 Zarr | 远程 Zarr 数据源 |
| `DIBFNO_CACHE_DIR` | `./era5_cache` | `.npy` 本地缓存目录 |
| `DIBFNO_COASTLINE` | `figures/paper_data/ne_110m_coastline.geojson` | 海岸线 GeoJSON（重建包用） |
| `DIBFNO_FIGROOT` | `figures/` | 重建包输出根目录 |

---

## 🌐 数据来源

本项目**不附带任何原始气象数据**。所有可用数据源及其官方入口如下：

| 数据源 | 用途 | 链接 |
| --- | --- | --- |
| **ARCO-ERA5**（Google Cloud 公共数据集） | 主数据源：0.25° 逐 6 小时再分析场，GCS 匿名访问 | [ARCO-ERA5 说明](https://github.com/google-research/arco-era5) · [Google Cloud 数据集页](https://console.cloud.google.com/marketplace/product/noaa-public/ecmwf-era5) |
| **ERA5**（ECMWF 官方再分析） | 原始 ERA5 下载（CDS） | [Copernicus CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) · [ERA5 文档](https://confluence.ecmwf.int/display/CKB/ERA5) |
| **WeatherBench 2** | 预报评估基准与基线参考 | [weatherbench2](https://github.com/google-research/weatherbench2) · [文档](https://weatherbench2.readthedocs.io/) |
| **Natural Earth 110m 海岸线** | 论文重建包地图底图（公有领域） | [110m coastline](https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-coastline/) |
| **IBTrACS**（NOAA NCEI） | 台风路径与强度真值 | [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) |
| **JMA RSMC 最佳路径** | 西太平洋台风最佳路径 | [JMA best track](https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/besttrack.html) |

代码中的默认数据源常量（`config.py` / `data/era5_reader.py`）：

```python
zarr_path = "gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr"
storage_options = {"token": "anon"}   # 匿名访问，无需凭证
```

气压层变量名映射（`data/era5_reader.py`）：

| 论文记号 | ERA5 变量名 |
| --- | --- |
| `z500` / `z850` … | `geopotential`（除以 9.80665 得位势高度） |
| `t850` | `temperature` |
| `u850` / `v850` | `u_component_of_wind` / `v_component_of_wind` |
| `r500` | `relative_humidity` |
| `q850` | `specific_humidity` |

**引用数据时请遵循各数据源自身的许可与致谢要求。**

---

## 🧩 模型与基线

| 模型 | 说明 | 代码 |
| --- | --- | --- |
| **DIB-FNO** | 本方法：动态谱掩码 + AFNO 主干 + 信息瓶颈损失 | `models/dibfno.py` |
| FourCastNet | 基线：AFNO 主干，无动态掩码 | `models/baselines.py` |
| AdaptFNO | 基线：自适应 FNO 变体 | `models/baselines.py` |

`utils/flops.py` 统一以 `(1, C, H, W)` 单样本前向统计**参数量**与**估算 FLOPs**，
便于与基线做等量对比。

---

## 📊 评估指标

| 指标 | 函数 | 说明 |
| --- | --- | --- |
| 纬度加权 RMSE | `training/metrics.py: lat_weighted_rmse` | 按 $\cos(\text{lat})$ 加权，避免极区网格过密导致失真 |
| 纬度加权 ACC | `training/metrics.py: lat_weighted_acc` | 相对气候态的距平相关系数 |
| 每变量分解 | `per_variable_metrics` | 分变量 RMSE / ACC |
| SSIM（涡度场） | `compute_ssim_for_pair` | $10^\circ \times 10^\circ$ 窗口上的结构相似度 |
| 台风路径误差 | `evaluation/typhoon.py: compute_track_error` | 大圆距离（km） |
| 台风强度误差 | `evaluate_typhoon_tracks` | 最大风速 / 最低气压误差 |
| 掩码保留率 | `DIBFNO.forward(..., return_mask=True)` | 间接衡量信息瓶颈压缩程度 |

---

## ⚠ 结果性质（务必先读）

仓库里存在**三类性质完全不同**的图形，请不要混为一谈：

| 类别 | 位置 | 生成方式 | 可否当作实验结果 |
| --- | --- | --- | --- |
| **流程演示图** | `figures/demo_en/`、`figures/demo_zh/` | 由 `main.py --mode quick` 产出：**合成数据 + 合成台风路径** | ❌ 仅供跑通流程演示 |
| **论文图形重建** | `figures/paper/`、`figures/paper_tables/` | 按论文表格**锚点数值反推绘制**（illustrative reconstruction） | ⚠ A 类可引用，C 类**不得**引用 |
| **表格锚点数据** | `figures/paper_data/` | 论文表格数值的 CSV 转写 + JMA 最佳路径 | ✅ 数值本身来自论文/官方 |

具体到演示图，`main.py` 在以下环节使用合成/模拟数据：

- 无本地缓存时数据来自 `SyntheticDataset`（`data/dataset.py`）；
- 台风路径由 `generate_synthetic_typhoon_tracks()` 合成，预报轨迹由
  `simulate_typhoon_forecast_tracks(model_rmse=...)` 按模型 RMSE 模拟；
- `补充图_每变量RMSE和ACC随预报时效变化.png` 的**每变量曲线**由整体多步 RMSE/ACC
  按变量索引做确定性缩放得到（`rmse × (1+0.1·i)`、`acc × (1−0.05·i)`），
  反映的是相对量级，**不是**逐变量的独立测量值。

**要得到真实结论**，需接入上表中的真实 ERA5 数据重跑，并把台风评估替换为真实
best track 数据集。

### 重建包的数据来源分级

`figures/paper_README.md` 把重建图分为三类，并逐图在左下角打了标签：

| 类别 | 含义 | 代表图 |
| --- | --- | --- |
| **A verbatim** | 数值 100% 取自论文表格/正文，仅重绘 | `fig01`、`fig06`、`fig14`、`figC1`、`figC3`、表 1–6 |
| **B mixed** | 部分数值来自论文，其余为示意重建 | `fig02`、`fig03`、`fig07`、`fig13`、`fig19` |
| **C reconstructed** | 无训练日志，全为锚点拟合的示意重建 | `fig04`、`fig05`、`fig08`–`fig12`、`fig15`–`fig18`、`fig20`、`fig21`、`figC2` |

该文件还记录了论文原文中 **10 处自相矛盾之处**（训练轮数 40/50/20、保留率 34% vs
0.5→0.1、复杂度两套数字、变量集与单位混乱、高频保留结论正反颠倒等）及其裁决方式，
并把实验拆分为 **Experiment P**（权威主实验）与 **Experiment C**（参考实验），
两者数字**不得混用**。详见 `figures/paper_README.md` 与 `figures/paper_PROTOCOL.md`。

---

## 🎨 论文图形重建包（`figures/`）

该包**不训练模型、不需要 GPU**，仅按论文已报告数值重绘图形，全部随机过程固定种子。

```bash
pip install -r requirements-figures.txt
cd figures/paper_scripts
python make_figures_a.py      # 图 1–7
python make_figures_b.py      # 图 8–12、14、21
python make_figures_c.py      # 图 13、15、16
python make_figures_d.py      # 图 17–20
python make_experiment_c.py   # figC1–figC3（Experiment C）
python make_tables.py         # 表 1–6 的 CSV 与表格图
```

产物：`figures/paper/`（每幅 300 dpi PNG + 可编辑矢量 PDF）、`figures/paper_tables/`、
`figures/paper_data/tables/`。脚本用 `DIBFNO_FIGROOT` 解析输出根目录，仓库放在任何
路径都能运行；未提供海岸线 GeoJSON 时自动跳过海岸线叠加而不会报错。

> **目录与显示名对应**：`figures/demo_zh/` 下用中文命名（如
> `图2-3_RMSE和ACC随预报时效变化.png`）的图，与 `figures/demo_en/` 下英文命名的图
> 是**两次不同运行**的产出（标签语言不同、数值略有差异），并非同一文件的副本。

---

## 📚 引用与参考

本项目实现所依据/对比的方法：

- **FourCastNet** — Pathak et al., *FourCastNet: A Global Data-driven High-resolution
  Weather Model using Adaptive Fourier Neural Operators*, 2022.
  [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)
- **AFNO** — Guibas et al., *Adaptive Fourier Neural Operators: Efficient Token Mixers
  for Transformers*, 2021. [arXiv:2111.13587](https://arxiv.org/abs/2111.13587)
- **FNO** — Li et al., *Fourier Neural Operator for Parametric Partial Differential
  Equations*, 2020. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- **ERA5** — Hersbach et al., *The ERA5 global reanalysis*, QJRMS, 2020.
  [doi:10.1002/qj.3803](https://doi.org/10.1002/qj.3803)

数据致谢：ECMWF / Copernicus Climate Change Service（ERA5）、Google Research
（ARCO-ERA5）、NOAA NCEI（IBTrACS）、JMA（RSMC 最佳路径）、Natural Earth（海岸线）。

---

## 📄 许可证

本仓库**未附许可证文件**。如需开源使用、修改或再分发，请先补充 `LICENSE`
（例如 [MIT](https://choosealicense.com/licenses/mit/) 或
[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)）。

注意：仓库中的第三方数据（ERA5、ARCO-ERA5、IBTrACS、Natural Earth）**不适用**本仓库
任何许可，需遵循各自数据源的条款。
