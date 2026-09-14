# DIB-FNO: Dynamic Information Bottleneck for Fourier Neural Operators

面向**中期天气预报**的傅里叶神经算子（FNO）改进实现：在潜空间频谱上引入**动态信息瓶颈（Dynamic Information Bottleneck, DIB）**，让网络自适应地保留对预报有用的频段、抑制噪声频段，并用纬度加权 RMSE / ACC、多步自回归预报、台风路径与强度误差等指标完成完整验证。

本仓库是论文《DIB-FNO: Dynamic Information Bottleneck for Fourier Neural Operators》的**可复现实验与出图代码**。

---

## 1. 核心思想

传统 FNO 在频域固定保留截断模态，无法区分「天气尺度有用信号」与「小尺度噪声」。DIB-FNO 的做法是：

1. **频谱掩码生成器**（`SpectralMaskGenerator`）：由当前大气场编码出一个与输入样本相关的软掩码（soft mask），逐样本、逐频段决定信息通过率；
2. **温度退火**（`TemperatureScheduler`）：Gumbel-Softmax 温度 $\tau$ 从 `1.0` 退火到 `0.1`，训练初期探索、后期趋于硬掩码，同时追踪掩码保留率；
3. **信息瓶颈正则**（`DIBLoss`）：主预报损失 + 稀疏正则 $\lambda$（带 warmup），显式压缩频域信息量；
4. **AFNO 算子块**：在 FNO 谱卷积基础上做结构化 MLP（AFNO），配合梯度检查点与 FFT 优化降低显存占用。

因此模型输出为 `(pred, mask)` 二元组，`mask` 可直接用于可视化「模型关注哪些频段」。

## 2. 任务设定

| 项目 | 设定 |
| --- | --- |
| 数据源 | Google Cloud ARCO-ERA5（`gs://gcp-public-data-arco-era5`，0.25°） |
| 变量 | 默认 4 个地面变量 `msl / u10 / v10 / t2m`；另内置 20 变量方案（气压层 + 地面层） |
| 预报任务 | 给定当前时刻大气场，预报 **6 小时后**的场（`lead_time=6h`） |
| 预报时效评估 | 1–10 天多步自回归（`n_steps_eval`） |
| 训练/验证 | 时序数据按 8:2 划分 |

脚本内置两套配置：

- `QUICK_CONFIG`：5° 分辨率 `(36, 72)`、hidden 64、2 blocks、20 epochs —— **可快速跑通全流程**（默认使用）；
- `PAPER_CONFIG`：0.25° 分辨率 `(720, 1440)`、hidden 256、4 blocks、80 epochs —— 论文级配置。

## 3. 仓库结构

```
DIB_FNO/
├── dibfno_validation.py        # 主脚本：数据、模型、训练、评估、出图（约 1500 行）
├── dibfno_best.pth             # 验证流程中保存的最佳权重（演示用）
├── DIB-FNO_代码优化与论文完善计划.md      # 实现计划
├── DIB-FNO_代码优化与论文完善计划plus.md  # 实现计划（plus 版，脚本按此实现）
├── fig1_architecture.png       # Fig.1  整体架构图
├── rmse_acc_vs_lead.png        # Fig.2-3 RMSE / ACC vs 预报时效
├── per_var_rmse_acc.png        #        每变量 RMSE / ACC 分解
├── model_comparison_bar.png    # Fig.4  三模型柱状对比
├── fig5_typhoon_tracks.png     # Fig.5  台风路径预报对比
├── fig6_typhoon_intensity.png  # Fig.6  台风强度预报误差
├── dibfno_mask.png             # Fig.7/8 频谱掩码可视化
├── mask_comparison.png         # Fig.7-9 台风期 vs 安静期掩码对比
├── spectral_analysis.png       #        频谱掩码详细分析
├── ablation.png                # Fig.10-11 消融实验（λ → 保留率 → RMSE）
├── training_curves.png         # Fig.12-14 训练曲线
├── lr_schedule.png             #        学习率调度
├── forecast_fields.png         # Fig.15 预报场可视化
├── forecast_fields_multi.png   # Fig.15 多时效预报场（6h / 24h / 48h）
├── model_complexity_table.png  #        模型参数量 / FLOPs 对比
├── fig11_spatial_error.pdf     # Fig.11 空间误差分布（矢量版）
├── fig12_spatial_error_matrix.pdf  # Fig.12 空间误差矩阵（矢量版）
└── figures/                    # 论文图形重建交付包（见第 9 节）
    ├── README.md               #   重建报告：10 条矛盾裁决 + 数据来源分级
    ├── PROTOCOL.md             #   技术协议：两个命名实验规格、公式、锚点
    ├── figures/                #   21 幅 Experiment P 图 + 3 幅 Experiment C 图
    │                           #   （每幅 300 dpi PNG + 可编辑矢量 PDF，共 55 个文件）
    ├── data/tables/*.csv       #   Table 1–6 原始数据（数字与论文一致）
    ├── data/tracks/*.csv       #   Rai / Chanthu / Noru 的 JMA 最佳路径
    └── scripts/*.py            #   出图脚本（common.py + make_figures_a~d + make_experiment_c + make_tables）
```

> 根目录的 `fig11_*` / `fig12_*` 两个 PDF 是空间误差分析的矢量出图，当前版本 `dibfno_validation.py` 未直接生成它们。注意它们与 `figures/figures/` 下的同名文件**内容不同**（各自保留，未做覆盖）。

> `figures.zip`（约 20 MB）是 `figures/` 的打包副本，包含 `README.md`、`PROTOCOL.md`、
> `figures/`、`data/`、`scripts/` 共 73 个条目，便于一次性下载整包；其内容与 `figures/`
> 目录逐字节相同（已用 SHA256 全量核验）。两者只需其一即可。

## 4. 环境依赖

- Python 3.9+
- PyTorch（建议 CUDA 版本；也支持纯 CPU 运行）
- 其余依赖：

```bash
pip install torch numpy xarray zarr gcsfs matplotlib tqdm
```

> 说明：`from torch.cuda.amp import autocast, GradScaler` 需要较新的 PyTorch（1.10+，且未移除 `torch.cuda.amp`）。

## 5. 快速开始

```bash
python dibfno_validation.py
```

脚本会自动按 10 个阶段执行并打印日志，无需命令行参数：

```
[1/10]  数据加载（本地缓存优先，缺失则使用内置合成数据）
[2/10]  构建 DIB-FNO / FourCastNet / AdaptFNO 三模型并统计参数量与 FLOPs
[3/10]  训练 DIB-FNO（AMP + 梯度累加 + cosine 学习率 + 温度退火）
[4/10]  多步自回归预报评估
[5/10]  每变量指标分解与模型对比
[6/10]  SSIM 涡度场分析（u10/v10 导出涡度）
[7/10]  台风路径 / 强度误差评估
[8/10]  消融实验（λ 扫描）
[9/10]  掩码、频谱与预报场可视化
[10/10] 架构图与模型复杂度表
```

### 真实 ERA5 数据

首次运行会从 GCS 下载并缓存为本地 `.npy`（`./era5_cache/`），此后直接读本地，约 10–50× 加速：

```python
from dibfno_validation import ERA5LocalCache
ds = ERA5LocalCache(variables=["msl", "u10", "v10", "t2m"],
                    years=(2020, 2020), lead_time=6, cache_dir="./era5_cache")
```

缓存目录被 `.gitignore` 忽略，不会入库。

## 6. 模型与基线

| 模型 | 说明 |
| --- | --- |
| **DIB-FNO** | 本方法：AFNO 块 + 动态频谱掩码 + 信息瓶颈损失 |
| FourCastNet | 基线：AFNO 主干，无动态掩码 |
| AdaptFNO | 基线：自适应 FNO 变体 |

`estimate_flops_and_params()` 统一以 `(1, C, H, W)` 输入统计**参数量**与**估算 FLOPs**，结果见 `model_complexity_table.png`。

## 7. 评估指标

- **纬度加权 RMSE**（`compute_lat_weighted_rmse`）：按 $\cos(\text{lat})$ 加权，避免极区网格过密导致指标失真；
- **纬度加权 ACC**（`compute_lat_weighted_acc`）：相对气候态的距平相关系数；
- **每变量指标分解**（`compute_per_variable_metrics`）；
- **SSIM**（`compute_ssim` / `compute_ssim_for_pair`）：在 $10^\circ \times 10^\circ$ 窗口上评估涡度场结构相似度；
- **台风路径误差**（`compute_track_error`，大圆距离）与**强度误差**；
- **频谱掩码保留率**：间接衡量信息瓶颈的压缩程度。

## 8. 重要说明（结果性质）

本仓库包含**两套互不相同的内容**，引用时请勿混淆：

### 8.1 根目录：代码 + 流程演示图

根目录出图所用的 `main()` 流程在**数据与台风评估部分使用了合成/模拟数据**：

- 若本地不存在 `./era5_cache/meta.npz`，脚本回退到 `_SyntheticDataset`（`_create_synthetic_dataset`）生成的数据；
- 台风路径与强度由 `generate_synthetic_typhoon_tracks()` 合成，预报轨迹由 `simulate_typhoon_forecast_tracks(model_rmse=...)` 依模型 RMSE 模拟生成；
- 每变量多时效曲线由整体 multi-step 结果按变量索引做比例缩放得到（见 `main()` 中 `[5/10]` 段）。

因此根目录的 PNG **是流程演示图（demo figures），不是论文最终实验结论**。要复现论文数值，需要按第 5 节接入真实 ERA5 数据后重跑，并把台风评估替换为真实最佳路径（best track）数据集。

### 8.2 `figures/`：论文图形的**示意性重建**，不得当作实验结果

`figures/` 是**论文图形重建交付包**，其自身 README 已明确声明其中绝大多数图为
**illustrative reconstruction（示意性重建）**——即按论文表格中的锚点数值反推绘制的曲线，
**不是模型运行输出**。该包把图形分为三类并逐图标注数据来源标签：

| 类别 | 含义 | 代表图 |
| --- | --- | --- |
| **A verbatim** | 数值 100% 取自论文表格/正文，仅重绘 | fig01、fig06、fig14、figC1、figC3、表 1–6 |
| **B mixed** | 部分数值来自论文，其余为示意重建 | fig02、fig03、fig07、fig13、fig19 |
| **C reconstructed** | 无任何训练日志，全为锚点拟合的示意重建 | fig04、fig05、fig08–fig12、fig15–fig18、fig20、fig21、figC2 |

**引用红线**：C 类图不得作为实验结果引用；`figures/README.md` 还记录了论文原文中 10 处
自相矛盾之处（训练轮数、保留率、复杂度两套数字、变量集与单位混乱、高频保留结论正反颠倒等）
及其裁决方式，并把实验拆分为 **Experiment P**（权威主实验）与 **Experiment C**（参考实验），
两者数字**不得混用**。详细内容以 `figures/README.md` 与 `figures/PROTOCOL.md` 为准。

## 9. 复现 `figures/` 重建包

```bash
cd figures/scripts
python make_figures_a.py && python make_figures_b.py && python make_figures_c.py \
  && python make_figures_d.py && python make_experiment_c.py && python make_tables.py
```

依赖 `numpy / scipy / matplotlib / pandas / geopandas`；所有随机过程使用固定种子（20240517），结果可复现。

> `.gitignore` 中 `/data/` 只忽略仓库根目录的数据目录，`figures/data/`（表格锚点与最佳路径 CSV）
> 通过 `!figures/data/**` 反向放行，以确保交付数据入库。

## 10. 许可证

本仓库未附许可证文件；如需开源使用，请先补充 `LICENSE`。
