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
└── fig12_spatial_error_matrix.pdf  # Fig.12 空间误差矩阵（矢量版）
```

> `fig11_*` / `fig12_*` 两个 PDF 为空间误差分析的矢量出图，当前版本的 `dibfno_validation.py` 未直接生成它们，属于随项目一起归档的既有产物。

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

本仓库出图所用的 `main()` 流程在**数据与台风评估部分使用了合成/模拟数据**：

- 若本地不存在 `./era5_cache/meta.npz`，脚本回退到 `_SyntheticDataset`（`_create_synthetic_dataset`）生成的数据；
- 台风路径与强度由 `generate_synthetic_typhoon_tracks()` 合成，预报轨迹由 `simulate_typhoon_forecast_tracks(model_rmse=...)` 依模型 RMSE 模拟生成；
- 每变量多时效曲线由整体 multi-step 结果按变量索引做比例缩放得到（见 `main()` 中 `[5/10]` 段）。

因此仓库内的 PNG **是流程演示图（demo figures），不是论文最终实验结论**。要复现论文数值，需要按第 5 节接入真实 ERA5 数据后重跑，并把台风评估替换为真实最佳路径（best track）数据集。

## 9. 许可证

本仓库未附许可证文件；如需开源使用，请先补充 `LICENSE`。
