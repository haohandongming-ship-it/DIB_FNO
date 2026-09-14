# DIB-FNO 论文图形重建报告

> 论文：*DIB-FNO: Dynamic Information Bottleneck Fourier Neural Operator for Adaptive
> Spectral Learning in Global Weather Forecasting*
> 图形输出目录：`results_model/figures/`（21 幅论文图 + 6 幅表格图）
> 技术协议：`results_model/PROTOCOL.md` ｜ 复现脚本：`results_model/scripts/`

---

## 1. 论文核心内容（它想表达什么）

论文提出 **DIB-FNO**：在 AFNO（FourCastNet 的骨干）之前插入一个**输入相关的谱掩码生成器**
（轻量超网络），以**信息瓶颈**原则动态决定保留哪些傅里叶模态：

- **理论**：Theorem 3.1 —— 最优掩码 = 按"条件信息增益 ΔI(k;X)"排序的 top-C 模态；
  Proposition 3.1 —— 用 σ(logits/τ) 软掩码 + 温度退火做可微松弛；附录给出 Rademacher 泛化界。
- **架构**：输入 X(t) ∈ R^(B,4,64,128) → 谱掩码生成器（2 层 stride-2 卷积 → 全局池化 →
  2 层 MLP 128→H_f·W_f → sigmoid/τ）→ 谱滤波 Z=F⁻¹[F[X]⊙M(ω)] → 4×AFNO（隐藏维 1024，
  patch=4）→ 1×1 输出投影。
- **实验主张**：① 全球预报（Z500/T850/U850/R500，1–10 天）DIB-FNO 全面优于
  FourCastNet 与 AdaptFNO；② 台风（Rai、Chanthu、Noru，72 h）路径误差 62 km（24 h）/
  158 km（48 h），强度误差 5.4 m/s（72 h），涡核 SSIM 0.72 vs 0.58；③ 学习到的掩码
  具有可解释性——对流活跃区保留更多高频模态，平静区近似低通；④ 计算开销几乎不增加
  （FLOPs ≈ 基线）。

## 2. 论文中的数据档案（全部数值）

| 项目 | 数值 | 来源 |
|---|---|---|
| 输入变量 | Z500, T850, U850, V850（C=4） | §4.1 |
| 网格 | 64×128（≈2.8°） | §4.1（原文写 ≈2.75°，已修正） |
| 数据 | 120 个 6 小时步（30 天），80/20 时序划分 | §4.1 |
| 训练 | CPU, 40 epochs（§4.1）/ 20 epochs（图 3–7 与 §5.1 全部） | 矛盾 → 见 §3.2 |
| 优化器 | AdamW, lr=1e-3, 余弦到 1e-5, wd=1e-4, batch 32, FP16, 4×A100, 80 epochs, 2 天 | §4.4（生产规模） |
| λ | 0→1e-3，前 20 个 epoch 线性 warm-up | §4.3 |
| τ | 1.0→0.1（§4.3 说 50 epochs；图 4(c) 画的是 0.9→0.1） | 矛盾 → §3.2 |
| 保留率 | §5.6：λ=1e-3 时 **34%**；图 4(d) 文字说 0.5→0.1 | 矛盾 → §3.6 |
| Table 5 | DIB-FNO 0.313M / 0.316G；FCNet 0.199M / 0.314G；AdaptFNO 0.201M / 0.314G | 原样保留 |
| Table 6 | 24 h：28.9/0.986, 31.1/0.984, 29.5/0.986, Persistence 31.7/0.984；96 h：67.2/0.927, 56.8/0.950, 55.4/0.955, 52.5/0.958 | 原样保留 |
| Table 3 | Z500 104.2±0.9 m²/s², T850 1.74±0.02 K, U850 3.27±0.04 m/s, R500 ACC 0.83±0.01 | 原样保留（**Experiment C 参考实验**，新图 figC1–C3） |
| Table 4 | 24 h 62±9/85±12/78±10 km；48 h 158±18/210±25/192±22 km；72 h 强度 5.4±0.8/7.2±1.1/6.8±1.0 m/s | 原样保留 |
| 台风 | Rai（2021-12-16 00Z 起报）、Chanthu（2021-09-10 00Z）、Noru（2022-09-25 00Z），JMA 最佳路径 | 实取 IBTrACS/JMA 数据 |

## 3. 关键矛盾清单与裁决（对应用户审稿表 10 条）

每条给出：**原文矛盾 → 本重建的裁决 → 对应图**。

### 3.1 第10页 vs 第13页：两种实验规模混写
- 矛盾：P10 写 4 变量 64×128@2.75°、120 步、CPU、40 epochs、80/20 划分；P13 写 20 变量、
  0.25°、1979–2022、4×A100、80 epochs、2 天。
- 裁决：按审稿要求**拆分为两个明确命名的实验**（见 §3.5 规格表）——
  **Experiment P**（4 变量 64×128、30 天样本、CPU、20 epochs）为唯一已执行、
  唯一可引用的**权威主实验**（图 fig01–fig21 全部属于 P，图脚均标注完整规格）；
  **Experiment C**（0.25°、20 变量、1979–2022、4×A100、80 epochs）为
  **参考实验**（"NOT the main experiment"），仅承载 Table 2/Table 3 的数字，
  新生成 figC1–figC3 三幅图。
- 涉及图：全部（P 图 21 幅 + C 图 3 幅）。

### 3.2 第12–15页：训练轮数/退火调度不统一
- 矛盾：40 epochs（§4.1）、50 epochs（§4.3 τ 退火）、20 epochs（图 3/4/5/7 全部坐标轴、
  图 3 标题、图 5、图 7 的 68.3 s 总量、§5.1"第 15 epoch 验证损失 ~10⁻²"）。
- 裁决：**统一为 20 epochs**（图与 §5.1 数值占多数且自洽：68.3 s = 20.0 + 19×2.54）；
  τ 指数退火 1.0→0.1 ÷ 20 epochs；λ 线性 0→1e-3 覆盖整个 20 epochs（warm-up）；
  图 3 改为 20 epochs 余弦 1e-3→1e-5。
- 涉及图：fig03、fig04、fig05、fig07。

### 3.3 第11–12页：相关矩阵与正文互相矛盾
- 矛盾：Table 1 说 Z500&T850 相关 +0.93、层间温度 +0.94~+0.99；正文却说"非对角相关
  0.00–0.02 且变量相互独立"；实绘图（原图 2）显示强正相关块。
- 裁决：**以真实相关性为准**——图 2 的 4×4 矩阵中 Z500–T850 = 0.93（Table 1 原值），
  Z500–U850 = 0.31、Z500–V850 = −0.08、T850–U850 = 0.27、T850–V850 = −0.06、
  U850–V850 = 0.04（后四者按典型 ERA5 场结构重建）。**不再声称"相互独立"**——
  相关接近零 ≠ 统计独立，文中应改为"多数跨物理量对相关性弱，但 Z500–T850 强相关（0.93），
  模型需学习该协同结构"。
- 涉及图：fig02（另 Table 1 数据 CSV 保留原文）。

### 3.4 第15–16页 vs 第22页：复杂度两套数字
- 矛盾：Table 2（0.24M/0.01G，FCNet 0.08M）vs Table 5（0.313M/0.316G，FCNet 0.199M）。
- 裁决：**PoC 图（图 6、图 14）一律使用 Table 5**（与 PoC 实验匹配）；Table 2 保留为
  生产规模估计并在报文标注。FLOPs 统一按"单样本一次前向（输入 4×64×128, batch=1）"统计；
  相对复杂度 = FLOPs/AdaptFNO 之比值（0.316/0.314 = 1.006×，故三模型 1.00×–1.01×）。
- 涉及图：fig06、fig14。

### 3.5 第17页 vs 第22页：Table 3 与 Table 6/图8/图9 结论相反
- 矛盾：Table 3 声称 1–10 天全面领先；Table 6（96 h：DIB-FNO 67.2/0.927 为四种方法中最差，
  且不如 persistence 52.5/0.958）与图 8/9 显示混合结果。两套结果除非来自**完全不同的、
  明确命名的实验**，否则不能同时存在。
- 裁决：**拆分为两个命名实验，每套完整写明规格**（分辨率、变量数、训练年份、测试年份、
  滚动方式、模型宽度、随机种子；见下表），并**只允许一套作为摘要/结论的权威主实验**：

| 项目 | **Experiment P（权威主实验）** | **Experiment C（参考实验，非主实验）** |
|---|---|---|
| 角色 | proof-of-concept，图 fig01–fig21 | full-scale reference，图 figC1–figC3 |
| 变量 | 4 个：Z500/T850/U850/V850 | 20 个（§4.4；评估用 Z500/T850/U850/R500） |
| 分辨率 | 64×128 = 2.8125° | ERA5 0.25°（≈720×1440） |
| 训练/测试年份 | 一个 30 天样本，80/20 时序划分（95/24） | 训练 1979–2017；验证 2018–2019；测试 2020–2022 |
| 滚动方式 | 6 小时间隔自回归至 96 h；台风 72 h | 6 小时间隔自回归，评估 1–10 天 |
| 模型宽度 | AFNO hidden 1024, patch 4 | 同左 |
| 随机种子 | 20240517（重建材料固定种子） | 20240517（重建材料固定种子） |
| 结果锚点 | Table 5 / Table 6 / Table 4 / §5.1 | Table 2 / Table 3 |

  - Fig 8/9/10 全部标注"Experiment P"（图脚含完整规格行）；
  - **Table 3 的数字单独生成 Experiment C 图**：`figC1_global_skill_reference.png`
    （Table 3 柱状图+误差棒）、`figC2_lead_time_reference.png`（1–10 天曲线，
    曲线均值严格等于 Table 3）、`figC3_complexity_reference.png`（Table 2）；
  - 每幅 C 图都带"NOT the main experiment"标注；两实验数字**不得混用或比较**；
  - 摘要/结论只能引用 Experiment P；若引用 Table 3 的数字（104.2 m²/s²、0.83），
    必须明确标注为"Experiment C 参考结果，未经复现"。
- 涉及图：fig08、fig09、fig10（P）+ figC1、figC2、figC3（C，新增）。

### 3.6 第17–20页：变量集与单位混乱（msl/u10/v10/t2m vs Z500/T850/U850/V850）
- 矛盾：Table 3 用 Z500/T850/U850/R500；图 8/10/11/12 用 msl/u10/v10/t2m（且为归一化
  量，RMSE≈0.1）；图 9 ACC≈0.001 而 Table 6 ACC≈0.93–0.99；图 8 文本称"msl RMSE 稳定
  低于 0.2"单位不明。
- 裁决：**全部统一为 PoC 的 4 个变量 Z500/T850/U850/V850，全部使用物理单位**
  （m²/s²、K、m/s），数值锚定 Table 6（Z500 严格一致）并在 README/PROTOCOL 中说明
  其余变量由 Table 3 的比例重建；**ACC≈0.001 的图删除**——ACC 是异常相关，0.001 与
  "显著技巧"不相容，真实值域为 0.85–0.99（图 8/9/10 用此值域并标注单位）。
  图 9 的 50 h 窗口延长至 96 h 以与 Table 6 的 24/96 h 锚点一致。
- 涉及图：fig08、fig09、fig10、fig11、fig12、fig21。

### 3.7 第15页：基线"训练失败"被当作优势
- 矛盾：正文称基线第 7 epoch 提前终止"原因可能是严重过拟合"，从而衬托 DIB-FNO。
- 裁决：图 5 如实绘制：基线在第 4–5 epoch 验证损失达最小后回升（过拟合特征），
  于 epoch 7 早停；**图中标注"验证损失回升 → 早停"**，并建议正文改为"在相同
  训练设置下基线早停后重训/调优，避免把训练失败当作优势"。
- 涉及图：fig05。

### 3.8 第20–24页：台风起报与统计口径
- 矛盾：正文说 rollout 72 h，图 15 却展示 230 h；只有 3 个台风却给出 ± 标准差；
  未说明初始化时次、中心定义、误差来源。
- 裁决：图 13/15/16 **全部统一为 0–72 h**，3 个台风 × 每 6 h 一个时次；
  图 15 标注"mean ± std over 3 storms"；72 h 端点严格等于 Table 4
  （5.4/7.2/6.8 m/s）；24/48 h 位置误差用于图 13 的路径偏差（62/158 km 等）；
  图 16 展示极端值区（v>55 m/s、p<950 hPa）。
  **起点与最佳路径**：Rai 2021-12-16 00Z、Chanthu 2021-09-10 00Z、Noru 2022-09-25 00Z
  （起报时刻 = 风暴达到近峰值强度，且其后 72 h 内有完整最佳路径），
  最佳路径取 JMA（IBTrACS v04r01 的 TOKYO/WMO 支），风速为 10 分钟最大风（kt→m/s ×0.5144）。
- 涉及图：fig13、fig15、fig16。

### 3.9 第24–25页：高频保留结论正反颠倒
- 矛盾：§5.5 第一段说台风保留更多高频模式、平静区近低通（与摘要一致）；
  但图 18 的正文段却说台风"严格截断高频"、平静区"噪声式保留高频"，两段互相矛盾。
- 裁决：**以定理 3.1 与摘要为准**（台风保留更多高频），图 18 重画为
  （a）台风掩码（保留率 0.42，截至更外圈）
  （b）平静高压（保留率 0.27，更窄）
  （c）差值图（±0.3，外环为正，即台风多保留的高频模态）。
  同时把"保留率/径向平均/高频占比"的计算口径写入 PROTOCOL（δ 即
  mean(a) over 64×65 实 FFT 面；置信区间因仅 3 台风×12 时次，用 ±1σ 给出）。
- 涉及图：fig17、fig18、fig20。

### 3.10 第27页：图 19/20 子图与标题不符、缺图 14
- 矛盾：图 19 标题的"2D 掩码与径向平均"实为图 20 的内容；图 14 整幅缺失
  （正文位置只有一行字 `results_model/fig_model_complexity.png`）。
- 裁决：图 19 =（a）RMSE vs 保留率散点 +（b）RMSE/ACC vs λ 双轴（与标题一致）；
  图 20 =（a）2D 掩码 +（b）径向平均（λ=1e-2 变体，径向平均峰值≤0.30，与正文一致）；
  **图 14 补齐**为"模型架构与复杂度对比"：(a) 三骨干架构示意图，
  (b) 参数 (c) FLOPs（Table 5 数值）。
- 涉及图：fig14（新增）、fig19、fig20。

## 4. 其他全局修正（不止 10 条）

| 位置 | 问题 | 修正 |
|---|---|---|
| §4.1 | 网格 64×128 对应 2.8125°，原文写 ≈2.75° | 全部改为 2.8°（图 1 标注 2.8°） |
| §4.2/图 2 | "相关 0.00–0.02 → 数据冗余小" | 已按 §3.3 修正 |
| 图 8 | 标题"50 h 窗口" | 窗口延长至 96 h（与 Table 6 锚点一致） |
| §5.2 | "图 11 展示 msl,u10,v10,t2m" | 统一为 Z500/T850/U850/V850 |
| §5.2 | 误差幅值"严格限制在 −0.4~0.4" | 保留该声明；图 11/12/21 共享 [−0.4,0.4] 色标 |
| §5.4 | "At this scale, no learned model beats persistence" | 与 Table 6 矛盾（24 h 时 DIB 28.9 < 31.7）；图 9 如实画出交叉 |
| §5.1 | 图 4(d)"0.5 → 0.1" | 统一为 0.5 → 0.34（= §5.6 的 λ=1e-3 最优保留率） |
| §5.6 | "λ=0 → RMSE 绝对最小 0.11, 掩码稀疏度大减" | 图 19(a) 画 λ=0 保留率 0.88（"稀疏度大减" ✓ 一致） |
| 图 4(b) | RMSE/ACC 量纲不明 | 明确标注"RMSE (normalized), +6 h"与 ACC |

## 5. 科学准确性与数据口径（审稿要求：重建内容不得冒充实验结果）

**分类原则**：本包中只有"论文自身报告过的数值"是正式数据；其余全部为
**示意性重建（illustrative reconstruction）**，不是模型运行输出。每个类别的
图均在其左下角带有**数据来源标签**（DATA: verbatim / PARTLY RECONSTRUCTED /
ILLUSTRATIVE RECONSTRUCTION）。

| 类别 | 定义 | 图 | 主文处理建议 |
|---|---|---|---|
| **A：论文已报告（verbatim）** | 数值 100% 来自论文表格/正文，仅重绘 | fig01（架构示意）、fig06（Table 5）、fig14（Table 5）、figC1（Table 3）、figC3（Table 2）、tables/01–06 | 可保留于主文；引用时注明"as reported in the paper" |
| **B：混合（mixed）** | 部分数值来自论文，其余为示意性重建 | fig02（0.93 为 Table 1，其余为典型 ERA5 估计值，图上已注明）、fig03（按文中公式计算的调度）、fig07（总量 68.3 s 为原文，逐 epoch 分解为示意）、fig13（JMA 真实最佳路径 + 示意预报线，表 4 误差锚点）、fig19（λ=0 与 λ=1e-3 的端点值见 §5.6，中间点为示意） | 主文仅引用"已报告数值"部分；示意部分移入附录或明确标注 |
| **C：重建（reconstructed）** | 论文未提供任何训练日志/原始曲线，全部为锚点拟合的示意重建 | fig04、fig05、fig08、fig09、fig10、fig11、fig12、fig15、fig16、fig17、fig18、fig20、fig21、figC2 | **不得作为实验结果引用**；建议移入附录标为 "illustrative reconstruction (anchored to Table X)"，并在正文删除相应过度结论（如"consistently outperforms"、"有效解决误差累积崩溃"等） |

**三类图的共同规则**
1. 每幅图左下角有数据来源标签 + 实验归属标签（Experiment P / Experiment C 及完整规格）。
2. 端点/锚点值（如 Table 6 的 28.9/0.986、Table 4 的 62 km、§5.6 的 34% 等）**均为论文原文数值**；
   曲线形状为"锚定这些数值的确定性重建"（公式见 PROTOCOL.md §3、§5、§7）。
3. **过平滑/误差界断言**（±0.4、±0.3、SSIM 0.72/0.58、λ=1e-3→34%、径向平均峰值 0.30、
   68.3 s、近 10⁻² by epoch 15）在 A 类或 B 类中，可保留；图 11/12/15/16/17/18/20/21 的
   具体空间/时间形态为示意。
4. **不得把重建图与正文结论绑定**：建议正文改为"图 X（示意重建，锚定表 Y 数值）"，
   或在正文中仅保留表格数值与 A 类图（fig06/14/C1/C3 及表 1–6）。

**关于可编辑交付**：每幅图同时提供 **300 dpi PNG** 与 **矢量 PDF（可编辑）**，
脚本与全部锚点数值（`data/tables/*.csv`、`data/tracks/*.csv`）一并交付，
便于按要求再修改或转存为论文可编辑格式。

**坐标与投影**：全球图为等经纬网格 + 天然地球 110m 海岸线；台风图为经纬度矩形。
6. **两个实验不得混用**：Table 3/Table 2 的数字仅在 Experiment C 图中出现（figC1–C3，
   含"NOT the main experiment"标注）；Table 5/Table 6/Table 4 的数字仅在 Experiment P
   图中出现。正文叙述应删去所有把 104.2 m²/s² 等 C 尺度数字与 P 尺度数字并列的说法，
   并在摘要/结论中只使用 Experiment P。

## 6. 交付清单

```
results_model/
├── README.md                        ← 本文件
├── PROTOCOL.md                      ← 技术协议（两个命名实验规格、公式、锚点、重建口径）
├── figures/                         ← 21+3 张论文图（300 dpi PNG + 矢量 PDF；
│   │                                 全部英文；每幅图左下角有【数据来源标签】+
│   │                                 【实验归属与完整规格】两行注记）
│   │                                 以下 21 幅 = Experiment P（权威主实验）
│   ├── fig01_architecture.png/.pdf        架构图（重绘，修正重叠/乱码）─ verbatim
│   ├── fig02_correlation_matrix.png/.pdf  相关矩阵（0.93 为 Table 1；其余为估计）─ mixed
│   ├── fig03_lr_schedule.png/.pdf         LR 余弦退火（按文中公式计算）─ mixed
│   ├── fig04_training_dynamics.png/.pdf   损失 / RMSE–ACC / τ / 保留率 ─ RECONSTRUCTION
│   ├── fig05_convergence.png/.pdf         收敛曲线（基线早停如实标注）─ RECONSTRUCTION
│   ├── fig06_model_complexity.png/.pdf    参数 / FLOPs / 相对复杂度（Table 5）─ verbatim
│   ├── fig07_training_time.png/.pdf       每 epoch 与累计训练时间（68.3 s）─ mixed
│   ├── fig08_rmse_acc_lead.png/.pdf       4 变量 × RMSE/ACC 随预报时效（6–96 h）─ RECONSTRUCTION
│   ├── fig09_autoregressive_stability.png/.pdf  24/96 h 锚点 + persistence（混合结果）─ RECONSTRUCTION
│   ├── fig10_per_variable_skill.png/.pdf  分变量 24 h/96 h 相对 RMSE 与 ACC ─ RECONSTRUCTION
│   ├── fig11_spatial_error.png/.pdf       24 h 误差空间分布（±0.4 色标）─ RECONSTRUCTION
│   ├── fig12_spatial_error_matrix.png/.pdf  4 变量 × 6h/24h/48h 误差矩阵 ─ RECONSTRUCTION
│   ├── fig13_typhoon_tracks.png/.pdf      三台风 72 h 路径（JMA 实路径 + 示意预报）─ mixed
│   ├── fig14_architecture_complexity.png/.pdf  ★缺失图已补齐（Table 5）─ verbatim
│   ├── fig15_intensity_errors.png/.pdf     最大风/最低气压误差（0–72 h，表4锚点）─ RECONSTRUCTION
│   ├── fig16_extreme_value.png/.pdf        极端值散点（v>55 m/s、p<950 hPa）─ RECONSTRUCTION
│   ├── fig17_spectral_mask.png/.pdf        掩码场 / 径向平均 / 直方图（保留率 0.34）─ RECONSTRUCTION
│   ├── fig18_mask_regimes.png/.pdf         台风 vs 平静区（0.42 vs 0.27）与差值 ─ RECONSTRUCTION
│   ├── fig19_sparsity_ablation.png/.pdf    λ 消融（端点值见 §5.6）─ mixed
│   ├── fig20_mask_ablation.png/.pdf        λ=1e-2 变体掩码（径向平均峰值 0.30）─ RECONSTRUCTION
│   ├── fig21_single_step_error.png/.pdf    单步预测 + 误差（过平滑叙事）─ RECONSTRUCTION
│   │                                 以下 3 幅 = Experiment C（参考实验，非主实验）
│   ├── figC1_global_skill_reference.png/.pdf  Table 3 柱状图 + 误差棒 ─ verbatim
│   ├── figC2_lead_time_reference.png/.pdf      1–10 天曲线（均值严格 = Table 3）─ RECONSTRUCTION
│   ├── figC3_complexity_reference.png/.pdf     Table 2 复杂度 ─ verbatim
│   └── tables/table01..06.png              6 张表格重排图（全部 verbatim）
├── data/
│   ├── tables/table01..06.csv             表格原始数据（数字与论文一致）
│   └── tracks/rai|chanthu|noru_besttrack.csv   JMA 最佳路径（6 小时间隔）
└── scripts/
    ├── common.py                          统一数据模型 / 两个实验规格 / 锚点 / 样式
    ├── make_figures_a.py                  图 1–7（Experiment P）
    ├── make_figures_b.py                  图 8–12、14、21（Experiment P）
    ├── make_figures_c.py                  图 13、15、16（Experiment P）
    ├── make_figures_d.py                  图 17–20（Experiment P）
    ├── make_experiment_c.py               figC1–figC3（Experiment C）
    └── make_tables.py                     表格 CSV + 表格图
```

**复现**：`cd results_model/scripts && python make_figures_a.py && python make_figures_b.py
&& python make_figures_c.py && python make_figures_d.py && python make_experiment_c.py
&& python make_tables.py`
（依赖 numpy/scipy/matplotlib/pandas/geopandas；所有随机过程固定种子，结果可复现。）
