---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 7642744257689698570-data_volume/files/所有对话/主对话/DIB-FNO_代码优化与论文完善计划.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 3999340532600835#1780759354983
    ReservedCode2: ""
---
# DIB-FNO 代码优化与论文证据完善计划

## 一、当前代码与论文的差距分析

### 1.1 论文声称 vs 代码实现的对应关系

| 论文章节 | 论文声称 | 代码现状 | 差距等级 |
|---------|---------|---------|---------|
| §4.1 架构 | 20个气象变量 | 仅4个变量(msl,u10,v10,t2m) | 🔴严重 |
| §4.3 训练细节 | 4×A100, batch=32, 80 epochs | batch=16, 50 epochs, 单GPU | 🟡中等 |
| §5.1 Table 1 | FourCastNet/AdaptFNO/DIB-FNO对比 | 无基线模型 | 🔴严重 |
| §5.1 Table 1 | Z500/T850/U850/R500 指标 | 仅总体RMSE/ACC | 🔴严重 |
| §5.1 | 纬度加权RMSE/ACC | 无纬度权重计算 | 🟡中等 |
| §5.1 | 1-10天多步预报 | 仅单步(6h)预测 | 🔴严重 |
| §5.2 Table 2 | 台风个例验证(Rai/Chanthu/Noru) | 无台风验证 | 🔴严重 |
| §5.2 | SSIM指标(涡度场10°×10°窗) | 无SSIM计算 | 🔴严重 |
| §5.3 | 台风vs安静期掩码对比可视化 | 仅单一掩码图 | 🟡中等 |
| §5.4 | 消融实验(λ变化→保留率→RMSE) | 无消融实验 | 🔴严重 |
| §4.2 Eq.11 | 温度退火(τ: 1.0→0.1) | 固定temperature=0.5 | 🟡中等 |
| §4.2 | λ线性warm-up(0→10⁻³, 前20 epoch) | warm-up 10 epoch | 🟢轻微 |

### 1.2 代码性能瓶颈分析

| 瓶颈 | 原因 | 影响程度 |
|------|------|---------|
| 数据加载慢 | 每次从Google Cloud远程读取Zarr，无本地缓存 | 🔴极慢 |
| 内存占用大 | 数据集`.load()`尝试全量加载到内存 | 🔴OOM风险 |
| AFNO块简化 | fft_conv只处理modes×modes，但clone整个fft | 🟡冗余计算 |
| 无梯度累加 | batch=16可能无法充分利用GPU | 🟡效率低 |
| 无梯度检查点 | 4个AFNO块全部存储中间激活 | 🟡显存浪费 |
| SpectralMaskGenerator | ConvTranspose上采样可能不精确 | 🟢轻微 |

---

## 二、代码功能优化计划

### 阶段1：数据与基础设施优化（优先级最高，解决运行速度）

#### 1.1 本地数据缓存
```python
# 当前：每次从GCS远程读取
# 优化：首次下载后缓存到本地
class ERA5LocalCache(Dataset):
    """本地缓存版数据集"""
    def __init__(self, cache_dir="./era5_cache", ...):
        self.cache_dir = cache_dir
        # 首次运行时下载并转为本地numpy/zarr
        # 后续直接从本地读取
```
- **操作**：首次运行时将ARCO-ERA5下载到本地SSD，后续训练直接读本地
- **预期加速**：数据加载速度提升10-50倍

#### 1.2 预处理数据为.pt文件
```python
# 将Zarr数据预转换为PyTorch张量文件
def preprocess_to_pt(zarr_path, output_dir, variables, years):
    """一次性预处理，后续直接load .pt文件"""
    ds = xr.open_zarr(zarr_path, ...)
    for t in tqdm(range(len(ds.time))):
        x = np.stack([ds[var].isel(time=t).values for var in variables])
        np.save(f"{output_dir}/sample_{t:06d}.npy", x)
```
- **操作**：离线预处理为.npy或.pt文件
- **预期加速**：DataLoader速度提升5-10倍

#### 1.3 梯度累加
```python
# 当前：batch_size=16
# 优化：accumulation_steps=2, 等效batch=32
accumulation_steps = 2
optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    pred, mask = model(x, return_mask=True)
    loss = criterion(pred, y, mask) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 1.4 梯度检查点（节省显存）
```python
from torch.utils.checkpoint import checkpoint

class DIBFNO(nn.Module):
    def forward(self, x, return_mask=False):
        mask = self.mask_gen(x)
        x_filtered = self.spectral_filter(x, mask)
        z = self.input_proj(x_filtered)
        for block in self.blocks:
            z = checkpoint(block, z)  # 用时间换显存
        pred = self.output_proj(z)
        ...
```
- **预期**：显存占用降低40-60%，可增大batch或模型

#### 1.5 AFNO块优化
```python
# 当前问题：clone整个fft只修改了modes×modes
# 优化：只计算需要的modes
class AFNOBlock(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        # 只取modes区域处理
        fft_crop = x_fft[:, :, :self.modes, :self.modes]
        conv_out = self.fft_conv(fft_crop.real) + 1j * self.fft_conv(fft_crop.imag)
        # 直接写回，不用clone
        x_fft[:, :, :self.modes, :self.modes] = conv_out
        x = torch.fft.irfft2(x_fft, s=(H, W), norm='ortho')
        ...
```

---

### 阶段2：核心实验功能补全（支撑论文Table 1-2和Section 5.1-5.4）

#### 2.1 多变量支持（论文要求20变量）
```python
# 扩展变量列表
variables_20 = [
    "z500", "z700", "z850", "z1000",  # 位势高度
    "t500", "t700", "t850",           # 温度
    "u850", "u1000",                   # 纬向风
    "v850", "v1000",                   # 经向风
    "r500", "r700", "r850",           # 相对湿度
    "q500", "q700", "q850",           # 比湿
    "msl",                             # 海平面气压
    "t2m",                             # 2米温度
    "u10", "v10",                      # 10米风
]
```
- **注意**：ARCO-ERA5单层只有地面变量，气压层变量需要从多层Zarr读取
- **如无法获取20变量**：至少扩展到8-10个关键变量

#### 2.2 纬度加权指标
```python
def compute_lat_weights(latitudes, H):
    """计算纬度权重（高纬度格点面积小，需加权）"""
    lat_rad = np.deg2rad(latitudes)
    weights = np.cos(lat_rad)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def compute_lat_weighted_rmse(pred, target, lat_weights):
    """纬度加权RMSE"""
    diff = (pred - target) ** 2
    weights = lat_weights.view(1, 1, -1, 1).to(pred.device)
    return torch.sqrt((diff * weights).mean()).item()

def compute_lat_weighted_acc(pred, target, clim_mean, lat_weights):
    """纬度加权ACC"""
    weights = lat_weights.view(1, 1, -1, 1).to(pred.device)
    pa = (pred - clim_mean) * weights
    ta = (target - clim_mean) * weights
    num = (pa * ta).sum()
    den = torch.sqrt((pa**2).sum() * (ta**2).sum())
    return (num / (den + 1e-8)).item()
```

#### 2.3 多步自回归预报（1-10天）
```python
def autoregressive_forecast(model, x_init, n_steps=40):
    """自回归多步预报，每步6h，共240h(10天)"""
    model.eval()
    predictions = []
    x = x_init.clone()
    with torch.no_grad():
        for step in range(n_steps):
            pred = model(x)
            predictions.append(pred)
            x = pred  # 下一步输入=上一步输出
    return predictions  # list of (B, C, H, W)

def evaluate_multi_step(model, dataloader, device, n_steps=40):
    """评估1-10天各步的RMSE和ACC"""
    rmse_by_step = {i: [] for i in [4, 8, 16, 24, 40]}  # 1d,2d,4d,6d,10d
    acc_by_step = {i: [] for i in [4, 8, 16, 24, 40]}
    for x, y_sequence in dataloader:
        preds = autoregressive_forecast(model, x.to(device), n_steps)
        for step_idx in rmse_by_step:
            pred = preds[step_idx]
            truth = y_sequence[step_idx].to(device)
            rmse = compute_lat_weighted_rmse(pred, truth, lat_weights)
            acc = compute_lat_weighted_acc(pred, truth, clim_mean, lat_weights)
            rmse_by_step[step_idx].append(rmse)
            acc_by_step[step_idx].append(acc)
    return rmse_by_step, acc_by_step
```

#### 2.4 基线模型实现（Table 1必需）
```python
class FourCastNetBaseline(nn.Module):
    """FourCastNet基线：固定低通截断64 modes"""
    def __init__(self, in_channels, img_shape, hidden_dim=256, n_blocks=4, modes=64):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            AFNOBlockFixed(hidden_dim, modes=modes) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, 1)

class AdaptFNOBaseline(nn.Module):
    """AdaptFNO基线：可学习mode权重，无稀疏性"""
    def __init__(self, in_channels, img_shape, hidden_dim=256, n_blocks=4):
        super().__init__()
        # 可学习的静态频率权重
        H, W = img_shape
        self.mode_weights = nn.Parameter(torch.ones(1, 1, H, W//2+1))
        ...
```

#### 2.5 每变量指标分解
```python
def compute_per_variable_metrics(pred, target, var_names, lat_weights):
    """计算每个变量的RMSE和ACC"""
    results = {}
    for i, var in enumerate(var_names):
        pred_var = pred[:, i:i+1]
        target_var = target[:, i:i+1]
        rmse = compute_lat_weighted_rmse(pred_var, target_var, lat_weights)
        acc = compute_lat_weighted_acc(pred_var, target_var, clim_means[var], lat_weights)
        results[var] = {'rmse': rmse, 'acc': acc}
    return results
```

#### 2.6 温度退火机制
```python
class TemperatureScheduler:
    """温度退火：τ从1.0指数衰减到0.1"""
    def __init__(self, init_temp=1.0, final_temp=0.1, total_epochs=50):
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs
        self.decay_rate = (final_temp / init_temp) ** (1.0 / total_epochs)
    
    def get_temperature(self, epoch):
        return self.init_temp * (self.decay_rate ** epoch)

# 在训练循环中更新
model.mask_gen.temperature = temp_scheduler.get_temperature(epoch)
```

---

### 阶段3：台风验证功能（支撑论文§5.2和Table 2）

#### 3.1 IBTrACS台风数据加载
```python
class IBTrACSDataset:
    """加载IBTrACS最佳路径数据"""
    def __init__(self, ibtracs_path, typhoons=["Rai", "Chanthu", "Noru"]):
        # 读取IBTrACS CSV/NetCDF
        # 提取指定台风的6小时间隔位置和强度
        self.storms = {}
        for name in typhoons:
            self.storms[name] = self._extract_storm(name)
    
    def _extract_storm(self, name):
        # 提取：时间、纬度、经度、最大风速、中心气压
        return {
            'time': [...], 'lat': [...], 'lon': [...],
            'max_wind': [...], 'min_pressure': [...]
        }
```

#### 3.2 台风路径预报评估
```python
def compute_track_error(pred_lat, pred_lon, true_lat, true_lon):
    """大圆距离计算台风路径误差(km)"""
    R = 6371  # 地球半径km
    lat1, lon1 = np.deg2rad(pred_lat), np.deg2rad(pred_lon)
    lat2, lon2 = np.deg2rad(true_lat), np.deg2rad(true_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def find_storm_center(field, lat_grid, lon_grid):
    """从预报场中定位台风中心（最小海平面气压或最大涡度）"""
    # 方法1：海平面气压最小值定位
    min_idx = field.argmin()
    lat_idx, lon_idx = np.unravel_index(min_idx, field.shape)
    return lat_grid[lat_idx], lon_grid[lon_idx]

def compute_ssim(field1, field2, window_size=11):
    """结构相似性指数（10°×10°窗口内）"""
    from skimage.metrics import structural_similarity
    return structural_similarity(field1, field2, win_size=window_size, data_range=field1.max()-field1.min())
```

---

### 阶段4：消融实验（支撑论文§5.4）

#### 4.1 稀疏性消融
```python
def run_ablation_study(base_config, lambda_values=[0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]):
    """消融实验：不同λ下的RMSE和掩码保留率"""
    results = []
    for lam in lambda_values:
        model = DIBFNO(**base_config)
        criterion = DIBLoss(sparsity_weight=lam, warmup_epochs=20)
        # 训练并记录
        metrics = train_and_evaluate(model, criterion, ...)
        avg_retention = get_mask_retention(model, val_loader)
        results.append({
            'lambda': lam,
            'rmse': metrics['rmse'],
            'acc': metrics['acc'],
            'retention_ratio': avg_retention,
        })
    return results
```

---

## 三、论文支撑图表生成计划

### 3.1 必需图表清单

| 图表编号 | 对应论文位置 | 图表内容 | 优先级 | 数据来源 |
|---------|-------------|---------|--------|---------|
| Fig.1 | §4.1 | DIB-FNO整体架构图 | P0 | 代码架构 |
| Fig.2 | §5.1 | 各变量RMSE随预报时效变化曲线 | P0 | 多步预报结果 |
| Fig.3 | §5.1 | 各变量ACC随预报时效变化曲线 | P0 | 多步预报结果 |
| Fig.4 | §5.1 | Table 1柱状对比图(三模型×四指标) | P0 | Table 1数据 |
| Fig.5 | §5.2 | 台风路径预报对比图(实际vs预报) | P0 | 台风验证结果 |
| Fig.6 | §5.2 | 台风强度预报误差随时间变化 | P1 | 台风验证结果 |
| Fig.7 | §5.3 | 台风期掩码可视化 | P0 | 模型mask输出 |
| Fig.8 | §5.3 | 安静期掩码可视化 | P0 | 模型mask输出 |
| Fig.9 | §5.3 | 掩码差值图(台风-安静) | P1 | mask差值 |
| Fig.10 | §5.4 | 消融曲线(RMSE vs 保留率) | P0 | 消融实验 |
| Fig.11 | §5.4 | 消融曲线(RMSE vs λ) | P1 | 消融实验 |
| Fig.12 | 训练过程 | 训练/验证Loss曲线 | P1 | 训练日志 |
| Fig.13 | 训练过程 | 温度退火曲线(τ vs epoch) | P2 | 训练日志 |
| Fig.14 | 训练过程 | 掩码保留率随训练变化 | P1 | 训练日志 |
| Fig.15 | §5.1 | 预报场可视化(地面场6h/24h/72h/240h) | P1 | 预报结果 |

### 3.2 各图表具体生成方案

#### Fig.2-3: RMSE/ACC随时效变化
```python
def plot_rmse_acc_vs_lead_time(rmse_data, acc_data, var_names):
    """生成RMSE和ACC随预报时效变化的折线图"""
    lead_hours = [6, 12, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, var in enumerate(['Z500', 'T850', 'U850', 'R500']):
        ax = axes[i//2, i%2]
        for model_name in ['FourCastNet', 'AdaptFNO', 'DIB-FNO']:
            ax.plot(lead_hours, rmse_data[model_name][var], label=model_name, marker='o')
        ax.set_xlabel('Lead time (hours)')
        ax.set_ylabel(f'{var} RMSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_rmse_vs_lead_time.png', dpi=300)
```

#### Fig.5: 台风路径图
```python
def plot_typhoon_tracks(true_tracks, pred_tracks, typhoon_names):
    """台风路径对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, name in enumerate(typhoon_names):
        ax = axes[i]
        # 绘制实际路径
        ax.plot(true_tracks[name].lon, true_tracks[name].lat, 'k-o', label='IBTrACS')
        # 绘制各模型预报路径
        for model_name in ['FourCastNet', 'AdaptFNO', 'DIB-FNO']:
            ax.plot(pred_tracks[model_name][name].lon, 
                   pred_tracks[model_name][name].lat, '--', label=model_name)
        ax.set_title(f'Typhoon {name}')
        ax.legend()
    plt.savefig('fig_typhoon_tracks.png', dpi=300)
```

#### Fig.7-9: 掩码可解释性可视化
```python
def plot_mask_comparison(typhoon_mask, quiescent_mask, save_path):
    """台风期 vs 安静期掩码对比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # 台风期掩码
    im0 = axes[0].imshow(typhoon_mask, cmap='viridis', origin='lower')
    axes[0].set_title('(a) Typhoon regime')
    plt.colorbar(im0, ax=axes[0])
    # 安静期掩码
    im1 = axes[1].imshow(quiescent_mask, cmap='viridis', origin='lower')
    axes[1].set_title('(b) Quiescent regime')
    plt.colorbar(im1, ax=axes[1])
    # 差值
    diff = typhoon_mask - quiescent_mask
    im2 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower', vmin=-0.3, vmax=0.3)
    axes[2].set_title('(c) Difference (Typhoon - Quiescent)')
    plt.colorbar(im2, ax=axes[2])
    plt.savefig(save_path, dpi=300)
```

#### Fig.10: 消融曲线
```python
def plot_ablation(ablation_results):
    """RMSE vs 掩码保留率散点图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    retentions = [r['retention_ratio'] for r in ablation_results]
    rmses = [r['rmse'] for r in ablation_results]
    lambdas = [r['lambda'] for r in ablation_results]
    scatter = ax.scatter(retentions, rmses, c=lambdas, cmap='viridis', s=100)
    ax.set_xlabel('Average mask retention ratio')
    ax.set_ylabel('RMSE')
    ax.set_title('Ablation: RMSE vs Sparsity')
    plt.colorbar(scatter, label='λ (sparsity weight)')
    # 标注最优点
    best_idx = np.argmin(rmses)
    ax.annotate(f'λ={lambdas[best_idx]:.0e}\n({retentions[best_idx]:.1%})',
                xy=(retentions[best_idx], rmses[best_idx]),
                xytext=(retentions[best_idx]+0.1, rmses[best_idx]+0.05),
                arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('fig_ablation.png', dpi=300)
```

---

## 四、运行速度优化方案

### 4.1 优化优先级排序

| 优化项 | 实施难度 | 预期加速 | 说明 |
|--------|---------|---------|------|
| 本地数据缓存 | 低 | 10-50x | 下载一次，后续本地读取 |
| 预处理为.npy文件 | 低 | 5-10x | 避免Zarr实时解码 |
| 降低分辨率训练 | 低 | 4-16x | 从0.25°降到1°或2°进行验证 |
| 梯度累加 | 低 | 等效2x batch | 无额外时间开销 |
| gradient checkpointing | 低 | 显存-40% | 可换更大batch |
| torch.compile | 已有 | 1.2-2x | 代码中已有 |
| AFNO块FFT优化 | 中 | 1.5-2x | 减少冗余clone和permute |
| 混合精度FP16 | 已有 | 1.5-2x | 代码中已有 |
| 多GPU DDP | 中 | 线性缩放 | 需要多卡环境 |

### 4.2 快速验证方案（降级配置）

如果GPU资源有限，建议先使用降级配置快速验证论文逻辑正确性：

```python
# ====== 快速验证配置（约1-2小时完成） ======
QUICK_CONFIG = {
    'resolution': '5.0°',       # 从0.25°降到5.0°，分辨率降低20倍
    'img_shape': (36, 72),       # 从720×1440降到36×72
    'variables': 4,              # 先用4个变量验证
    'hidden_dim': 64,            # 从256降到64
    'n_blocks': 2,               # 从4降到2
    'batch_size': 32,
    'epochs': 20,                # 从80降到20
    'n_steps_eval': [1, 4, 8],   # 评估6h, 24h, 48h
}

# ====== 论文级配置（需4×A100，约2天） ======
PAPER_CONFIG = {
    'resolution': '0.25°',
    'img_shape': (720, 1440),
    'variables': 20,
    'hidden_dim': 1024,
    'n_blocks': 4,
    'batch_size': 32,
    'epochs': 80,
    'n_steps_eval': [4, 8, 16, 24, 40],  # 1d-10d
}
```

### 4.3 推荐执行流程

```
Step 1: 降级配置快速验证（1-2小时）
  → 确认模型逻辑正确、损失下降正常、掩码可生成

Step 2: 添加基线模型+评估代码（2-3小时编码）
  → FourCastNetBaseline + AdaptFNOBaseline
  → 多步预报 + 每变量指标 + 纬度加权

Step 3: 降级配置完整实验（4-6小时）
  → 三模型对比 → Table 1降级版
  → 消融实验 → 消融曲线
  → 掩码可视化 → 台风vs安静期对比

Step 4: 生成所有图表（1-2小时）
  → 15张论文支撑图

Step 5: 如有GPU资源 → 论文级配置完整训练（2天）
  → 替换降级结果为完整结果
```

---

## 五、实施检查清单

### 必须完成（P0）
- [ ] 本地数据缓存或预处理为.npy
- [ ] 基线模型(FourCastNet/AdaptFNO)实现
- [ ] 纬度加权RMSE/ACC计算
- [ ] 多步自回归预报(至少1-4天)
- [ ] 每变量指标分解(Z500/T850/U850/R500)
- [ ] 消融实验(λ变化)
- [ ] 台风vs安静期掩码对比可视化
- [ ] RMSE/ACC随时效变化曲线图
- [ ] Table 1/2柱状对比图
- [ ] 温度退火机制(τ: 1.0→0.1)

### 建议完成（P1）
- [ ] 台风路径预报评估
- [ ] SSIM计算
- [ ] 训练Loss曲线图
- [ ] 掩码保留率随训练变化图
- [ ] 预报场可视化(6h/24h/72h/240h)
- [ ] 梯度累加
- [ ] gradient checkpointing
- [ ] AFNO块FFT冗余优化

### 可选完成（P2）
- [ ] 20变量完整支持
- [ ] 多GPU DDP训练
- [ ] IBTrACS完整台风验证
- [ ] 掩码径向频谱分析图
- [ ] 温度退火曲线图

---

## 六、预期时间表

| 阶段 | 内容 | 预计时间 | 前置依赖 |
|------|------|---------|---------|
| 阶段1 | 数据+性能优化 | 2-3小时 | 无 |
| 阶段2 | 核心实验功能 | 3-4小时 | 阶段1 |
| 阶段3 | 台风验证 | 2-3小时 | 阶段2 |
| 阶段4 | 消融实验 | 1-2小时 | 阶段2 |
| 图表生成 | 15张图 | 2-3小时 | 阶段2-4 |
| **总计** | | **10-15小时编码** | |

降级配置下完整实验+图表约需 **1天GPU时间**；
论文级配置需 **2-3天GPU时间（4×A100）**。

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
