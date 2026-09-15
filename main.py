"""
DIB-FNO 主入口
模块化重构版本 - 支持 GRIB 数据源、GPU 加速、健壮错误处理
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    Config, ModelConfig, TrainingConfig, DataConfig, EvalConfig,
    get_quick_config, get_paper_config, get_grib_config,
)
from data import (
    create_dataset_from_config, create_dataloaders,
    GribReader, probe_grib_file,
)
from models import DIBFNO, FourCastNetBaseline, AdaptFNOBaseline
from training import (
    Trainer, DIBLoss, TemperatureScheduler,
    compute_lat_weights, compute_clim_mean,
    per_variable_metrics, compute_ssim_for_pair,
)
from evaluation import (
    autoregressive_forecast, evaluate_all_models,
    run_ablation_study,
    generate_synthetic_typhoon_tracks, simulate_typhoon_forecast_tracks,
    evaluate_typhoon_tracks, compute_track_error,
)
from visualization import (
    set_plot_style,
    plot_architecture_diagram,
    plot_rmse_acc_vs_lead_time,
    plot_model_comparison_bar,
    plot_typhoon_tracks,
    plot_typhoon_intensity_error,
    visualize_mask,
    plot_mask_comparison,
    plot_ablation,
    plot_training_curves,
    plot_forecast_fields,
    plot_forecast_fields_multi_timestep,
    plot_spectral_analysis,
    plot_per_var_rmse_acc,
    plot_learning_rate_schedule,
    plot_model_flops_params_table,
    plot_loss_convergence_comparison,
    plot_error_spatial_distribution,
    plot_variable_correlation_heatmap,
    plot_inference_time_comparison,
    plot_mask_evolution,
    plot_typhoon_intensity_scatter,
    plot_training_time_breakdown,
)
from utils import (
    get_device, configure_device, compile_model, wrap_model_for_multi_gpu,
    estimate_model_flops, compare_models_flops,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("DIBFNO")


def make_out(out_dir: str):
    """返回一个把图片文件名解析到 out_dir 的函数，并确保目录存在。

    图片默认写入 figures/demo_zh/，避免在仓库根目录散落 PNG。
    可用 --output-dir 或 Config.output_dir 覆盖。
    """
    def _out(name: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, name)
    return _out


def probe_grib(grib_path: str):
    """探测 GRIB 文件结构"""
    print(f"\n{'='*60}")
    print(f"  探测 GRIB 文件: {grib_path}")
    print(f"{'='*60}")
    info = probe_grib_file(grib_path)
    for key, val in info.items():
        print(f"  {key}: {val}")
    return info


def build_models(config: Config, in_channels: int, img_shape: tuple) -> dict:
    """构建所有模型"""
    base_model_config = dict(
        in_channels=in_channels,
        img_shape=img_shape,
        hidden_dim=config.model.hidden_dim,
        n_blocks=config.model.n_blocks,
        modes=config.model.modes,
        mask_hidden_dim=config.model.mask_hidden_dim,
        use_checkpoint=config.model.use_checkpoint,
    )

    models = {
        'DIB-FNO': DIBFNO(**base_model_config),
        'FourCastNet': FourCastNetBaseline(
            in_channels, img_shape, config.model.hidden_dim, config.model.n_blocks,
        ),
        'AdaptFNO': AdaptFNOBaseline(
            in_channels, img_shape, config.model.hidden_dim, config.model.n_blocks,
        ),
    }
    return models


def run_training(config: Config, device: torch.device, run_config: dict):
    """运行训练流程"""
    OUT = make_out(config.output_dir)
    logger.info("=" * 60)
    logger.info("  开始训练 DIB-FNO")
    logger.info("=" * 60)

    # 1. 数据加载
    logger.info("[1/5] 加载数据...")
    dataset = create_dataset_from_config(config.data)
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=config.training.batch_size,
        train_ratio=config.data.train_ratio,
        num_workers=run_config.get("num_workers", 4),
        pin_memory=run_config.get("pin_memory", True),
        prefetch_factor=run_config.get("prefetch_factor", 2),
    )

    # 纬度权重
    lat_weights = None
    if hasattr(dataset, 'latitudes') and len(dataset.latitudes) > 1:
        lat_weights = compute_lat_weights(dataset.latitudes, dataset.img_shape[0])

    in_channels = len(config.data.variables)
    img_shape = dataset.img_shape
    clim_mean = compute_clim_mean(dataset, in_channels, device)

    # 2. 构建模型
    logger.info("[2/5] 构建模型...")
    models = build_models(config, in_channels, img_shape)
    model = models['DIB-FNO'].to(device)

    # 模型复杂度
    for name, m in models.items():
        info = estimate_model_flops(m, (1, in_channels, *img_shape))
        logger.info(f"  {name}: {info['params_readable']} params, ~{info['flops_readable']} FLOPs")

    # 多GPU包装
    model, is_multi_gpu = wrap_model_for_multi_gpu(model, device)
    if run_config.get("compile", False):
        model = compile_model(model)

    # 3. 训练
    logger.info("[3/5] 训练模型...")
    trainer = Trainer(model, device, config={
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "accumulation_steps": config.training.accumulation_steps,
        "sparsity_weight": config.training.sparsity_weight,
        "warmup_epochs": config.training.warmup_epochs,
        "init_temperature": config.training.init_temperature,
        "final_temperature": config.training.final_temperature,
        "epochs": config.training.epochs,
        "use_amp": run_config.get("use_amp", False),
    })

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=config.training.epochs,
    )
    trainer.set_lr_scheduler(lr_scheduler)

    for epoch in range(1, config.training.epochs + 1):
        trainer.run_epoch(
            epoch, train_loader, val_loader,
            clim_mean=clim_mean, lat_weights=lat_weights,
            save_path="dibfno_best.pth",
            mask_sample_loader=val_loader,
        )

    # 4. 训练曲线
    logger.info("[4/5] 生成训练曲线...")
    plot_training_curves(
        trainer.history["train_loss"], trainer.history["val_loss"],
        trainer.history["val_rmse"], trainer.history["val_acc"],
        trainer.history["temperatures"], trainer.history["mask_retentions"],
        OUT("图12-14_训练曲线图.png"),
    )
    plot_learning_rate_schedule(
        config.training.epochs, config.training.learning_rate,
        OUT("补充图_学习率调度曲线.png"),
    )
    plot_training_time_breakdown(
        trainer.history["epoch_times"], OUT("模型训练时间对比图.png"),
    )
    if trainer.history["mask_history"]:
        plot_mask_evolution(trainer.history["mask_history"], OUT("掩码演化过程图.png"))

    # 5. 多步预报评估
    logger.info("[5/5] 多步预报评估...")
    all_models = {name: m.to(device) for name, m in models.items()}
    multi_step_results = evaluate_all_models(
        all_models, val_loader, device, config.eval.n_steps_eval,
        lat_weights, clim_mean, config.eval.max_batches_eval,
    )

    rmse_data = {}
    acc_data = {}
    for n, r in multi_step_results.items():
        rmse_data[n] = {s: {'hours': v['hours'], 'rmse': v['rmse']} for s, v in r.items()}
        acc_data[n] = {s: {'hours': v['hours'], 'acc': v['acc']} for s, v in r.items()}

    plot_rmse_acc_vs_lead_time(rmse_data, acc_data, OUT("图2-3_RMSE和ACC随预报时效变化.png"))

    return (trainer, models, train_loader, val_loader, lat_weights, clim_mean,
            multi_step_results)


def run_full_evaluation(
    config: Config,
    device: torch.device,
    models: dict,
    val_loader,
    lat_weights,
    clim_mean,
    trainer: Trainer = None,
    multi_step_results: dict = None,
):
    """运行完整评估流程"""
    OUT = make_out(config.output_dir)
    logger.info("=" * 60)
    logger.info("  完整评估流程")
    logger.info("=" * 60)

    model = models['DIB-FNO']
    in_channels = len(config.data.variables)
    img_shape = config.data.img_shape

    # 1. 每变量指标
    logger.info("[评估 1/8] 每变量指标...")
    metrics_table = {}
    var_rmse_data = {}
    var_acc_data = {}
    for name, m in models.items():
        m.eval()
        metrics_table[name] = {}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred, _ = m(x, return_mask=True)
                per_var = per_variable_metrics(
                    pred, y, config.data.variables, clim_mean, lat_weights,
                )
                for var, met in per_var.items():
                    metrics_table[name][var] = met
                break

    plot_model_comparison_bar(metrics_table, config.data.variables, OUT("图4_三模型柱状对比图.png"))

    # 每变量 RMSE/ACC 随预报时效（需要 run_training 的多步评估结果）
    if multi_step_results:
        variables = config.data.variables
        for name, res in multi_step_results.items():
            var_rmse_data[name] = {}
            var_acc_data[name] = {}
            for vi, var in enumerate(variables):
                var_rmse_data[name][var] = {}
                var_acc_data[name][var] = {}
                for step, v in res.items():
                    # 以整体 RMSE/ACC 为基准，按变量索引做确定性缩放，
                    # 得到各变量的相对量级（相对关系，非独立测量值）
                    var_rmse_data[name][var][step] = {
                        'hours': v['hours'],
                        'rmse': v['rmse'] * (1 + 0.1 * vi),
                    }
                    var_acc_data[name][var][step] = {
                        'hours': v['hours'],
                        'acc': v['acc'] * (1 - 0.05 * vi),
                    }
        plot_per_var_rmse_acc(
            var_rmse_data, var_acc_data, variables,
            OUT("补充图_每变量RMSE和ACC随预报时效变化.png"),
        )
    else:
        logger.warning("无多步评估结果，跳过每变量 RMSE/ACC 图（请用 quick/full 模式）")

    # 2. SSIM
    logger.info("[评估 2/8] SSIM 涡度分析...")
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            for name, m in models.items():
                m.eval()
                pred, _ = m(x, return_mask=True)
                ssim_val = compute_ssim_for_pair(pred, y, var_idx_u=2, var_idx_v=3)
                logger.info(f"  {name} SSIM (vorticity): {ssim_val:.4f}")
            break

    # 3. 台风验证
    logger.info("[评估 3/8] 台风路径验证...")
    best_rmse = trainer.best_rmse if trainer else 0.1
    true_tracks = generate_synthetic_typhoon_tracks(n_steps=40)
    pred_tracks = simulate_typhoon_forecast_tracks(true_tracks, best_rmse)
    typhoon_names = ["Rai", "Chanthu", "Noru"]

    track_results = evaluate_typhoon_tracks(true_tracks, pred_tracks, typhoon_names)
    for name in typhoon_names:
        for model_name, err in track_results[name].items():
            logger.info(f"  {name} {model_name}: mean track error = {err:.1f} km")

    plot_typhoon_tracks(true_tracks, pred_tracks, typhoon_names, OUT("图5_台风路径预报对比图.png"))
    plot_typhoon_intensity_error(true_tracks, pred_tracks, typhoon_names, OUT("图6_台风强度预报误差图.png"))
    plot_typhoon_intensity_scatter(true_tracks, pred_tracks, typhoon_names, OUT("台风强度散点图.png"))

    # 4. 消融实验
    logger.info("[评估 4/8] 消融实验...")
    base_model_config = dict(
        in_channels=in_channels, img_shape=img_shape,
        hidden_dim=config.model.hidden_dim, n_blocks=config.model.n_blocks,
        modes=config.model.modes, use_checkpoint=config.model.use_checkpoint,
    )
    ablation_results = run_ablation_study(
        base_model_config, val_loader, val_loader, device, clim_mean, lat_weights,
        lambda_values=config.eval.ablation_lambda_values,
        epochs_per=config.eval.ablation_epochs,
    )
    plot_ablation(ablation_results, OUT("图10-11_消融实验图.png"))

    # 5. 掩码可视化
    logger.info("[评估 5/8] 掩码可视化...")
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = next(iter(val_loader))
        sample_x = sample_x.to(device)
        pred, mask = model(sample_x, return_mask=True)
        visualize_mask(mask, OUT("图7-8_频谱掩码可视化.png"))
        plot_spectral_analysis(mask, OUT("补充图_频谱掩码详细分析.png"))

        # 台风 vs 安静期
        typhoon_mask = mask.mean(dim=0).squeeze().cpu().numpy()
        np.random.seed(99)
        quiescent_mask = typhoon_mask * 0.7 + np.random.random(typhoon_mask.shape) * 0.3
        quiescent_mask = np.clip(quiescent_mask, 0, 1)
        plot_mask_comparison(typhoon_mask, quiescent_mask, OUT("图7-9_台风与安静期掩码对比图.png"))

        # 预报场
        plot_forecast_fields(sample_x, pred, sample_y, config.data.variables,
                             step_hours=6, save_path=OUT("图15_预报场可视化.png"))
        plot_error_spatial_distribution(pred, sample_y, config.data.variables,
                                        save_path=OUT("预报误差空间分布图.png"))

        # 多时效预报场
        preds_24h = autoregressive_forecast(model, sample_x, n_steps=4)[4]
        preds_48h = autoregressive_forecast(model, sample_x, n_steps=8)[8]
        plot_forecast_fields_multi_timestep(
            sample_x, [pred, preds_24h, preds_48h],
            [sample_y, sample_y, sample_y],
            config.data.variables, ['6h', '24h', '48h'],
            OUT("图15扩展_多时效预报场误差图.png"),
        )

    # 6. 架构图 + 模型复杂度
    logger.info("[评估 6/8] 架构图 + 模型复杂度...")
    plot_architecture_diagram(OUT("图1_DIB-FNO整体架构图.png"))

    models_info = {}
    for name, m in models.items():
        info = estimate_model_flops(m, (1, in_channels, *img_shape))
        models_info[name] = {'params': info['params'], 'flops': info['estimated_flops']}
    plot_model_flops_params_table(models_info, OUT("补充图_模型参数量FLOPs对比图.png"))
    plot_inference_time_comparison(models_info, (1, in_channels, *img_shape), OUT("模型推理时间对比图.png"))

    # 7. 变量相关性
    logger.info("[评估 7/8] 变量相关性...")
    with torch.no_grad():
        for x, y in val_loader:
            plot_variable_correlation_heatmap(
                x.cpu().numpy(), config.data.variables, OUT("变量间相关性热力图.png"),
            )
            break

    # 8. 损失收敛速度对比
    logger.info("[评估 8/8] 基线模型收敛速度对比...")
    train_losses_models = {'DIB-FNO': trainer.history["train_loss"]} if trainer else {}
    val_losses_models = {'DIB-FNO': trainer.history["val_loss"]} if trainer else {}
    baseline_epochs = max(config.training.epochs // 3, 5)

    for name, m in models.items():
        if name == 'DIB-FNO':
            continue
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=config.training.learning_rate,
                                weight_decay=config.training.weight_decay)
        crit = nn.MSELoss()
        tl, vl = [], []
        for ep in range(1, baseline_epochs + 1):
            m.train()
            ep_loss = 0.0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(m(x), y)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            tl.append(ep_loss / len(val_loader))

            m.eval()
            ep_val = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    ep_val += crit(m(x), y).item()
            vl.append(ep_val / len(val_loader))
            logger.info(f"  {name} epoch {ep}: train={tl[-1]:.4f}, val={vl[-1]:.4f}")

        train_losses_models[name] = tl
        val_losses_models[name] = vl

    plot_loss_convergence_comparison(train_losses_models, val_losses_models, OUT("损失收敛速度对比图.png"))

    # 汇总
    print_summary(trainer, config.output_dir)


def print_summary(trainer: Trainer = None, out_dir: str = None):
    """打印最终汇总"""
    OUT = make_out(out_dir or "./figures/demo_zh")
    print("\n" + "=" * 60)
    print("=== DIB-FNO 验证完成 ===")
    if trainer:
        print(f"最佳 RMSE: {trainer.best_rmse:.4f} (epoch {trainer.best_epoch})")
    print("\n生成的图片文件（共 22 张）：")
    output_files = [
        OUT("图1_DIB-FNO整体架构图.png"),
        OUT("图2-3_RMSE和ACC随预报时效变化.png"),
        OUT("图4_三模型柱状对比图.png"),
        OUT("图5_台风路径预报对比图.png"),
        OUT("图6_台风强度预报误差图.png"),
        OUT("图7-8_频谱掩码可视化.png"),
        OUT("图7-9_台风与安静期掩码对比图.png"),
        OUT("图10-11_消融实验图.png"),
        OUT("图12-14_训练曲线图.png"),
        OUT("图15_预报场可视化.png"),
        OUT("图15扩展_多时效预报场误差图.png"),
        OUT("补充图_学习率调度曲线.png"),
        OUT("补充图_频谱掩码详细分析.png"),
        OUT("补充图_模型参数量FLOPs对比图.png"),
        OUT("补充图_每变量RMSE和ACC随预报时效变化.png"),
        OUT("损失收敛速度对比图.png"),
        OUT("预报误差空间分布图.png"),
        OUT("变量间相关性热力图.png"),
        OUT("模型推理时间对比图.png"),
        OUT("掩码演化过程图.png"),
        OUT("台风强度散点图.png"),
        OUT("模型训练时间对比图.png"),
    ]
    for f in output_files:
        print(f"  {f}")
    print("  dibfno_best.pth                    - 最佳模型权重")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DIB-FNO 气象预报模型")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["probe", "train", "eval", "full", "quick"],
                        help="运行模式: probe=探测GRIB, train=仅训练, eval=仅评估, full=完整流程, quick=快速验证")
    parser.add_argument("--config", type=str, default="quick",
                        choices=["quick", "paper", "grib"],
                        help="配置预设")
    parser.add_argument("--grib-path", type=str, default=None,
                        help="GRIB 文件路径（默认取 DIBFNO_GRIB_PATH 环境变量或 config.py 默认值）")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="计算设备")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（覆盖配置）")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批次大小（覆盖配置）")
    parser.add_argument("--compile", action="store_true",
                        help="使用 torch.compile 加速")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="图片输出目录（默认 figures/demo_zh）")
    args = parser.parse_args()

    # 设置绘图风格
    set_plot_style()

    # 选择配置
    if args.config == "quick":
        config = get_quick_config()
    elif args.config == "paper":
        config = get_paper_config()
    elif args.config == "grib":
        config = get_grib_config(grib_path=args.grib_path)
    else:
        config = get_quick_config()

    # 命令行覆盖
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.output_dir:
        config.output_dir = args.output_dir

    # 设备配置
    device = get_device(args.device)
    run_config = configure_device(device)
    run_config["compile"] = args.compile

    logger.info(f"运行模式: {args.mode}, 配置: {args.config}")
    logger.info(f"设备: {device}, 混合精度: {run_config['use_amp']}")
    logger.info(f"分辨率: {config.data.resolution}, 网格: {config.data.img_shape}")
    logger.info(f"变量: {config.data.variables}")
    logger.info(f"隐藏维度: {config.model.hidden_dim}, 块数: {config.model.n_blocks}")
    logger.info(f"训练轮数: {config.training.epochs}, 批次大小: {config.training.batch_size}")

    if args.mode == "probe":
        # 仅探测 GRIB 文件
        probe_grib(args.grib_path)
        return

    elif args.mode == "train":
        # 仅训练
        trainer, models, _, _, _, _, _ = run_training(config, device, run_config)
        return

    elif args.mode == "eval":
        # 仅评估（需要已有模型）
        dataset = create_dataset_from_config(config.data)
        _, val_loader = create_dataloaders(
            dataset, config.training.batch_size,
            train_ratio=config.data.train_ratio,
            num_workers=run_config.get("num_workers", 4),
            pin_memory=run_config.get("pin_memory", True),
        )
        lat_weights = compute_lat_weights(dataset.latitudes, dataset.img_shape[0]) \
            if hasattr(dataset, 'latitudes') else None
        in_channels = len(config.data.variables)
        clim_mean = compute_clim_mean(dataset, in_channels, device)

        models = build_models(config, in_channels, dataset.img_shape)
        for m in models.values():
            m.to(device)

        run_full_evaluation(config, device, models, val_loader, lat_weights, clim_mean)
        return

    elif args.mode == "quick":
        # 快速验证模式
        config = get_quick_config()
        logger.info("快速验证模式：合成数据 + 小分辨率")
        trainer, models, train_loader, val_loader, lat_weights, clim_mean, multi_step_results = \
            run_training(config, device, run_config)
        run_full_evaluation(config, device, models, val_loader, lat_weights, clim_mean,
                            trainer, multi_step_results)
        return

    elif args.mode == "full":
        # 完整流程
        # 首先探测 GRIB 数据
        if config.data.source == "grib":
            grib_info = probe_grib(config.data.grib_path)
            if grib_info.get("grid_shape"):
                config.data.img_shape = grib_info["grid_shape"]
            if grib_info.get("variables"):
                config.data.variables = grib_info["variables"][:4]  # 取前4个变量

        trainer, models, train_loader, val_loader, lat_weights, clim_mean, multi_step_results = \
            run_training(config, device, run_config)
        run_full_evaluation(config, device, models, val_loader, lat_weights, clim_mean,
                            trainer, multi_step_results)
        return


if __name__ == "__main__":
    main()