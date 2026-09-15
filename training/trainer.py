"""
训练器模块
包含: 训练循环、混合精度训练、梯度累加、学习率调度
"""

import time
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Dict, List

from .loss import DIBLoss, TemperatureScheduler
from .metrics import validate_epoch

logger = logging.getLogger(__name__)


class Trainer:
    """DIB-FNO 训练器"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict = None,
    ):
        self.model = model
        self.device = device
        self.config = config or {}

        # 训练参数
        self.learning_rate = self.config.get("learning_rate", 5e-4)
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.accumulation_steps = self.config.get("accumulation_steps", 1)
        self.use_amp = self.config.get("use_amp", False)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)

        # 损失函数
        self.criterion = DIBLoss(
            sparsity_weight=self.config.get("sparsity_weight", 0.001),
            warmup_epochs=self.config.get("warmup_epochs", 16),
        )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # 混合精度
        self.scaler = GradScaler('cuda', enabled=(device.type == "cuda" and self.use_amp))

        # 温度调度器
        self.temp_scheduler = TemperatureScheduler(
            init_temp=self.config.get("init_temperature", 1.0),
            final_temp=self.config.get("final_temperature", 0.1),
            total_epochs=self.config.get("epochs", 50),
        )

        # 学习率调度器（在外部设置）
        self.lr_scheduler = None

        # 训练历史
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_rmse": [], "val_acc": [],
            "temperatures": [], "mask_retentions": [],
            "mask_history": [], "epoch_times": [],
            "learning_rates": [],
        }

        self.best_rmse = float("inf")
        self.best_epoch = 0

    def set_lr_scheduler(self, scheduler):
        self.lr_scheduler = scheduler

    def train_epoch(self, train_loader, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        self.criterion.set_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)

        # 更新温度
        if hasattr(self.model, 'module'):
            self.model.module.set_temperature(self.temp_scheduler.get_temperature(epoch))
        else:
            self.model.set_temperature(self.temp_scheduler.get_temperature(epoch))

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for i, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if self.use_amp and self.device.type == "cuda":
                with autocast(enabled=True):
                    pred, mask = self.model(x, return_mask=True)
                    loss, pred_loss, sparsity = self.criterion(pred, y, mask)
                    loss = loss / self.accumulation_steps

                self.scaler.scale(loss).backward()

                if (i + 1) % self.accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                pred, mask = self.model(x, return_mask=True)
                loss, pred_loss, sparsity = self.criterion(pred, y, mask)
                loss = loss / self.accumulation_steps
                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.accumulation_steps
            pbar.set_postfix({
                "loss": f"{loss.item() * self.accumulation_steps:.4f}",
                "sparsity": f"{sparsity.item():.3f}",
            })

        return total_loss / len(train_loader)

    def validate(self, val_loader, clim_mean=None, lat_weights=None):
        """验证"""
        return validate_epoch(
            self.model, val_loader, self.criterion,
            self.device, clim_mean, lat_weights,
        )

    def run_epoch(
        self,
        epoch: int,
        train_loader,
        val_loader,
        clim_mean=None,
        lat_weights=None,
        save_path: str = "dibfno_best.pth",
        mask_sample_loader=None,
    ):
        """运行一个完整 epoch（训练 + 验证 + 记录）"""
        t_start = time.time()

        # 训练
        train_loss = self.train_epoch(train_loader, epoch)

        # 验证
        val_loss, val_rmse, val_acc = self.validate(
            val_loader, clim_mean, lat_weights,
        )

        # 学习率调度
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_time = time.time() - t_start
        tau = self.temp_scheduler.get_temperature(epoch)

        # 记录掩码保留率
        mask_retention = 0.0
        if mask_sample_loader is not None:
            with torch.no_grad():
                sx, _ = next(iter(mask_sample_loader))
                sx = sx.to(self.device)
                _, mask = self.model(sx, return_mask=True)
                mask_retention = mask.mean().item()

        # 保存历史
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_rmse"].append(val_rmse)
        self.history["val_acc"].append(val_acc)
        self.history["temperatures"].append(tau)
        self.history["mask_retentions"].append(mask_retention)
        self.history["epoch_times"].append(epoch_time)
        self.history["learning_rates"].append(
            self.optimizer.param_groups[0]["lr"]
        )

        # 保存掩码快照
        total_epochs = self.config.get("epochs", 50)
        if epoch % max(1, total_epochs // 6) == 0 or epoch == 1 or epoch == total_epochs:
            if mask_sample_loader is not None:
                with torch.no_grad():
                    sx, _ = next(iter(mask_sample_loader))
                    sx = sx.to(self.device)
                    _, mask = self.model(sx, return_mask=True)
                    self.history["mask_history"].append(mask.clone().detach())

        # 保存最佳模型
        if val_rmse < self.best_rmse:
            self.best_rmse = val_rmse
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"  -> 最佳模型已保存 (RMSE: {val_rmse:.4f})")

        logger.info(
            f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"RMSE: {val_rmse:.4f} | ACC: {val_acc:.4f} | τ: {tau:.3f} | "
            f"Time: {epoch_time:.1f}s"
        )

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_acc": val_acc,
            "temperature": tau,
            "mask_retention": mask_retention,
        }

    def load_best_model(self, path: str = "dibfno_best.pth"):
        """加载最佳模型"""
        state_dict = torch.load(path, map_location=self.device)
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        logger.info(f"已加载最佳模型 (RMSE: {self.best_rmse:.4f})")