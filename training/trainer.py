"""组织完整训练过程，保存与恢复 checkpoint。"""

import csv
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .engine import train_one_epoch, validate


def seed_everything(seed):
    """尽量减少重复运行时的随机差异；不保证跨设备完全一致。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name):
    """自动选择 GPU，Apple 芯片使用 MPS。"""
    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif mps_available:
            name = "mps"
        else:
            name = "cpu"

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA")

    if name == "mps" and not mps_available:
        raise RuntimeError("当前环境无法使用 MPS")

    return torch.device(name)


def atomic_save(state, path):
    """先写临时文件，写完后替换目标，减少中断导致文件损坏的风险。"""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


class Trainer:
    def __init__(self, model, config, classes):
        self.config = config
        self.classes = list(classes)
        self.device = select_device(config.device)

        # 先移动模型，再创建优化器。
        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        # 每个 epoch 结束后调用一次 scheduler.step()。
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.min_learning_rate,
        )

        self.output_dir = Path(config.output_dir)
        self.start_epoch = 1
        self.best_val_accuracy = -1.0
        self.history = []

    def _save_checkpoint(self, epoch, filename):
        # checkpoint 不只是权重，还包含继续训练需要的状态。
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "classes": self.classes,
            "config": asdict(self.config),
            "history": self.history,
        }

        atomic_save(checkpoint, self.output_dir / filename)

    def _load_checkpoint(self, path):
        # 先读到 CPU，再将模型权重复制到模型所在设备。
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["classes"] != self.classes:
            raise ValueError("checkpoint 的类别名称或顺序与当前数据不一致")

        # 防止恢复时悄悄改变余弦调度计划或优化器设置。
        saved_config = checkpoint["config"]
        fixed_settings = (
            "epochs",
            "learning_rate",
            "momentum",
            "weight_decay",
            "min_learning_rate",
        )

        for key in fixed_settings:
            if saved_config[key] != getattr(self.config, key):
                raise ValueError(
                    f"恢复训练时 {key} 必须保持为 {saved_config[key]}。"
                    "此功能用于继续原计划内中断的训练。"
                )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # SGD 的动量张量也需要位于正确的设备。
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_accuracy = checkpoint["best_val_accuracy"]
        self.history = checkpoint["history"]

        print(
            f"已恢复第 {checkpoint['epoch']} 轮的 checkpoint，"
            f"从第 {self.start_epoch} 轮继续。"
        )

    def _save_history(self):
        path = self.output_dir / "history.csv"
        temporary_path = path.with_suffix(".csv.tmp")

        fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "seconds",
        ]

        with temporary_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

        temporary_path.replace(path)

    def fit(self, train_loader, val_loader, resume=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if resume is not None:
            self._load_checkpoint(resume)
        else:
            # 避免新的实验覆盖之前保存的结果。
            filenames = ("last.pt", "best.pt", "history.csv")
            if any(
                (self.output_dir / name).exists()
                for name in filenames
            ):
                raise FileExistsError(
                    f"{self.output_dir} 已有训练结果。"
                    "请换一个 output_dir，或指定 resume 恢复训练。"
                )

        print(f"训练设备：{self.device}")
        print(f"结果目录：{self.output_dir.resolve()}")

        if self.start_epoch > self.config.epochs:
            print("checkpoint 已完成计划中的全部 epoch，无需继续。")
            return self.history

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            start_time = time.perf_counter()

            # 记录本轮实际使用的学习率。
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch}/{self.config.epochs}"
                f" | learning_rate={current_lr:.6f}",
                flush=True,
            )

            train_metrics = train_one_epoch(
                model=self.model,
                loader=train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                log_every=self.config.log_every,
            )

            val_metrics = validate(
                model=self.model,
                loader=val_loader,
                criterion=self.criterion,
                device=self.device,
            )

            # 完成本轮训练和验证后，设置下一轮的学习率。
            self.scheduler.step()

            is_best = val_metrics["accuracy"] > self.best_val_accuracy

            if is_best:
                self.best_val_accuracy = val_metrics["accuracy"]

            row = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "seconds": time.perf_counter() - start_time,
            }
            self.history.append(row)

            # 保存的是 scheduler.step() 之后的状态，
            # 恢复时下一轮就能使用正确的学习率。
            if is_best:
                self._save_checkpoint(epoch, "best.pt")

            self._save_checkpoint(epoch, "last.pt")
            self._save_history()

            print(
                f"训练 loss={row['train_loss']:.4f}"
                f" | 训练 acc={row['train_accuracy']:.2%}"
                f" | 验证 loss={row['val_loss']:.4f}"
                f" | 验证 acc={row['val_accuracy']:.2%}"
                f" | 耗时={row['seconds']:.1f}s"
            )

            if is_best:
                print("已更新 best.pt：验证准确率达到目前最佳。")

        print(
            f"\n训练完成，最佳验证准确率："
            f"{self.best_val_accuracy:.2%}"
        )

        return self.history