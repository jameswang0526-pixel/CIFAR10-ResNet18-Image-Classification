"""集中保存训练设置。"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # 完整遍历训练集的次数。
    epochs: int = 30

    # 初始学习率：每次更新参数的步幅。
    learning_rate: float = 0.01

    # SGD 的动量。
    momentum: float = 0.9

    # 权重衰减，用于正则化。
    weight_decay: float = 0.0005

    # 余弦调度器最终接近的学习率。
    min_learning_rate: float = 0.00001

    # auto 自动选择；也可以指定 mps、cuda 或 cpu。
    device: str = "auto"

    # 每隔多少个 batch 打印一次进度。
    log_every: int = 50

    # 保存权重和训练记录的文件夹。
    output_dir: str = "runs/resnet18"

    seed: int = 42

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs 必须至少为 1")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")

        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError(
                "min_learning_rate 必须在 0 与 learning_rate 之间"
            )

        if not 0 <= self.momentum < 1:
            raise ValueError("momentum 必须在 [0, 1) 之间")

        if self.weight_decay < 0:
            raise ValueError("weight_decay 不能为负数")

        if self.log_every < 1:
            raise ValueError("log_every 必须至少为 1")

        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device 必须是 auto、cpu、mps 或 cuda")