"""数据模块配置。请先修改 data_path 为你 Mac 上的实际路径。"""
from dataclasses import dataclass


@dataclass
class DataConfig:
    # 支持 cifar-10-batches-py 本身或其上一级目录。
    data_path: str = "Your pathway_cifar-10-batches-py"
    batch_size: int = 64
    val_ratio: float = 0.1  # 从官方训练集取 10% 作验证集
    seed: int = 42
    num_workers: int = 0  # Mac 下默认使用单进程加载
    augment: bool = True
    mean: tuple = (0.5, 0.5, 0.5)
    std: tuple = (0.5, 0.5, 0.5)

    def __post_init__(self):
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio 必须在 0 和 1 之间")
        for name, minimum in (("batch_size", 1), ("num_workers", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} 必须是大于等于 {minimum} 的整数")
        if len(self.mean) != 3 or len(self.std) != 3 or any(v <= 0 for v in self.std):
            raise ValueError("mean/std 需要三个通道的数值，std 必须大于 0")
