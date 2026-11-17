"""本地 CIFAR-10 读取、划分、DataLoader 构建。不会下载数据。

数据须为已解压的官方 Python 版本，含 data_batch_1～5、test_batch、batches.meta。
保留官方 10000 张测试集；默认把 50000 张训练图片分为 45000 训练 + 5000 验证。
"""
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from .cogif import DataConfig
from .transform import build_transforms


class _TransformedSubset(Dataset):
    """共享原始数据，但让训练集与验证集使用不同的 transform。"""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), label


def _split_indices(targets, val_ratio, seed):
    """按类别划分，确保各类在训练与验证中都有样本。"""
    groups = defaultdict(list)
    for index, label in enumerate(targets):
        groups[label].append(index)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for label, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ValueError(f"类别 {label} 至少需要两张图片")
        rng.shuffle(indices)
        n_val = max(1, min(len(indices) - 1, round(len(indices) * val_ratio)))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_dataloaders(config=None):
    """返回字典：train、val、test 是 DataLoader，classes 是类别名列表。

    固定 seed 可复现划分和批次洗牌；不保证整个模型训练逐位复现。
    num_workers > 0 时，调用代码请放在 if __name__ == '__main__': 内。
    """
    config = config if config is not None else DataConfig()
    config.__post_init__()
    root = Path(config.data_path).expanduser()
    if root.name == "cifar-10-batches-py":
        root = root.parent
    folder = root / "cifar-10-batches-py"
    required = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"请在 dataload/cogif.py 中修改 data_path。\n"
            f"当前检查目录：{folder}\n缺少：{', '.join(missing)}\n"
            "需要已解压的 CIFAR-10 Python 版本。"
        )

    train_transform, eval_transform = build_transforms(config)

    # ===== 真正读取本地图片及标签的地方 =====
    # train=True：读取 data_batch_1～5，内部自动解析 pickle 并还原 RGB 图片。
    original_train = CIFAR10(root=str(root), train=True, download=False)
    # train=False：读取 test_batch，保持官方测试集独立。
    test_dataset = CIFAR10(
        root=str(root), train=False, download=False, transform=eval_transform,
    )

    # 先划分原始样本，再分别应用增强，防止训练增强影响验证集。
    train_indices, val_indices = _split_indices(
        original_train.targets, config.val_ratio, config.seed,
    )
    train_dataset = _TransformedSubset(original_train, train_indices, train_transform)
    val_dataset = _TransformedSubset(original_train, val_indices, eval_transform)

    def make_loader(dataset, shuffle, seed):
        return DataLoader(
            dataset, batch_size=config.batch_size, shuffle=shuffle,
            num_workers=config.num_workers, drop_last=False,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )

    return {
        "train": make_loader(train_dataset, True, config.seed),
        "val": make_loader(val_dataset, False, config.seed + 1),
        "test": make_loader(test_dataset, False, config.seed + 2),
        "classes": list(original_train.classes),
    }
