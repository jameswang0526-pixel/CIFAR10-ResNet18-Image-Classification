"""图片处理：仅训练集使用随机增强，验证集和测试集保持确定性。"""
from torchvision import transforms


def build_transforms(config):
    # ToTensor：RGB 图片 → [3, 32, 32] float32，像素从 0～255 转为 0～1。
    # 默认 Normalize：将 0～1 转为 -1～1。
    def base():
        return [transforms.ToTensor(), transforms.Normalize(config.mean, config.std)]

    augmentation = []
    if config.augment:
        augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    return transforms.Compose(augmentation + base()), transforms.Compose(base())
