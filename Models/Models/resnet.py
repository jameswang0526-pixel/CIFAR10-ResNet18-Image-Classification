"""从零实现 CIFAR-10 版 ResNet-18；只依赖 torch，不下载权重。

保留 ResNet-18 的 [2, 2, 2, 2] 残差块布局。
为适配 32×32 图片，入口使用 3×3/stride=1 卷积，不使用入口最大池化。
"""
import torch
from torch import nn


def _show(enabled, name, x):
    """只打印形状，不保存中间 Tensor，避免额外占用内存。"""
    if enabled:
        print(f"{name:44s} {list(x.shape)}")


class BasicBlock(nn.Module):
    """残差块：输出 = ReLU(主分支 F(x) + 捷径分支 shortcut(x))。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 两条分支必须形状一致才能逐元素相加。
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()  # 形状相同时直接传递原输入。
        self.relu2 = nn.ReLU()

    def forward(self, x, verbose=False, name="block"):
        _show(verbose, f"  {name} input", x)
        identity = self.shortcut(x)
        out = self.conv1(x)
        _show(verbose, "    main.conv1", out)
        out = self.bn1(out)
        _show(verbose, "    main.bn1", out)
        out = self.relu1(out)
        _show(verbose, "    main.relu1", out)
        out = self.conv2(out)
        _show(verbose, "    main.conv2", out)
        out = self.bn2(out)
        _show(verbose, "    main.bn2 = F(x)", out)
        kind = "identity" if isinstance(self.shortcut, nn.Identity) else "1x1 conv + BN"
        _show(verbose, f"    shortcut ({kind})", identity)
        out = out + identity  # 对应位置相加，不是拼接；通道数不会翻倍。
        _show(verbose, "    ADD: F(x) + shortcut(x)", out)
        out = self.relu2(out)
        _show(verbose, "    output.relu", out)
        return out


class ResNet(nn.Module):
    """四个阶段，每阶段两个 BasicBlock，共 8 个残差块。"""

    def __init__(self, num_classes=10):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes 必须是正整数")
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    @staticmethod
    def _make_layer(in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Linear 使用 PyTorch 自带的初始化。

    def forward(self, x, verbose=False):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("输入必须是 [batch, 3, height, width] 的 Tensor")
        _show(verbose, "INPUT", x)
        for label, operation in zip(("conv", "bn", "relu"), self.stem):
            x = operation(x)
            _show(verbose, f"stem.{label}", x)
        for index, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4), 1):
            for block_index, block in enumerate(layer, 1):
                x = block(x, verbose=verbose, name=f"layer{index}.block{block_index}")
            _show(verbose, f"LAYER{index} OUTPUT", x)
        x = self.avgpool(x)
        _show(verbose, "avgpool", x)
        x = torch.flatten(x, start_dim=1)  # 保留 batch 维度。
        _show(verbose, "flatten", x)
        x = self.fc(x)
        _show(verbose, "fc -> logits", x)
        # 不做 softmax：后续 CrossEntropyLoss 直接接收 logits。
        return x


def resnet18(num_classes=10):
    """创建新的、尚未训练的模型。"""
    return ResNet(num_classes=num_classes)
