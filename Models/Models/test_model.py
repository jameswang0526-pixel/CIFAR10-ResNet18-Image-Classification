"""最小数据流测试。默认随机输入；--real 接入你已有的 dataload。"""
import argparse
import torch
from torch import nn
from models import resnet18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="使用本地 CIFAR-10")
    parser.add_argument("--backward", action="store_true", help="额外验证反向传播")
    args = parser.parse_args()
    torch.manual_seed(42)

    if args.real:
        from dataload import DataConfig, build_dataloaders
        loaders = build_dataloaders(DataConfig(batch_size=2, num_workers=0))
        images, labels = next(iter(loaders["train"]))
        classes = loaders["classes"]
        print("输入来源：本地 CIFAR-10 训练集的两张图片")
    else:
        images = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        classes = None
        print("输入来源：随机 Tensor，仅验证网络，不代表真实图片或准确率")

    model = resnet18(num_classes=len(classes) if classes else 10)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    # eval 控制 BatchNorm；no_grad 避免为这次观察建立反向传播计算图。
    model.eval()
    with torch.no_grad():
        logits = model(images, verbose=True)
    assert logits.shape == (images.shape[0], model.fc.out_features)
    assert torch.isfinite(logits).all(), "输出含 NaN/Inf"
    print("\n第一张图的 10 个原始分数：", logits[0].tolist())
    predicted = logits.argmax(dim=1)
    print("预测编号：", predicted.tolist())
    print("真实/测试标签：", labels.tolist())
    if classes:
        print("预测类别：", [classes[i] for i in predicted.tolist()])
    print("注意：模型尚未训练，预测没有分类效果保证。")

    if args.backward:
        model.train()
        model.zero_grad(set_to_none=True)
        loss = nn.CrossEntropyLoss()(model(images), labels)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} 没有梯度"
            assert torch.isfinite(parameter.grad).all(), f"{name} 梯度含 NaN/Inf"
        print(f"反向传播通过：loss={loss.item():.4f}，所有参数都有有限梯度")
    print("模型测试通过！")


if __name__ == "__main__":
    main()
