
"""项目可视化入口：展示模型在 CIFAR-10 测试集上预测错误的图片。"""

import torch
import matplotlib.pyplot as plt

from dataload import DataConfig, build_dataloaders
from Models.Models import resnet18
from training import TrainConfig, seed_everything
from training.trainer import select_device


def main():
    # 1. 初始化配置
    train_config = TrainConfig()
    seed_everything(train_config.seed)

    data_config = DataConfig(seed=train_config.seed)

    # 2. 加载 CIFAR-10 测试集
    print("正在加载 CIFAR-10 数据……")

    loaders = build_dataloaders(data_config)

    test_loader = loaders["test"]
    classes = loaders["classes"]

    # 3. 选择运行设备
    device = select_device(train_config.device)
    print(f"当前运行设备：{device}")

    # 4. 创建 ResNet-18 模型
    model = resnet18(num_classes=len(classes))
    model = model.to(device)

    # 5. 加载训练过程中保存的最佳模型
    checkpoint_path = train_config.output_dir + "/best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["classes"] != classes:
        raise ValueError("模型类别与测试集类别不一致！")

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"已加载最佳模型：Epoch {checkpoint['epoch']}")

    # 6. 设置模型为评估模式
    model.eval()

    # 7. 保存预测错误的图片
    wrong_images = []
    wrong_labels = []
    predicted_labels = []

    # 关闭梯度计算
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # 模型进行预测
            logits = model(images)

            # 找出每张图片分数最高的类别
            predictions = logits.argmax(dim=1)

            # 判断哪些图片预测错误
            wrong_mask = predictions != labels

            # 获取预测错误的图片
            wrong_indices = torch.where(wrong_mask)[0]

            for index in wrong_indices:

                wrong_images.append(images[index].cpu())
                wrong_labels.append(labels[index].item())
                predicted_labels.append(predictions[index].item())

                # 找够 16 张就停止
                if len(wrong_images) >= 16:
                    break

            if len(wrong_images) >= 16:
                break

    # 8. 检查是否找到了错误图片
    if len(wrong_images) == 0:
        print("没有找到预测错误的图片！")
        return

    # 9. 创建图片展示区域
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    # 将二维坐标轴转换成一维，方便逐张展示
    axes = axes.flatten()

    # CIFAR-10 的标准归一化参数
    mean = torch.tensor(data_config.mean).view(3, 1, 1)
    std = torch.tensor(data_config.std).view(3, 1, 1)

    # 10. 逐张显示预测错误的图片
    for i in range(len(wrong_images)):

        image = wrong_images[i]

        # 反归一化，恢复图片原来的像素范围
        image = image * std + mean

        # 限制像素值在 0 到 1 之间
        image = image.clamp(0, 1)

        # Tensor 转换成 Matplotlib 可以显示的格式
        image = image.permute(1, 2, 0).numpy()

        actual_name = classes[wrong_labels[i]]
        predicted_name = classes[predicted_labels[i]]

        axes[i].imshow(image)

        axes[i].set_title(
            f"Actual: {actual_name}\nPredicted: {predicted_name}",
            color="red",
            fontsize=10,
        )

        axes[i].axis("off")

    # 隐藏没有使用的子图
    for i in range(len(wrong_images), len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        "ResNet-18 Misclassified CIFAR-10 Images",
        fontsize=16,
    )

    plt.tight_layout()

    # 11. 展示结果
    plt.show()

    print(f"一共展示了 {len(wrong_images)} 张预测错误的图片。")


if __name__ == "__main__":
    main()
