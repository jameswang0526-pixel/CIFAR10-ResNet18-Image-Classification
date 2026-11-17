
"""项目预测入口：使用训练好的 ResNet-18 对单张图片进行分类。"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dataload import DataConfig
from Models.Models import resnet18
from training import TrainConfig
from training.trainer import select_device


# ==================================================
# 1. 配置需要识别的图片路径
# ==================================================

IMAGE_PATH = "Imy_cat.jpg"


def main():

    # 2. 初始化配置
    train_config = TrainConfig()
    data_config = DataConfig()

    # 3. 检查图片是否存在
    image_path = Path(IMAGE_PATH)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"找不到图片：{image_path.resolve()}\n"
            "请检查 IMAGE_PATH 是否填写正确。"
        )

    # 4. 选择运行设备
    device = select_device(train_config.device)
    print(f"当前运行设备：{device}")

    # 5. 创建 ResNet-18 模型
    checkpoint_path = (
        Path(__file__).resolve().parent
        / train_config.output_dir
        / "best.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"找不到模型权重：{checkpoint_path}"
        )

    # 6. 加载训练好的模型权重
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]

    model = resnet18(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # 7. 设置为评估模式
    model.eval()

    # 8. 定义图片预处理流程
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=data_config.mean,
            std=data_config.std,
        ),
    ])

    # 9. 读取用户提供的图片
    image = Image.open(image_path).convert("RGB")

    # 10. 对图片进行预处理
    image_tensor = transform(image)

    # 增加一个 Batch 维度
    # 原来：[3, 32, 32]
    # 现在：[1, 3, 32, 32]
    image_tensor = image_tensor.unsqueeze(0)

    # 将图片移动到模型所在的设备
    image_tensor = image_tensor.to(device)

    # 11. 模型预测
    with torch.no_grad():

        logits = model(image_tensor)

        # 将原始分数转换为概率
        probabilities = torch.softmax(logits, dim=1)

        # 获取概率最高的 3 个类别
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(3, len(classes)),
            dim=1,
        )

    # 12. 输出预测结果
    predicted_index = top_indices[0, 0].item()
    predicted_class = classes[predicted_index]
    confidence = top_probabilities[0, 0].item()

    print("\n========== Prediction Result ==========")

    print(f"Image: {image_path.name}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")

    print("\n========== Top 3 Predictions ==========")

    for rank in range(top_indices.shape[1]):

        class_index = top_indices[0, rank].item()
        class_name = classes[class_index]
        probability = top_probabilities[0, rank].item()

        print(
            f"{rank + 1}. {class_name}: "
            f"{probability:.2%}"
        )

    print("\n图片识别完成！")


if __name__ == "__main__":
    main()
