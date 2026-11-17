
"""项目评估入口：加载最佳模型，并在 CIFAR-10 测试集上进行评估。"""

import torch

from dataload import DataConfig, build_dataloaders
from Models.Models import resnet18
from training import TrainConfig, seed_everything
from evaluation import evaluate


def main():

    # ==========================================
    # 1. 准备配置和随机种子
    # ==========================================

    train_config = TrainConfig()

    seed_everything(train_config.seed)

    data_config = DataConfig(seed=train_config.seed)

    # ==========================================
    # 2. 加载 CIFAR-10 数据
    # ==========================================

    print("正在加载 CIFAR-10 数据……")

    loaders = build_dataloaders(data_config)

    test_loader = loaders["test"]
    classes = loaders["classes"]

    print(f"类别数量：{len(classes)}")
    print(f"类别名称：{classes}")
    print(f"测试图片数量：{len(test_loader.dataset)}")

    # ==========================================
    # 3. 选择计算设备
    # ==========================================

    from training.trainer import select_device

    device = select_device(train_config.device)

    print(f"评估设备：{device}")

    # ==========================================
    # 4. 创建 ResNet-18 模型
    # ==========================================

    model = resnet18(
        num_classes=len(classes) if classes else 10
    )

    model = model.to(device)

    # ==========================================
    # 5. 加载训练过程中保存的最佳模型
    # ==========================================

    checkpoint_path = train_config.output_dir + "/best.pt"

    print(f"\n正在加载模型：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # 检查 checkpoint 的类别顺序是否一致。
    if checkpoint["classes"] != classes:
        raise ValueError(
            "checkpoint 的类别名称或顺序与当前数据不一致"
        )

    # 将最佳模型参数加载到当前模型。
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"最佳模型来自 Epoch {checkpoint['epoch']}"
    )

    print(
        f"最佳验证准确率："
        f"{checkpoint['best_val_accuracy']:.2%}"
    )

    # ==========================================
    # 6. 在测试集上评估模型
    # ==========================================

    print("\n开始评估测试集……")

    test_results = evaluate(
        model=model,
        loader=test_loader,
        classes=classes,
        device=device,
    )

    # ==========================================
    # 7. 输出整体评估结果
    # ==========================================

    print("\n========== Test Results ==========")

    print(f"Test Loss: {test_results['loss']:.4f}")

    print(
        f"Test Accuracy: "
        f"{test_results['accuracy']:.2%}"
    )

    print(
        f"正确识别：{test_results['total_correct']}"
        f" / {test_results['total_samples']}"
    )

    # ==========================================
    # 8. 输出 Confusion Matrix
    # ==========================================

    print("\n========== Confusion Matrix ==========")

    print("行 = 真实类别，列 = 模型预测类别")

    print("类别".ljust(15), end="")

    for name in classes:
        print(f"{name[:8]:>10}", end="")

    print()

    for class_name, row in zip(
        classes,
        test_results["confusion_matrix"],
    ):
        print(f"{class_name:<15}", end="")

        for count in row:
            print(f"{count:>10}", end="")

        print()

    # ==========================================
    # 9. 输出 Per-class Metrics
    # ==========================================

    print("\n========== Per-class Metrics ==========")

    for item in test_results["per_class_metrics"]:

        print(
            f"{item['class_name']:<12}"
            f"Accuracy: {item['accuracy']:.2%}"
            f" | Precision: {item['precision']:.2%}"
            f" | Recall: {item['recall']:.2%}"
            f" | F1: {item['f1']:.2%}"
            f" | {item['correct']}/{item['total']}"
        )

    # ==========================================
    # 10. 输出最容易混淆的类别
    # ==========================================

    print("\n========== Most Confused Pairs ==========")

    for item in test_results["most_confused_pairs"]:

        print(
            f"{item['actual_class']} "
            f"被误认为 {item['predicted_class']}"
            f"：{item['count']} 张"
        )

    print("\n测试集评估完成！")


if __name__ == "__main__":
    main()
