
"""模型评估模块：计算 Loss、Accuracy、混淆矩阵和分类指标。"""

import torch
from torch import nn


@torch.no_grad()
def evaluate(model, loader, classes, device):
    """
    在验证集或测试集上评估模型。

    参数：
        model: 已训练好的 CNN 模型
        loader: 验证集或测试集 DataLoader
        classes: CIFAR-10 的类别名称列表
        device: 模型运行设备，例如 mps、cuda 或 cpu

    返回：
        包含 loss、accuracy、confusion_matrix 等指标的字典
    """

    # 1. 切换到评估模式。
    # BatchNorm 不再更新运行统计量。
    was_training = model.training
    model.eval()

    num_classes = len(classes)

    # 使用 CrossEntropyLoss 计算分类损失。
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 创建一个 num_classes × num_classes 的混淆矩阵。
    confusion_matrix = torch.zeros(
        num_classes * num_classes,
        dtype=torch.int64,
        device=device,
    )

    try:
        # 2. 遍历整个验证集或测试集。
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # 3. 前向传播，得到每张图片的分类分数。
            logits = model(images)

            # 4. 计算当前 batch 的 loss。
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            # 5. 取得每张图片预测分数最高的类别。
            predicted = logits.argmax(dim=1)

            # 6. 累计预测正确的图片数量。
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size

            # 7. 统计真实类别和预测类别的组合。
            # 行代表真实类别，列代表预测类别。
            indices = labels * num_classes + predicted

            confusion_matrix += torch.bincount(
                indices,
                minlength=num_classes * num_classes,
            )

        if total_samples == 0:
            raise ValueError("评估 DataLoader 没有提供任何样本")

        # 8. 将一维统计结果还原成二维矩阵。
        confusion_matrix = confusion_matrix.reshape(
            num_classes,
            num_classes,
        )

        # 转移到 CPU，方便使用 Python 和 NumPy 处理。
        matrix = confusion_matrix.cpu().numpy()

        # 9. 计算整体指标。
        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # 10. 计算每个类别的分类指标。
        per_class_metrics = []

        for i, class_name in enumerate(classes):
            # 该类别真实图片的总数。
            support = int(matrix[i, :].sum())

            # 该类别预测正确的图片数。
            correct = int(matrix[i, i])

            # 该类别被模型预测出来的总数。
            predicted_count = int(matrix[:, i].sum())

            # Recall = 正确识别该类别的数量 / 该类别真实数量。
            recall = correct / support if support > 0 else 0.0

            # Precision = 正确识别该类别的数量 / 预测为该类别的数量。
            precision = (
                correct / predicted_count
                if predicted_count > 0
                else 0.0
            )

            # F1 是 Precision 和 Recall 的调和平均数。
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            per_class_metrics.append({
                "class_name": class_name,
                "accuracy": recall,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct": correct,
                "total": support,
            })

        # 11. 找出最容易混淆的类别组合。
        confused_pairs = []

        for actual_index in range(num_classes):
            for predicted_index in range(num_classes):

                # 跳过对角线，因为对角线代表预测正确。
                if actual_index == predicted_index:
                    continue

                count = int(matrix[actual_index, predicted_index])

                if count > 0:
                    confused_pairs.append({
                        "actual_class": classes[actual_index],
                        "predicted_class": classes[predicted_index],
                        "count": count,
                    })

        # 按照错误数量从高到低排列。
        confused_pairs.sort(
            key=lambda item: item["count"],
            reverse=True,
        )

        # 12. 返回结构化结果。
        return {
            "loss": average_loss,
            "accuracy": accuracy,
            "total_samples": total_samples,
            "total_correct": total_correct,
            "confusion_matrix": matrix.tolist(),
            "per_class_metrics": per_class_metrics,
            "most_confused_pairs": confused_pairs[:10],
        }

    finally:
        # 恢复模型原来的训练/评估模式。
        model.train(was_training)
