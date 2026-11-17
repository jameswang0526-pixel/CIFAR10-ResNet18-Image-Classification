"""单个 epoch 的训练，以及验证逻辑。"""

import torch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    log_every=50,
):
    # 进入训练模式，BatchNorm 会更新运行统计量。
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        # 讲图片和标签移动到计算设备上。
        images = images.to(device)
        labels = labels.to(device)

        # 1. 清空上一个 batch 的梯度。
        optimizer.zero_grad(set_to_none=True)

        # 2. 前向传播，得到每张图片的类别分数。
        logits = model(images)

        # 3. 计算预测分数与真实标签之间的损失。
        # CrossEntropyLoss 直接接收 logits，不要先做 softmax。
        loss = criterion(logits, labels)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"第 {batch_index} 个 batch 的训练 loss 出现 NaN/Inf"
            )

        # 4. 反向传播，计算每个参数的梯度。
        loss.backward()

        # 5. 根据梯度更新模型参数。
        optimizer.step()

        # 当前 batch 实际包含的图片数量。
        batch_size = labels.size(0)

        # loss 默认是 batch 平均值，乘图片数还原成总和。
        # 这样最后一个较小的 batch 不会被赋予过大的权重。
        total_loss += loss.item() * batch_size

        predicted = logits.detach().argmax(dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += batch_size

        if batch_index % log_every == 0 or batch_index == len(loader):
            print(
                f"  batch {batch_index}/{len(loader)}"
                f" | loss={total_loss / total_samples:.4f}"
                f" | acc={total_correct / total_samples:.2%}",
                flush=True,
            )

    if total_samples == 0:
        raise ValueError("训练 DataLoader 没有提供任何样本")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    # 验证不计算梯度，也不更新模型参数。
    # 记录原来的模式，结束后恢复，便于其他模块复用。
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    try:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError("验证 loss 出现 NaN/Inf")

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predicted = logits.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("验证 DataLoader 没有提供任何样本")

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }
    finally:
        model.train(was_training)