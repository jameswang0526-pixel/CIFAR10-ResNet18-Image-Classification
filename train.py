
"""项目训练入口：加载数据、创建模型并启动训练。"""

from dataload import DataConfig, build_dataloaders
from Models.Models import resnet18
from training import TrainConfig, Trainer, seed_everything


def main():

    # 1. 创建训练配置
    train_config = TrainConfig()

    # 2. 固定随机种子，尽量保证实验可重复
    seed_everything(train_config.seed)

    # 3. 创建数据配置
    data_config = DataConfig(seed=train_config.seed)

    # 4. 加载 CIFAR-10 数据
    loaders = build_dataloaders(data_config)

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    classes = loaders["classes"]

    print(f"类别数量：{len(classes)}")
    print(f"类别名称：{classes}")
    print(f"训练图片数量：{len(train_loader.dataset)}")
    print(f"验证图片数量：{len(val_loader.dataset)}")
    print(f"测试图片数量：{len(test_loader.dataset)}")

    # 5. 根据类别数量创建 ResNet-18
    model = resnet18(
        num_classes=len(classes) if classes else 10
    )

    # 6. 创建 Trainer，初始化 Loss、Optimizer 和 Scheduler
    trainer = Trainer(
        model=model,
        config=train_config,
        classes=classes,
    )

    # 7. 开始训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print("\n训练流程结束！")
    print(f"总共完成 {len(history)} 个 Epoch。")



if __name__ == "__main__":
    main()
