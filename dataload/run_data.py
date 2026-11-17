from dataload import build_dataloaders

#主要执行流程
def main():
    print("正在加载 CIFAR-10……")
    #调用函数，读取数据，划分子集，准备处理流程，创建dataloader
    loaders = build_dataloaders()
#依次检查三个数据集
    for name in ["train", "val", "test"]:
        print(f"{name} 图片数量：{len(loaders[name].dataset)}")

    images, labels = next(iter(loaders["train"]))
    print("图片形状：", images.shape)#shape表示各个维度的大小
    print("标签形状：", labels.shape)
    print("数据加载成功！")


if __name__ == "__main__":
    main()