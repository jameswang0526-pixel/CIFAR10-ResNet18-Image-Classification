# CIFAR-10 ResNet-18 模型模块

把 models 文件夹和 test_model.py 放到现有项目根目录，和 dataload、run.py 同级。
保留原来的数据模块。包入口文件必须是 `__init__.py`（前后各两个下划线），
不是 `init.py`。现有配置文件名 `cogif.py` 可以保持不变。

## 运行

在项目根目录，用你已安装 torch/torchvision 的 Python 环境运行：

```bash
python test_model.py
python test_model.py --real
python test_model.py --real --backward
```

第一条无需数据，只用两个随机 Tensor 验证数据流。
第二条读取你现有 DataConfig 的本地路径，取两张真实训练图片。
第三条额外计算交叉熵并验证所有参数梯度。它不更新参数，也不是完整训练程序。
默认在 CPU 上运行，便于做最小测试。模型可在未来训练模块中通过 `.to(device)` 移到 MPS。

## 文件职责

- `models/__init__.py`：对外提供导入入口。
- `models/resnet.py`：完整的 BasicBlock、ResNet-18、参数初始化与数据流打印。
- `test_model.py`：随机输入或真实数据的前向测试，以及可选的反向传播验证。

## 形状变化

形状 `[N, C, H, W]` 分别表示图片数量、通道数、高、宽。
这里 N=2，实际使用 batch_size=64 时 N=64，其他维度相同。

| 步骤 | 输出形状 | 作用 |
|---|---|---|
| 输入 | [2, 3, 32, 32] | RGB 图片 |
| stem | [2, 64, 32, 32] | 提取初始特征 |
| layer1：2 个残差块 | [2, 64, 32, 32] | 保持空间大小 |
| layer2：2 个残差块 | [2, 128, 16, 16] | 增加通道，宽高减半 |
| layer3：2 个残差块 | [2, 256, 8, 8] | 继续提取特征 |
| layer4：2 个残差块 | [2, 512, 4, 4] | 继续提取特征 |
| avgpool | [2, 512, 1, 1] | 每个通道求空间平均 |
| flatten | [2, 512] | 展平成特征向量 |
| fc | [2, 10] | 每张图输出 10 个类别分数 |

## 残差块如何工作

主分支对 x 执行：Conv3×3 → BN → ReLU → Conv3×3 → BN，得到 F(x)。
捷径分支保留 x；若通道数或宽高改变，就通过 Conv1×1 + BN 对齐形状。
然后逐元素相加，最后做 ReLU：`output = ReLU(F(x) + shortcut(x))`。
相加不是拼接，不会让通道数翻倍。两条分支都参与计算和梯度传播。

例如 layer2 的第一个残差块：输入 [2,64,32,32]；
主分支输出 [2,128,16,16]，捷径也变成 [2,128,16,16]，相加后仍为该形状。

18 层的常规计数：入口 1 个卷积 + 8 个残差块 × 2 个卷积 + 1 个全连接。
捷径投影卷积不计入这个命名。
这是为 CIFAR-10 改入口的 ResNet-18，不是原论文专门用于 CIFAR 的 ResNet-20。
与 ImageNet 版相比，入口改为 3×3、步长 1，并去掉入口最大池化，避免小图过早缩小。

## 接入未来训练模块

```python
from models import resnet18

model = resnet18(num_classes=len(loaders["classes"]))
images, labels = next(iter(loaders["train"]))
logits = model(images)                 # 默认不打印
# logits = model(images, verbose=True)  # 打印全部数据流
loss = criterion(logits, labels)       # criterion 可用 nn.CrossEntropyLoss()
```

logits 是原始分数，不是概率；训练时直接交给 CrossEntropyLoss。
仅展示概率时可调用 `logits.softmax(dim=1)`。
新建模型尚未训练，测试通过仅代表数据能流通，不能说明分类准确率高。

结构参考（实现为独立教学代码）：
https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
