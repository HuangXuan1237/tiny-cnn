<h3 align="center">tiny-cnn</h3>

<p align="center">
    一个用 C 语言实现的轻量级、卷积神经网络框架（支持自动求导、自定义高效堆栈内存管理与计算图）。
  </p>
</p>

## 目录

- [致谢](#致谢)
- [上手指南](#上手指南)
  - [开发前的准备](#开发前的准备)
  - [获取数据集](#安装步骤)
- [文件结构](#文件结构)
- [使用示例](#使用示例)
- [路线图](#路线图)
- [许可证](#许可证)

---

## 致谢

* 本项目的开发受到了 **Magicalbat** 的精彩视频 [《coding a machine learning library in c from scratch》](https://www.youtube.com/watch?v=hL_n_GljC0I)的深度启发。非常感谢 **Magicalbat** 分享的优雅架构思路，特别是关于视频中探讨的高效 Arena（堆栈式）内存分配管理策略。
* 本项目中的 `pcg32` 随机数生成器实现基于 GitHub 上的开源项目 [PCG-C (pcg-random)](https://github.com/imneme/pcg-c)，为其高效且统计学优异的伪随机性能表示感谢。

## 上手指南

以下指南将帮助您在本地搭建开发环境并运行此项目。

### 开发前的准备

在编译本项目之前，请确保您的 Linux 系统（推荐 Ubuntu 24.04）已安装以下依赖：

* **GCC / Clang** (支持 C17 标准)
* **CMake** (版本 >= 3.16)
* **OpenBLAS** (矩阵乘法硬件加速)
* **OpenMP** (多线程并行计算支持)

您可以通过运行以下命令在 Ubuntu 上快速安装它们：

```sh
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev libomp-dev
```

### 获取数据集

本项目支持经典视觉数据集。请确保在训练前已下载并格式化好对应的数据集。

1. 生成数据集：
   以 **CIFAR-10** 为例，项目内已为您提供了现成的数据获取与转换脚本。您只需在项目根目录下执行以下命令，脚本会自动下载并将其转化为本项目专用的二进制格式：

   ```
   python ./data/cifar10/cifar10.py
   ```
2. 自定义或扩展其他数据集

   如果您希望获取并处理其他数据集，该 Python 脚本内同样集成了 **MNIST** 和 **Fashion-MNIST** 的处理逻辑。您可以将该脚本作为 **模板** 。如果您需要引入全新的自定义数据集，可以参考脚本中相关函数的实现结构：

---

## 文件结构

```
tiny-cnn
├── CMakeLists.txt
├── data/
│   ├── cifar10/
│   ├── fashion-mnist/
│   └── mnist/
├── include/
│   ├── main.h
│   ├── nn.h
│   ├── matrix.h
│   ├── optimizer.h
│   ├── data_utils.h
│   ├── stack.h
│   ├── pcg32.h
│   └── utils.h
├── third_party/
└── src/
    ├── main.c
    ├── nn.c
    ├── matrix.c
    ├── optimizer.c
    ├── data_utils.c
    ├── stack.c
    ├── pcg32.c
    └── utils.c
```

---

## 使用示例

以下是如何在代码中利用此框架快速定义并训练一个自定义 **ResNet** 模型的精简范例（源自 `main.c`）：

**C**

```
void cifar10_custom_resnet(stack *ar, nn_model *model) {
    // 1. 定义输入与目标张量 (3通道, 32x32 图像, 指定 BATCH_SIZE)
    nn_tensor *input = nn_layer_input(ar, model, 3, 32, 32, BATCH_SIZE);
    nn_tensor *target = nn_layer_target(ar, model, 1, 1, 10, BATCH_SIZE);

    // 2. 基础卷积 + 批归一化 + 激活函数
    nn_tensor *conv1 = nn_layer_conv2d(ar, model, input, 16, 3, 1, 1);
    nn_tensor *bn1 = nn_layer_batchnorm2d(ar, model, conv1);
    nn_tensor *relu1 = nn_layer_relu(ar, model, bn1);

    // 3. 堆叠残差块 (Residual Blocks)
    nn_tensor *block1_1 = nn_layer_residual_block(ar, model, relu1, 16, 16, 1);
    nn_tensor *block2_1 = nn_layer_residual_block(ar, model, block1_1, 16, 32, 2); // 步长为 2 降采样
    nn_tensor *block3_1 = nn_layer_residual_block(ar, model, block2_1, 32, 64, 2);

    // 4. 全局平均池化、Dropout 与全连接层输出
    nn_tensor *gap = nn_layer_gapool2d(ar, model, block3_1);
    nn_tensor *dropout = nn_layer_dropout(ar, model, gap, 0.2F);
    nn_tensor *fc = nn_layer_linear(ar, model, dropout, 10);

    // 5. 最终 Softmax 输出与交叉熵损失函数
    nn_tensor *output = nn_layer_softmax(ar, model, fc);
    nn_tensor *loss = nn_layer_cross_entropy(ar, model, output, target);
}
```

---

## 路线图

* [X] 基于 C 语言的自动求导与计算图系统
* [X] 高效对齐的自定义内存池（Stack Allocation）
* [X] 支持余弦退火、EMA 及标签平滑
* [ ] 支持 GPU / CUDA 算子硬件加速
* [ ] 增加更多经典网络拓扑的支持（如 MobileNet, VGG）
* [ ] 提供模型参数导出与导入的通用格式（.json 或可序列化二进制格式）

---

## 许可证

本项目基于 **MIT License** 许可证开源。请查阅 [LICENSE](LICENSE) 文件了解更多详情。
