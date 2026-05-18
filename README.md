<h3 align="center">tiny-cnn</h3>

<p align="center">
    一个用 C 语言实现的轻量级、卷积神经网络框架（支持自动求导、自定义高效堆栈内存管理与计算图）。
    <br />
    <a href="https://github.com/your_username/tiny-cnn"><strong>探索本项目的文档 »</strong></a>
    <br />
  </p>
</p>

## 目录

- [上手指南](#上手指南)
  - [开发前的准备](#开发前的准备)
  - [安装步骤](#安装步骤)
- [文件结构](#文件结构)
- [使用示例](#使用示例)
- [路线图](#路线图)
- [许可证](#许可证)
- [致谢](#致谢)

---

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

### 安装步骤

1. 克隆本仓库：
   **Bash**

   ```
   git clone [https://github.com/your_username/tiny-cnn.git](https://github.com/your_username/tiny-cnn.git)
   cd tiny-cnn
   ```
2. 放置数据集：
   请确保您已准备好格式化的 CIFAR-10 图像集，并存放在项目根目录外的相应路径（如 `../data/cifar10/`），或者根据您的本地需求修改 `main.c` 中的数据集读取路径。
3. 构建并编译：
   **Bash**

   ```
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```
4. 运行可执行程序：
   **Bash**

   ```
   ./bin/tiny-cnn
   ```

---

## 文件结构

```
tiny-cnn/
├── CMakeLists.txt
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
void cifar10_custom_resnet(stack *stk, nn_model *model) {
    // 1. 定义输入与目标张量 (3通道, 32x32 图像, 指定 BATCH_SIZE)
    nn_tensor *input = nn_layer_input(stk, model, 3, 32, 32, BATCH_SIZE);
    nn_tensor *target = nn_layer_target(stk, model, 1, 1, 10, BATCH_SIZE);

    // 2. 基础卷积 + 批归一化 + 激活函数
    nn_tensor *conv1 = nn_layer_conv2d(stk, model, input, 16, 3, 1, 1);
    nn_tensor *bn1 = nn_layer_batchnorm2d(stk, model, conv1);
    nn_tensor *relu1 = nn_layer_relu(stk, model, bn1);

    // 3. 堆叠残差块 (Residual Blocks)
    nn_tensor *block1_1 = nn_layer_residual_block(stk, model, relu1, 16, 16, 1);
    nn_tensor *block2_1 = nn_layer_residual_block(stk, model, block1_1, 16, 32, 2); // 步长为 2 降采样
    nn_tensor *block3_1 = nn_layer_residual_block(stk, model, block2_1, 32, 64, 2);

    // 4. 全局平均池化、Dropout 与全连接层输出
    nn_tensor *gap = nn_layer_gapool2d(stk, model, block3_1);
    nn_tensor *dropout = nn_layer_dropout(stk, model, gap, 0.2F);
    nn_tensor *fc = nn_layer_linear(stk, model, dropout, 10);

    // 5. 最终 Softmax 输出与交叉熵损失函数
    nn_tensor *output = nn_layer_softmax(stk, model, fc);
    nn_tensor *loss = nn_layer_cross_entropy(stk, model, output, target);
}
```

---

## 路线图

* [X] 基于纯 C 语言的自动求导与计算图系统
* [X] 高效对齐的自定义内存池（Stack Allocation）
* [X] 支持余弦退火、EMA 及标签平滑
* [ ] 支持 GPU / CUDA 算子硬件加速
* [ ] 增加更多经典网络拓扑的支持（如 MobileNet, VGG）
* [ ] 提供模型参数导出与导入的通用格式（.json 或可序列化二进制格式）

---

## 许可证

本项目基于 **MIT License** 许可证开源。请查阅 `LICENSE` 文件了解更多详情。

---

## 致谢

* [OpenBLAS团队](https://www.openblas.net/)
* [OpenMP规范委员会](https://www.openmp.org/)
* [stb 单文件开源库系列](https://github.com/nothings/stb)
* [Best-README-Template 提供者](https://github.com/shaojintian/Best_README_template)
