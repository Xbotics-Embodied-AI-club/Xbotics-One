"""基于 CNN 的 MNIST 手写数字识别模型。

本模块实现了一个简单的卷积神经网络（CNN），用于 MNIST 手写数字分类任务。
网络结构包含两个卷积层和两个全连接层，能够达到较高的识别准确率。
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNNMNIST(nn.Module):
    """用于 MNIST 分类的卷积神经网络模型。

    该网络采用经典的 LeNet 风格 CNN 架构。

    网络结构:
        - Conv2d(1->10, 5x5): 28x28 -> 24x24
        - MaxPool2d(2x2): 24x24 -> 12x12
        - Conv2d(10->20, 5x5): 12x12 -> 8x8
        - MaxPool2d(2x2): 8x8 -> 4x4
        - Dropout(0.25)
        - Flatten: 20*4*4 = 320
        - Linear(320->100)
        - Linear(100->10)
        - LogSoftmax

    Example:
        >>> model = CNNMNIST()
        >>> x = torch.randn(1, 1, 28, 28)
        >>> output = model(x)
        >>> output.shape
        torch.Size([1, 10])
    """

    def __init__(self):
        """初始化 CNN 网络结构。"""
        super().__init__()

        # 第一个卷积层: 1 -> 10 通道, 5x5 卷积核
        # 输入: (N, 1, 28, 28) -> 输出: (N, 10, 24, 24)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        # 第二个卷积层: 10 -> 20 通道, 5x5 卷积核
        # 输入: (N, 10, 12, 12) -> 输出: (N, 20, 8, 8)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # 2x2 最大池化层
        self.pool = nn.MaxPool2d(2)

        # Dropout 层，丢弃率 0.25
        self.dropout = nn.Dropout(0.25)

        # 全连接层
        # 展平后: 20 * 4 * 4 = 320
        self.fc1 = nn.Linear(20 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)

        # Log Softmax 输出层
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入图像张量，形状为 (N, 1, 28, 28)。

        Returns:
            形状为 (N, 10) 的 log 概率张量。
        """
        # 第一个卷积块: Conv -> Pool
        x = self.conv1(x)          # (N, 1, 28, 28) -> (N, 10, 24, 24)
        x = self.pool(x)           # (N, 10, 24, 24) -> (N, 10, 12, 12)

        # 第二个卷积块: Conv -> Pool
        x = self.conv2(x)          # (N, 10, 12, 12) -> (N, 20, 8, 8)
        x = self.pool(x)           # (N, 20, 8, 8) -> (N, 20, 4, 4)

        # Dropout
        x = self.dropout(x)

        # 展平
        x = x.view(-1, 20 * 4 * 4)  # (N, 20, 4, 4) -> (N, 320)

        # 全连接层
        x = self.fc1(x)            # (N, 320) -> (N, 100)
        x = self.fc2(x)            # (N, 100) -> (N, 10)

        # Log Softmax 输出
        x = self.log_softmax(x)

        return x


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """在给定数据集上评估模型准确率。

    使用 @torch.no_grad() 装饰器禁用梯度计算，节省内存并加速推理。

    Args:
        model: 待评估的 PyTorch 模型。
        loader: 数据加载器，提供评估数据。
        device: 计算设备（CPU 或 CUDA）。

    Returns:
        分类准确率，范围 [0.0, 1.0]。

    Example:
        >>> accuracy = evaluate(model, test_loader, device)
        >>> print(f"测试准确率: {accuracy * 100:.2f}%")
    """
    model.eval()  # 切换到评估模式，禁用 Dropout 和 BatchNorm 的训练行为
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数

    for x, y in loader:
        # 将数据移至指定设备
        x, y = x.to(device), y.to(device)

        # 前向传播获取预测
        logits = model(x)
        pred = logits.argmax(dim=1)  # 取概率最大的类别作为预测结果

        # 统计正确预测数量
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """训练模型一个 epoch。

    遍历整个训练数据集一次，对每个批次执行前向传播、损失计算、
    反向传播和参数更新。

    Args:
        model: 待训练的 PyTorch 模型。
        loader: 训练数据加载器。
        optimizer: 优化器，用于更新模型参数。
        criterion: 损失函数，用于计算预测与真实标签之间的误差。
        device: 计算设备（CPU 或 CUDA）。

    Returns:
        该 epoch 的平均训练损失。

    Note:
        - 使用 optimizer.zero_grad(set_to_none=True) 比 zero_grad() 更高效，
          因为它将梯度设为 None 而非零张量。
        - running_loss 累加时乘以 batch_size，最后除以总样本数得到平均损失。
    """
    model.train()  # 切换到训练模式，启用 Dropout 和 BatchNorm 的训练行为
    running_loss = 0.0  # 累计损失

    for x, y in loader:
        # 将数据移至指定设备
        x, y = x.to(device), y.to(device)

        # 清零梯度，set_to_none=True 比设为零更高效
        optimizer.zero_grad(set_to_none=True)

        # 前向传播
        logits = model(x)

        # 计算损失（CrossEntropyLoss 内部包含 softmax）
        loss = criterion(logits, y)

        # 反向传播计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 累加损失（乘以批次大小以便后续计算平均值）
        running_loss += loss.item() * y.size(0)

    # 返回平均损失
    return running_loss / len(loader.dataset)


def main():
    """主函数：执行完整的模型训练流程。

    流程包括:
        1. 设置超参数和计算设备
        2. 准备 MNIST 数据集和数据加载器
        3. 初始化模型、损失函数和优化器
        4. 执行多轮训练和评估
        5. 保存训练好的模型权重
    """
    # ==================== 超参数配置 ====================
    batch_size = 64  # 训练批次大小
    epochs = 3  # 训练轮数
    lr = 1e-3  # 学习率（Adam 优化器的默认推荐值）
    data_dir = "./data"  # 数据集存储目录

    # ==================== 设备配置 ====================
    # 自动选择 GPU（如果可用）或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==================== 数据预处理 ====================
    # MNIST 数据集的标准归一化参数
    # 均值 0.1307 和标准差 0.3081 是在整个 MNIST 训练集上统计得到的
    # 归一化后数据近似服从标准正态分布，有助于模型训练
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL 图像转换为张量，像素值从 [0,255] 缩放到 [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化：(x - mean) / std
    ])

    # ==================== 加载数据集 ====================
    # download=True 会在数据不存在时自动下载
    train_ds = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # ==================== 创建数据加载器 ====================
    # 训练集：打乱顺序，使用较小的 batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # 打乱数据顺序，增加训练随机性
        num_workers=2,  # 使用 2 个子进程加载数据
        pin_memory=True  # 将数据固定在内存中，加速 GPU 数据传输
    )

    # 测试集：不打乱，使用较大的 batch_size（因为不需要计算梯度，可以用更大批次）
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ==================== 初始化模型和训练组件 ====================
    model = CNNMNIST().to(device)  # 创建模型并移至指定设备

    # NLLLoss 损失函数：配合 LogSoftmax 使用
    criterion = nn.NLLLoss()

    # Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ==================== 训练循环 ====================
    for epoch in range(1, epochs + 1):
        # 训练一个 epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # 在测试集上评估
        test_acc = evaluate(model, test_loader, device)

        # 打印训练进度
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"test_acc={test_acc*100:.2f}%"
        )

    # ==================== 保存模型 ====================
    os.makedirs("checkpoints", exist_ok=True)  # 创建检查点目录（如果不存在）
    ckpt_path = "checkpoints/cnn_mnist.pth"
    torch.save(model.state_dict(), ckpt_path)  # 只保存模型参数，不保存整个模型
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
