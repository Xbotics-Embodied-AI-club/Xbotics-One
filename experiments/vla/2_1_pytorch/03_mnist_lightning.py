"""基于 PyTorch Lightning 的 MNIST 手写数字识别模型。

将 02_mnist.py 中的纯 PyTorch 实现重构为 Lightning 风格，
自动管理训练循环、设备分配、日志记录等样板代码。
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import lightning as L
from torchmetrics import Accuracy


class CNNMNIST(L.LightningModule):
    """用于 MNIST 分类的卷积神经网络 LightningModule。

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
    """

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

        # 全连接层
        self.fc1 = nn.Linear(20 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # 损失函数和指标
        self.criterion = nn.NLLLoss()
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, 20 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = logits.argmax(dim=1)
        self.test_acc(pred, y)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


class MNISTDataModule(L.LightningDataModule):
    """MNIST 数据模块，封装数据加载逻辑。"""

    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def setup(self, stage=None):
        self.train_ds = datasets.MNIST(
            root=self.data_dir, train=True, transform=self.transform, download=True
        )
        self.test_ds = datasets.MNIST(
            root=self.data_dir, train=False, transform=self.transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=2, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=256,
            shuffle=False, num_workers=2, pin_memory=True,
        )


def main():
    """主函数：使用 Lightning Trainer 执行训练流程。"""
    dm = MNISTDataModule(data_dir="./data", batch_size=64)
    model = CNNMNIST(lr=1e-3)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",  # 自动选择 GPU/CPU
        default_root_dir="checkpoints",
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
