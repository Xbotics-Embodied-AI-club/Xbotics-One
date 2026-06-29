# Xbotics-One

Xbotics 具身智能社区的公开发布整合仓库：把课程演示代码、参考资料与硬件设计收在一处，方便学员对照学习、动手复现。

## 结构

```
Xbotics-One/
├── reference/      # 参考资料（学习路线、论文/资源索引）
├── hardware/       # 硬件设计（外壳/底座 3D 模型、BOM）
└── experiments/    # 演示代码 + 统一 uv 环境
    ├── vla/        # VLA（视觉-语言-动作）课程演示
    └── lerobot/    # lerobot 本地补丁 + RDK 板端部署
```

- **experiments/** —— 可跑的课堂演示代码（VLA），共用一套 uv 环境；上手见 [`experiments/README.md`](experiments/README.md)。
- **hardware/** —— 配套硬件（如 RDK S100 外壳）的 SolidWorks 模型与物料清单。
- **reference/** —— 学习路线与外部资源索引（持续补充）。

## 使用

```bash
git clone https://github.com/Xbotics-Embodied-AI-club/Xbotics-One.git
cd Xbotics-One/experiments
cp .env.example .env        # 填入自己的 token 与数据集路径
bash lerobot/fetch_lerobot.sh
uv sync --extra vla_train
```

## 说明

本仓库是面向公开分发的整理版：已移除内部密钥、个人路径、内网 IP 与机器型号等私有信息。真实 token / 数据集路径请按 `experiments/.env.example` 自行配置。

由 [Xbotics 具身智能社区](https://github.com/Xbotics-Embodied-AI-club) 维护。
