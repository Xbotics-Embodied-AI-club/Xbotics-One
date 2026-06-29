# experiments/ — 演示代码 + 统一 uv 环境

Xbotics 教学环境的**可跑演示代码树**：VLA（视觉-语言-动作）课程的 notebook / demo 脚本，以及配套的 RDK 板端部署工具。所有课程共用 `experiments/` 根的**一套 uv 管理 Python 环境**。

## 目录

```
experiments/
├── pyproject.toml / uv.lock      # 统一环境（按 extra 切换：vla_train / tele / rdks600_act …）
├── vla/                          # VLA 课程演示代码（按模块号组织）
│   ├── 2_1_pytorch  2_3_GPT2     # 基础：PyTorch / GPT2
│   ├── 3_1_robotics 3_3_lerobot  # 机器人学 / LeRobot / LIBERO / SO-101
│   ├── 4_2_lerobot_dataset 4_3_tele_so101
│   ├── 5_2_ACT  6_3_pi0  6_7_vla0_exp
└── lerobot/                      # lerobot 本地补丁 + RDK 板端部署（见 lerobot/rdk/README.md）
```

> 演示代码面向课堂：常量就近内联、自上而下按讲解顺序读，不用命令行参数层。

## 一、环境（uv）

改 `pyproject.toml` 后用 `uv sync`，不要 `pip install`。各场景对应一个互斥 extra，例如：

| extra | 用途 | torch | lerobot |
| --- | --- | --- | --- |
| `vla_train` | VLA GPU 训练 / 演示 | cu128（GPU，CUDA 12.8） | `lerobot[all]` |
| `tele` | SO-101 数据采集（无卡 x86 数采机） | CPU | `lerobot[feetech]` |
| `rdks600_act` | RDK S600/S100 ACT BPU 导出 | GPU | `lerobot[feetech]` |

```bash
cd experiments
bash lerobot/fetch_lerobot.sh        # 拉取 lerobot 0.5.1 源并打补丁（不入库）
uv sync --extra vla_train
```

### lerobot（本地补丁，不分发源树）

lerobot 不提交源码：`lerobot/fetch_lerobot.sh` 从 git 拉取 v0.5.1 到 `lerobot/lerobot/`（gitignore）并打 `0001`/`0002` 补丁，再以 editable 安装。补丁内容见 `lerobot/*.patch`。

## 二、环境变量

代码直接读取环境变量、不设默认值。复制模板并填值：

```bash
cp .env.example .env          # 填 HF_TOKEN / WANDB_API_KEY / DATASETS_ROOT
cp .envrc.example .envrc && direnv allow   # 由 DATASETS_ROOT 派生 HF_HOME / HF_LEROBOT_HOME
```

- `DATASETS_ROOT` —— 数据集 / 模型产物根
- `HF_HOME = $DATASETS_ROOT/hf-hub`、`HF_LEROBOT_HOME = $HF_HOME/lerobot`
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` / `WANDB_API_KEY`

## 三、机器约定

- **x86 训练机**：带 GPU，CUDA 锁定 12.8，用于训练 / 导出 / 编译。
- **x86 数采机**：无显卡，用于 SO-101 遥操作数据采集。
- **RDK S100 / S600**：地瓜开发板，板端 BPU 推理；ACT 上板流程见 `lerobot/rdk/`。
