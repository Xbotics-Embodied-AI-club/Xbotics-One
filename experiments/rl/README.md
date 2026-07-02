# RL 实验代码

强化学习课程段的全部演示代码。按**主题组**组织，每个组是一讲的完整材料（代码 +
数据 + 结果），整个组目录打包拿走即可运行与阅读。

## 目录架构

```
rl/
├── 1_rl_basics/            # 组1 RL 基础：同任务三算法对照（REINFORCE→A2C→PPO）
│   ├── 1_1_g1_walk_rl/     #   G1 行走
│   ├── 1_2_video_to_g1_reference/   #   人类视频 → G1 参考动作（支撑工具链）
│   ├── 1_3_g1_motion_tracking/      #   G1 动作跟随（BeyondMimic）
│   ├── data/               #   组内共享数据（按内容命名）
│   └── result/             #   各模块结果（json 摘要 + 演示视频）
└── 2_grpo_posttraining/    # 组2 GRPO 后训练：让模型自我提升
    ├── 2_1_grpo_vlm_counting/       #   GRPO 微调小 VLM 学数数
    ├── 2_2_grpo_vla0_libero/        #   GRPO 后训练 VLA-0 提升成功率
    ├── data/
    └── result/
```

约定：

- **双编号 `<组>_<序>_<名字>`**：组号是主题、组内序号是建立顺序，都只增不改；
  增删实验不会引起既有编号重排。
- 模块内代码引组内数据一律用相对路径（`../data/...`），组目录整体迁移后仍可运行。
- 面向课堂走读的训练/评测入口都配有同名 `.ipynb`（中文分节，与 `.py` 内容一致）。
- 大体积产物（训练 checkpoint、预处理中间量）不入库，统一落
  `DATASETS_ROOT/models/trained/` 下（环境变量由 `.env` 提供）。

## 课程讲次映射

| 组 / 模块 | 内容 | 课程对应 |
|---|---|---|
| `1_rl_basics/1_1_g1_walk_rl` | REINFORCE→A2C→PPO 让 G1 行走 | 讲14 |
| `1_rl_basics/1_2_video_to_g1_reference` | 人类视频 → G1 参考动作工具链 | 讲14（支撑） |
| `1_rl_basics/1_3_g1_motion_tracking` | 同三算法对照 · G1 动作跟随 | 讲14 |
| `2_grpo_posttraining/2_1_grpo_vlm_counting` | GRPO 微调小 VLM 学数数 | 讲15 |
| `2_grpo_posttraining/2_2_grpo_vla0_libero` | GRPO 后训练 VLA-0 提升成功率 | 讲15 |
| `3_hilserl_real/`（待建） | 真机人在环学接触型任务（HIL-SERL） | 讲16 |

> 讲次调整只改这张表，不动目录。

## 环境

全部模块共用 `experiments/pyproject.toml` 的统一 uv 环境，按需选 extra：

| extra | 用途 |
|---|---|
| `rl_train` | 组1：G1 行走 / 动作跟随训练（mjlab + GPU） |
| `rl_vlm_grpo` | 组2：VLM 数数 GRPO（Unsloth + TRL） |
| `vla0_grpo` | 组2：VLA-0 GRPO（lerobot + LIBERO + xgrammar） |

```bash
cd experiments
uv sync --extra rl_train    # 以组1为例
```
