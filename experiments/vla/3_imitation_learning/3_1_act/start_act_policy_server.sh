#!/usr/bin/env bash
# 用 bash 运行这个脚本。
set -euo pipefail
# 任一命令失败就退出。

/opt/miniforge3/envs/vla_class_lerobot/bin/python \
`# 用装了 lerobot 的 Python 启动。` \
  -m lerobot.async_inference.policy_server \
`# 启动 LeRobot 的 async inference policy server。` \
  --host=0.0.0.0 \
`# 监听所有网卡，方便机器人 client 连接。` \
  --port=8080
`# 远程推理服务端口写死为 8080。`
# 这个 server 本身不指定模型，模型会在 client 握手时发给远程 GPU 机加载。
