#!/usr/bin/env bash
# 拉取 lerobot v0.5.1 源码到 ./lerobot（与本脚本同目录的 lerobot/ 子目录），并按序打上本地补丁。
#
# 三种部署都先跑这一步：
#   素材机/训练机本地：  cd experiments && bash lerobot/fetch_lerobot.sh
#   Docker 构建：        Dockerfile 内 RUN bash lerobot/fetch_lerobot.sh
#
# 设计：
#   - clone 落点 = experiments/lerobot/lerobot/（外层仓库 gitignore，不分发源树）。
#   - 保留 lerobot 自带 .git（可见 tag / 可重拉；补丁以工作区未提交改动形式存在）。
#   - 幂等：源已存在则跳过 clone；补丁已打则跳过 apply。可反复运行。
# 补丁清单（按文件名顺序 apply）：
#   - 0001-groot-remove-dataclass.patch：GR00TN15Config 去掉 @dataclass（HF PretrainedConfig
#     子类套 @dataclass 会崩；该 bug 在上游 v0.5.0 与 v0.5.1 间未修，必须本地打补丁）。
#   - 0002-vla0-smol-policy.patch：加入 VLA-0(vla0_smol) 原生 policy（SmolVLM2-500M、整数串动作、
#     xgrammar 约束解码）+ factory 注册。用户授权为加必要新功能改 lerobot 源（EAI-exp-001）。
set -euo pipefail

LEROBOT_REF="${LEROBOT_REF:-v0.5.1}"
LEROBOT_URL="${LEROBOT_URL:-https://github.com/huggingface/lerobot.git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$SCRIPT_DIR/lerobot}"
PATCHES=(
  "$SCRIPT_DIR/0001-groot-remove-dataclass.patch"
  "$SCRIPT_DIR/0002-vla0-smol-policy.patch"
)

for p in "${PATCHES[@]}"; do
  if [ ! -f "$p" ]; then
    echo "[fetch_lerobot] 找不到补丁: $p" >&2
    exit 1
  fi
done

if [ -f "$TARGET/src/lerobot/__init__.py" ]; then
  echo "[fetch_lerobot] $TARGET 已存在，跳过 clone。"
else
  echo "[fetch_lerobot] clone lerobot $LEROBOT_REF -> $TARGET"
  git clone --depth 1 --branch "$LEROBOT_REF" "$LEROBOT_URL" "$TARGET"
fi

cd "$TARGET"
for p in "${PATCHES[@]}"; do
  name="$(basename "$p")"
  if git apply --reverse --check "$p" >/dev/null 2>&1; then
    echo "[fetch_lerobot] 补丁已应用，跳过：$name"
  else
    echo "[fetch_lerobot] 应用补丁：$name"
    git apply "$p"
  fi
done

echo "[fetch_lerobot] 完成：$TARGET （lerobot $LEROBOT_REF + groot 修复 + vla0_smol policy）"
