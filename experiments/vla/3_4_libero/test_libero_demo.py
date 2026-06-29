import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Cell 3: 环境创建
print("=== Cell 3: 环境创建 ===")
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env

env_cfg = LiberoEnvConfig(
    task="libero_spatial",
    task_ids=[0],
    obs_type="pixels_agent_pos",
    observation_height=256,
    observation_width=256,
)

envs = make_env(env_cfg, n_envs=1)
env  = envs['libero_spatial'][0]
print("环境创建成功")

# Cell 5-6: Observation 结构验证
print("\n=== Cell 5-6: Observation 结构 ===")
obs, info = env.reset()

def print_obs_tree(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}/")
            print_obs_tree(v, prefix + "  ")
        elif isinstance(v, np.ndarray):
            print(f"{prefix}{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{prefix}{k}: type={type(v).__name__}")

print_obs_tree(obs)

# Cell 9: Action Space 验证
print("\n=== Cell 9: Action Space ===")
action_space = env.single_action_space
print(f"Action space: {action_space}")
assert action_space.shape == (7,)
print("Action space 验证通过")

# Cell 11: Step 控制
print("\n=== Cell 11: Step 控制 ===")
obs, _ = env.reset()
eef_before = obs["robot_state"]["eef"]["pos"][0].copy()
print(f"初始 EEF 位置: {eef_before}")

action = np.zeros((1, 7), dtype=np.float32)
action[0, 0] = 1.0
obs2, reward, terminated, truncated, info = env.step(action)
eef_after = obs2["robot_state"]["eef"]["pos"][0].copy()
print(f"Step 后 EEF 位置: {eef_after}")

# Cell 13: 轨迹跟踪
print("\n=== Cell 13: 轨迹跟踪 ===")
env = env.envs[0]
obs, info = env.reset()
dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
for _ in range(20):
    obs, _, _, _, _ = env.step(dummy_action)

positions = [obs["robot_state"]["eef"]["pos"].copy()]
actions_record = []

for _ in range(30):
    a = np.array([1.0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    obs, _, _, _, _ = env.step(a)
    positions.append(obs["robot_state"]["eef"]["pos"].copy())
    actions_record.append(a[:3].copy())

for _ in range(20):
    a = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    obs, _, _, _, _ = env.step(a)
    positions.append(obs["robot_state"]["eef"]["pos"].copy())
    actions_record.append(a[:3].copy())

for _ in range(30):
    a = np.array([-1.0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    obs, _, _, _, _ = env.step(a)
    positions.append(obs["robot_state"]["eef"]["pos"].copy())
    actions_record.append(a[:3].copy())

positions = np.array(positions)
print(f"Total steps: {len(positions) - 1}")
print(f"Start pos: {positions[0]}")
print(f"End pos:   {positions[-1]}")

# Cell 15
env.close()
print("\n环境已关闭，全部通过！")
