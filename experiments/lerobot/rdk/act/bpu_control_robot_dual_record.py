"""S100 实物测试（录制版）。

跑 ACT 实机控制环，同时录下性能视频要的所有数据：
  - 相机：两路逐帧 jpg（不在板上编码视频，后期 ffmpeg 再转）；
  - 系统：system_info.csv 逐 tick 记录
      帧率(fps) / 每帧处理耗时(proc_ms) /
      BPU(推理 ms、利用率 %、温度 °C) / CPU(利用率 %、温度 °C) /
      6 关节当前位 + 模型指令。
  - 可选 rerun（USE_RERUN=1，默认关，板上软渲染卡）。

跑法（板子桌面终端 或 SSH）：
  cd ~/act_s100_bench
  DRY_RUN=1 ./.venv/bin/python bpu_control_robot_dual_record.py   # 干跑(臂不动)，验证
  DRY_RUN=0 ./.venv/bin/python bpu_control_robot_dual_record.py   # 真动+录制(8°步长钳制)
  # 可选：INFERENCE_TIME=60（默认30秒）、USE_RERUN=1
产物在 ~/act_s100_bench/recordings/<时间戳>/。
"""

import csv
import datetime
import glob
import logging
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(BASE, "rdk_tools_s600")
os.environ.setdefault("HF_LEROBOT_HOME", os.path.join(BASE, "hf_lerobot"))
sys.path.insert(0, TOOLS_DIR)

from bpu_control_robot import (
    BPUACTPolicy,
    build_policy_batch,
    detect_cameras_from_model,
    detect_n_action_steps,
    sanity_check_policy,
)
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

# —— 配置 ——
BPU_ACT_PATH = os.path.join(BASE, "bpu_output_fixed")     # 修正版模型（正常物理映射）
ROBOT_PORT = "/dev/ttyFollower"                           # udev 别名（两板通用）
CAM_DEV = {"top": "/dev/top_camera", "wrist": "/dev/wrist_camera"}  # udev 别名：top=俯视(Realtek) wrist=夹爪(Sonix)
FPS = 30
INFERENCE_TIME = int(os.environ.get("INFERENCE_TIME", "30"))   # 秒
MAX_RELATIVE_TARGET = 8.0          # 每步步长钳制(度)；None=不钳制
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"                # 默认干跑(安全)；DRY_RUN=0 才真发指令
USE_RERUN = os.environ.get("USE_RERUN", "0") != "0"           # 默认关(板上软渲染卡)；USE_RERUN=1 才开 rerun

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
log = logging.getLogger("record")

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = os.path.join(BASE, "recordings", STAMP)
os.makedirs(OUTDIR, exist_ok=True)
log.info("输出目录: %s  DRY_RUN=%s  时长=%ds", OUTDIR, DRY_RUN, INFERENCE_TIME)


# —— 系统状态采样（后台线程，1Hz，不拖慢控制环）——
def _read_cpu_times():
    parts = open("/proc/stat").readline().split()[1:]
    vals = list(map(int, parts[:8]))
    idle = vals[3] + vals[4]            # idle + iowait
    return idle, sum(vals)


def _cpu_temp_C():
    temps = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            temps.append(int(open(z).read().strip()) / 1000.0)
        except Exception:
            pass
    return max(temps) if temps else float("nan")


def _hrut_bpu():
    """从 hrut_somstatus 取 (bpu_temp_C, bpu_util_pct)。"""
    bpu_temp = float("nan")
    bpu_util = float("nan")
    try:
        out = subprocess.run(["hrut_somstatus"], capture_output=True, text=True, timeout=2).stdout
        for line in out.splitlines():
            if "pvt_bpu" in line and ":" in line and bpu_temp != bpu_temp:
                bpu_temp = float(line.split(":")[1].split("(")[0])
            if "bpu0" in line and ":" in line:
                try:
                    bpu_util = float(line.split(":")[1].strip().split()[0])
                except Exception:
                    pass
    except Exception:
        pass
    return bpu_temp, bpu_util


SYS = {"bpu_temp": float("nan"), "bpu_util": float("nan"),
       "cpu_temp": float("nan"), "cpu_util": float("nan")}
_stop = threading.Event()


def _sampler():
    prev_idle, prev_total = _read_cpu_times()
    while not _stop.is_set():
        _stop.wait(1.0)
        try:
            idle, total = _read_cpu_times()
            d_total = total - prev_total
            cpu_util = 100.0 * (1.0 - (idle - prev_idle) / d_total) if d_total > 0 else float("nan")
            prev_idle, prev_total = idle, total
            bt, bu = _hrut_bpu()
            SYS.update(bpu_temp=bt, bpu_util=bu, cpu_temp=_cpu_temp_C(), cpu_util=cpu_util)
        except Exception:
            pass


# —— rerun（可选，默认关）——
rr = None
if USE_RERUN:
    import rerun as rr
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    rr.init("S100_ACT_realtest", spawn=True)
    rr.save(os.path.join(OUTDIR, "session.rrd"))
else:
    log.info("rerun 已关闭（USE_RERUN=1 才开）；逐帧 jpg + csv 录制照常。")

# —— 模型 + sanity ——
camera_names = detect_cameras_from_model(BPU_ACT_PATH)
n_action_steps = detect_n_action_steps(BPU_ACT_PATH, None)
log.info("model=%s cameras=%s n_action_steps=%d", BPU_ACT_PATH, camera_names, n_action_steps)
policy = BPUACTPolicy(BPU_ACT_PATH, n_action_steps, camera_names)
sanity_check_policy(policy, camera_names)
policy.reset()

# —— 相机 + 臂 ——
cameras = {
    name: OpenCVCameraConfig(index_or_path=CAM_DEV[name], width=640, height=480, fps=FPS, warmup_s=10, fourcc="MJPG")
    for name in camera_names
}
robot = SO100Follower(
    SO100FollowerConfig(
        port=ROBOT_PORT, id="so100_follower",
        max_relative_target=(None if DRY_RUN else MAX_RELATIVE_TARGET),
        cameras=cameras,
    )
)
robot.connect()
motor_names = list(robot.bus.motors.keys())
log.info("connected. motors=%s", motor_names)

# —— 逐帧 jpg + 系统 csv ——
framedirs = {name: os.path.join(OUTDIR, f"frames_{name}") for name in camera_names}
for d in framedirs.values():
    os.makedirs(d, exist_ok=True)
sysf = open(os.path.join(OUTDIR, "system_info.csv"), "w", newline="")
sysw = csv.writer(sysf)
sysw.writerow(["tick", "wall_s", "fps", "proc_ms",
               "bpu_inference_ms", "bpu_util_pct", "bpu_temp_C",
               "cpu_util_pct", "cpu_temp_C"]
              + [f"state_{m}" for m in motor_names] + [f"cmd_{m}" for m in motor_names])

_sampler_thread = threading.Thread(target=_sampler, daemon=True)
_sampler_thread.start()

last_inference_ms = float("nan")
t_start = time.perf_counter()
t_prev = t_start
try:
    for tick in range(INFERENCE_TIME * FPS):
        t0 = time.perf_counter()
        obs = robot.get_observation()

        prev_cnt = policy._inference_count
        batch = build_policy_batch(obs, policy, motor_names)
        action_values = policy.select_action(batch)
        if policy._inference_count > prev_cnt:        # 这一步发生了 BPU 推理
            last_inference_ms = (time.perf_counter() - t0) * 1000.0

        cur = [float(obs[f"{m}.pos"]) for m in motor_names]
        cmd = [float(action_values[i].item()) for i in range(len(motor_names))]

        # 存原始帧 jpg（RGB->BGR）
        for name in camera_names:
            fr = obs[name]
            fr = fr.numpy() if hasattr(fr, "numpy") else np.asarray(fr)
            cv2.imwrite(os.path.join(framedirs[name], f"{tick:06d}.jpg"),
                        cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_RGB2BGR))

        if rr is not None:
            try:
                rr.set_time("tick", sequence=tick)
                for name in camera_names:
                    fr = obs[name]
                    fr = fr.numpy() if hasattr(fr, "numpy") else np.asarray(fr)
                    rr.log(f"camera/{name}", rr.Image(fr.astype(np.uint8)))
                for i, m in enumerate(motor_names):
                    rr.log(f"arm_state/{m}", rr.Scalars(cur[i]))
                    rr.log(f"action_cmd/{m}", rr.Scalars(cmd[i]))
                if np.isfinite(last_inference_ms):
                    rr.log("bpu/inference_ms", rr.Scalars(last_inference_ms))
                rr.log("bpu/util_pct", rr.Scalars(SYS["bpu_util"]))
                rr.log("cpu/util_pct", rr.Scalars(SYS["cpu_util"]))
            except Exception as e:
                if tick == 0:
                    log.warning("rerun log failed (继续控制): %s", e)

        now = time.perf_counter()
        proc_ms = (now - t0) * 1000.0
        fps = 1.0 / (now - t_prev) if now > t_prev else float("nan")
        t_prev = now
        sysw.writerow([tick, round(now - t_start, 3), round(fps, 2), round(proc_ms, 2),
                       round(last_inference_ms, 2), round(SYS["bpu_util"], 1), round(SYS["bpu_temp"], 1),
                       round(SYS["cpu_util"], 1), round(SYS["cpu_temp"], 1)]
                      + [round(v, 2) for v in cur] + [round(v, 2) for v in cmd])
        sysf.flush()    # 每行立即落盘，Ctrl+C/异常都不丢数据

        if tick == 0 or (tick + 1) % FPS == 0:
            log.info("tick=%d fps=%.1f proc=%.1fms | BPU %.1fms util=%.0f%% %.0f°C | CPU util=%.0f%% %.0f°C",
                     tick, fps, proc_ms, last_inference_ms, SYS["bpu_util"], SYS["bpu_temp"],
                     SYS["cpu_util"], SYS["cpu_temp"])

        if not DRY_RUN:
            robot.send_action({f"{m}.pos": action_values[i].item() for i, m in enumerate(motor_names)})
        time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))
finally:
    _stop.set()
    try:
        robot.disconnect()
    except Exception as e:
        log.warning("disconnect 出错(忽略): %s", e)
    sysf.close()
    log.info("done. 产物: %s （frames_top/ frames_wrist/ 逐帧 jpg + system_info.csv）", OUTDIR)
    log.info("后期转视频(开发机): ffmpeg -framerate %d -i frames_top/%%06d.jpg -c:v libx264 -pix_fmt yuv420p cam_top.mp4", FPS)
