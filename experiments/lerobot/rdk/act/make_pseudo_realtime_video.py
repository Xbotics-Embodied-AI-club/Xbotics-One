"""把一次录制（frames_top/ frames_wrist/ + system_info.csv）合成伪实时仪表盘视频。

两路画面并排 + 底部叠加：BPU 推理 ms / 利用率(空闲) / 温度、CPU 利用率/温度、控制帧率。
按录制时的真实帧率回放 → 看起来像现场实时。

跑法（板上 cv2 就地合成；REC_DIR 不给则取 recordings/ 最新一次）：
  ./.venv/bin/python make_pseudo_realtime_video.py [REC_DIR]
产物：<REC_DIR>/pseudo_realtime.mp4
"""

import csv
import glob
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(BASE, "recordings", "*")))[-1]
OUT = os.path.join(REC_DIR, "pseudo_realtime.mp4")

rows = list(csv.DictReader(open(os.path.join(REC_DIR, "system_info.csv"))))
motors = [c[len("state_"):] for c in rows[0] if c.startswith("state_")]
mean_fps = np.mean([float(r["fps"]) for r in rows if r["fps"] not in ("", "nan")])
print(f"REC_DIR={REC_DIR}  帧数={len(rows)}  平均fps={mean_fps:.1f}")

CW, CH = 640, 480          # 单路相机
PANEL = 300                # 底部仪表盘高
W, H = CW * 2, CH + PANEL
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), round(mean_fps), (W, H))

F = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREEN, CYAN, YELLOW, GREY = (255, 255, 255), (90, 230, 90), (230, 230, 90), (90, 200, 255), (150, 150, 150)


def fmt(v, suffix="", nd=1):
    try:
        if v in ("", "nan"):
            return "--"
        return f"{float(v):.{nd}f}{suffix}"
    except Exception:
        return "--"


def put(img, text, org, scale=0.7, color=WHITE, th=2):
    cv2.putText(img, text, org, F, scale, color, th, cv2.LINE_AA)


for i, r in enumerate(rows):
    top = cv2.imread(os.path.join(REC_DIR, "frames_top", f"{i:06d}.jpg"))
    wrist = cv2.imread(os.path.join(REC_DIR, "frames_wrist", f"{i:06d}.jpg"))
    if top is None or wrist is None:
        continue
    top = cv2.resize(top, (CW, CH))
    wrist = cv2.resize(wrist, (CW, CH))
    put(top, "TOP (overhead)", (12, 28), 0.7, CYAN)
    put(wrist, "WRIST (gripper)", (12, 28), 0.7, CYAN)
    cams = np.hstack([top, wrist])

    panel = np.zeros((PANEL, W, 3), np.uint8)
    put(panel, "RDK S100 (nash-e)  -  ACT real-robot inference", (24, 46), 0.95, WHITE, 2)

    bpu_ms, bpu_u, bpu_t = r["bpu_inference_ms"], r["bpu_util_pct"], r["bpu_temp_C"]
    idle = "--"
    try:
        if bpu_u not in ("", "nan"):
            idle = f"{100 - float(bpu_u):.0f}%"
    except Exception:
        pass
    put(panel, f"BPU inference: {fmt(bpu_ms,' ms')}", (24, 104), 0.95, GREEN, 2)
    put(panel, f"BPU util: {fmt(bpu_u,'%',0)}   (idle {idle})", (24, 150), 0.85, GREEN)
    put(panel, f"BPU temp: {fmt(bpu_t,' C',0)}", (24, 192), 0.8, GREY)

    put(panel, f"CPU util: {fmt(r['cpu_util_pct'],'%',0)}   temp: {fmt(r['cpu_temp_C'],' C',0)}", (560, 104), 0.85, YELLOW)
    put(panel, f"control loop: {fmt(r['fps'],' fps',0)}", (560, 150), 0.85, YELLOW)
    put(panel, "action chunk 100 steps  ->  1 BPU infer / 3.33 s", (560, 192), 0.7, GREY)

    # 关节条（state 当前位，归一到 [-180,180]）
    x0, y0 = 24, 230
    for j, m in enumerate(motors):
        try:
            sv = float(r[f"state_{m}"])
        except Exception:
            sv = 0.0
        bx = x0 + j * 200
        cv2.putText(panel, m[:9], (bx, y0), F, 0.5, GREY, 1, cv2.LINE_AA)
        w = int(np.clip((sv + 180) / 360, 0, 1) * 150)
        cv2.rectangle(panel, (bx, y0 + 12), (bx + 150, y0 + 28), (60, 60, 60), -1)
        cv2.rectangle(panel, (bx, y0 + 12), (bx + w, y0 + 28), CYAN, -1)
        cv2.putText(panel, f"{sv:.0f}", (bx + 155, y0 + 26), F, 0.45, WHITE, 1, cv2.LINE_AA)

    writer.write(np.vstack([cams, panel]))

writer.release()
print(f"done -> {OUT}")
