"""把一次录制拆成一组时间对齐的视频（开发机跑，cv2+ffmpeg）。

  1) top.mp4            —— 俯视相机（640x480）
  2) wrist.mp4          —— 夹爪相机（640x480）
  3) metrics/ 一系列小视频（480x270，黑底便于叠加抠图）：
       每个指标(BPU推理ms / BPU空闲% / BPU利用% / CPU% / FPS / BPU温 / CPU温)各出小视频，
       重要指标再给几种风格：number(大数字) / spark(迷你曲线) / gauge(仪表条)。

全部从 tick 0 起、同样帧数、同样 fps → 逐帧对齐，丢同一时间线即同步。

跑法：
  experiments/.venv/bin/python experiments/lerobot/rdk/act/make_separate_videos.py <REC_DIR>
产物在 <REC_DIR>/videos/ 与 <REC_DIR>/videos/metrics/。
"""

import csv
import os
import subprocess
import sys

import cv2
import numpy as np

REC_DIR = sys.argv[1]
OUTDIR = os.path.join(REC_DIR, "videos")
MDIR = os.path.join(OUTDIR, "metrics")
os.makedirs(MDIR, exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(REC_DIR, "system_info.csv"))))
N = len(rows)
FPS = max(1, round(np.mean([float(r["fps"]) for r in rows if r["fps"] not in ("", "nan")])))
print(f"REC_DIR={REC_DIR}  帧数={N}  fps={FPS}")

FONT, DUP = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX
GREEN, CYAN, YELLOW, GREY, WHITE = (120, 240, 120), (255, 220, 120), (120, 230, 255), (140, 140, 150), (255, 255, 255)
ORANGE = (90, 170, 255)


def series(col):
    out, last = [], 0.0
    for r in rows:
        try:
            f = float(r[col])
            if f == f:
                last = f
        except Exception:
            pass
        out.append(last)
    return np.array(out, np.float32)


BPU_MS, BPU_U, BPU_T = series("bpu_inference_ms"), series("bpu_util_pct"), series("bpu_temp_C")
CPU_U, CPU_T, FPS_S = series("cpu_util_pct"), series("cpu_temp_C"), series("fps")


def T(img, s, org, sc, col, th=2, font=FONT):
    cv2.putText(img, s, org, font, sc, col, th, cv2.LINE_AA)


def ffmpeg_jpg(pattern, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern,
                    "-frames:v", str(N), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], check=True)


def ffmpeg_pipe(out, w, h):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-framerate", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "18", out], stdin=subprocess.PIPE)


# —— 1 & 2 相机 ——
ffmpeg_jpg(os.path.join(REC_DIR, "frames_top", "%06d.jpg"), os.path.join(OUTDIR, "top.mp4"))
ffmpeg_jpg(os.path.join(REC_DIR, "frames_wrist", "%06d.jpg"), os.path.join(OUTDIR, "wrist.mp4"))
print("cameras done: top.mp4 / wrist.mp4")

# —— 3 指标小视频 ——
SW, SH = 480, 270
# (key, label, series, unit, decimals, vmax, color, styles)
_CURVE = ("number", "spark", "gauge", "winscroll", "winhist")
METRICS = [
    ("bpu_ms", "BPU INFERENCE", BPU_MS, " ms", 1, 50, GREEN, _CURVE),
    ("bpu_idle", "BPU IDLE", 100 - BPU_U, " %", 0, 100, GREEN, _CURVE),
    ("bpu_util", "BPU UTIL", BPU_U, " %", 0, 100, GREEN, _CURVE),
    ("cpu_util", "CPU UTIL", CPU_U, " %", 0, 100, YELLOW, _CURVE),
    ("fps", "CONTROL FPS", FPS_S, "", 0, 40, CYAN, _CURVE),
    ("bpu_temp", "BPU TEMP", BPU_T, " C", 0, 80, ORANGE, _CURVE),
    ("cpu_temp", "CPU TEMP", CPU_T, " C", 0, 80, ORANGE, _CURVE),
]


WIN = 150  # 滚动窗口帧数（~5s @30fps），类似任务管理器


def draw_taskmgr(img, s, i, vmax, col, scroll):
    """Windows 任务管理器风格：网格 + 半透明填充区 + 实时线。"""
    x0, y0, w, h = 14, 84, SW - 28, SH - 100
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (18, 22, 20), -1)
    for g in range(1, 5):                       # 横网格
        gy = y0 + h - int(h * g / 5)
        cv2.line(img, (x0, gy), (x0 + w, gy), (44, 52, 48), 1)
    if scroll:                                  # 竖网格随时间滚动
        off = (i * (w // WIN)) % (w // 6 or 1)
        for gx in range(x0 - off, x0 + w, max(1, w // 6)):
            if x0 <= gx <= x0 + w:
                cv2.line(img, (gx, y0), (gx, y0 + h), (40, 48, 44), 1)
        seg = s[max(0, i - WIN + 1): i + 1]
        n = len(seg)
        pts = [(x0 + w - int(w * (n - 1 - k) / (WIN - 1)), y0 + h - int(np.clip(seg[k] / vmax, 0, 1) * h)) for k in range(n)]
    else:                                       # 全程从左铺满
        for gx in range(1, 6):
            xx = x0 + int(w * gx / 6)
            cv2.line(img, (xx, y0), (xx, y0 + h), (40, 48, 44), 1)
        pts = [(x0 + int(w * k / max(1, N - 1)), y0 + h - int(np.clip(s[k] / vmax, 0, 1) * h)) for k in range(i + 1)]
    if len(pts) >= 2:
        poly = np.array(pts + [(pts[-1][0], y0 + h), (pts[0][0], y0 + h)], np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [poly], col)
        cv2.addWeighted(ov, 0.30, img, 0.70, 0, img)
        cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        cv2.circle(img, pts[-1], 4, WHITE, -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 80, 75), 1)


def draw_small(label, s, unit, nd, vmax, col, style, i):
    img = np.zeros((SH, SW, 3), np.uint8)
    val = f"{s[i]:.{nd}f}{unit}"
    T(img, label, (24, 44), 0.75, GREY, 2)
    if style == "number":
        T(img, val, (24, 175), 2.2, col, 4, DUP)
    elif style == "spark":
        T(img, val, (24, 100), 1.3, col, 3, DUP)
        x0, y0, w, h = 24, 130, SW - 48, 110
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (28, 28, 34), -1)
        pts = [(x0 + int(w * k / max(1, N - 1)), y0 + h - int(np.clip(s[k] / vmax, 0, 1) * h)) for k in range(i + 1)]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[-1], 5, col, -1)
    elif style == "gauge":
        T(img, val, (24, 110), 1.5, col, 3, DUP)
        x0, y0, w = 24, 150, SW - 48
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + 46), (45, 45, 55), -1)
        fw = int(np.clip(s[i] / vmax, 0, 1) * w)
        cv2.rectangle(img, (x0, y0), (x0 + fw, y0 + 46), col, -1)
    elif style in ("winscroll", "winhist"):
        T(img, val, (SW - 190, 44), 1.0, col, 2, DUP)
        draw_taskmgr(img, s, i, vmax, col, style == "winscroll")
    return img


count = 0
for key, label, s, unit, nd, vmax, col, styles in METRICS:
    for style in styles:
        out = os.path.join(MDIR, f"{key}_{style}.mp4")
        p = ffmpeg_pipe(out, SW, SH)
        for i in range(N):
            p.stdin.write(draw_small(label, s, unit, nd, vmax, col, style, i).tobytes())
        p.stdin.close()
        p.wait()
        count += 1
        print(f"  metrics/{key}_{style}.mp4")

print(f"\nALL done -> {OUTDIR}")
print(f"相机 2 个 + 指标小视频 {count} 个，全部 {N} 帧 @ {FPS}fps，逐帧对齐。")
