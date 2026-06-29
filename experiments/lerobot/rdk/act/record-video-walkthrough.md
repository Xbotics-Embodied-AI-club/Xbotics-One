# 录一条 ACT 实机视频 —— 从头到尾对着敲

> 目标：在 RDK 板上跑一次 ACT 实机推理（机械臂动），录下两路相机 + 系统指标，最后在开发机合成成视频。
> 屏幕上全程只出现 `~/...`，不暴露真实目录。下面命令按顺序敲即可。
> 例子用 **S600**（`<BOARD_IP>`）；换 **S100** 只改 IP=`<BOARD_IP>`、板上目录 `act_s600_bench`→`act_s100_bench`。

---

## 0. 隐藏真实路径（开发机，只做一次）

把项目目录软链接到 home，之后所有开发机命令用 `~/exp/...`，屏幕不暴露真实路径：

```bash
ln -sfn <EXPERIMENTS_ROOT> ~/exp
```

---

## 1. 板端录制（机械臂会动）

连板子（输密码 `sunrise` 时不显示在屏幕上）：

```bash
ssh sunrise@<BOARD_IP>
```

进工作目录，先干跑确认（臂不动），再真录（臂动、8°步长钳制、默认 30 秒）：

```bash
cd ~/act_s600_bench
DRY_RUN=1 ./.venv/bin/python bpu_control_robot_dual_record.py
DRY_RUN=0 ./.venv/bin/python bpu_control_robot_dual_record.py
```

- 录久一点：`INFERENCE_TIME=60 DRY_RUN=0 ./.venv/bin/python bpu_control_robot_dual_record.py`
- 随时 `Ctrl-C` 停，数据自动保存。
- 跑完最后一行会打印产物目录，**记住那个时间戳**，例如 `recordings/20260619_213815`。

退出板子：

```bash
exit
```

---

## 2. 把录制拉回开发机

把 `<时间戳>` 换成第 1 步记下的那个：

```bash
mkdir -p ~/exp/.result/rdk/myrec
rsync -az sunrise@<BOARD_IP>:act_s600_bench/recordings/<时间戳>/ ~/exp/.result/rdk/myrec/
```

---

## 3. 合成视频（开发机）

```bash
~/exp/.venv/bin/python ~/exp/lerobot/rdk/act/make_separate_videos.py ~/exp/.result/rdk/myrec
```

产物在 `~/exp/.result/rdk/myrec/videos/`：

- `top.mp4`、`wrist.mp4` —— 两路相机（俯视 / 夹爪）
- `metrics/` —— 35 个指标小视频（BPU 推理ms / BPU空闲 / CPU / FPS / 温度，每个含 number/spark/gauge/winscroll/winhist 五种风格，黑底可叠加）

全部 30fps、逐帧对齐，丢进剪辑软件同一时间线即同步。

---

## 4.（可选）顺手录板子桌面

想要“桌面 + 机械臂”那种实拍感：在 NoMachine 连板子桌面（`<BOARD_IP>:4000`，`sunrise/sunrise`），用 NoMachine 自带录屏录下第 1 步真录的过程，得到 `.nxr`，在 NoMachine Player 里导出成 mp4/webm 即可。

---

### 一页速查

```bash
# 开发机一次
ln -sfn <EXPERIMENTS_ROOT> ~/exp
# 板端录
ssh sunrise@<BOARD_IP>
cd ~/act_s600_bench
DRY_RUN=0 ./.venv/bin/python bpu_control_robot_dual_record.py    # 记下产物时间戳
exit
# 拉回 + 合成（<时间戳> 替换）
mkdir -p ~/exp/.result/rdk/myrec
rsync -az sunrise@<BOARD_IP>:act_s600_bench/recordings/<时间戳>/ ~/exp/.result/rdk/myrec/
~/exp/.venv/bin/python ~/exp/lerobot/rdk/act/make_separate_videos.py ~/exp/.result/rdk/myrec
```
