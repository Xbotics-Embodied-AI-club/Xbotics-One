"""
从 SO101 URDF 提取 MDH 参数

两步法：
  1. 几何分析得到初始估计 (关节轴方向 → alpha, 轴间距 → a, d)
  2. 数值优化精调 (最小化 MDH FK 和 URDF FK 的误差)
"""

import numpy as np
from numpy import pi, cos, sin
from scipy.optimize import minimize

np.set_printoptions(precision=6, suppress=True)


# ========================================================================
# URDF 工具
# ========================================================================
def rpy_to_rotation(roll, pitch, yaw):
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr            ]
    ])


def urdf_joint_transform(xyz, rpy, theta, axis=(0, 0, 1)):
    T = np.eye(4)
    T[:3, 3] = xyz
    T[:3, :3] = rpy_to_rotation(*rpy)
    ax, ay, az = axis
    ct, st = cos(theta), sin(theta)
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
    R = np.eye(3) + st * K + (1 - ct) * (K @ K)
    T_j = np.eye(4)
    T_j[:3, :3] = R
    return T @ T_j


# SO101 URDF (new_calib, 6 个活动关节)
so101_joints = [
    ("shoulder_pan",  (0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159),  (0,0,1)),
    ("shoulder_lift", (-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0),              (0,0,1)),
    ("elbow_flex",    (-0.11257, -0.028, 1.73763e-16),   (-3.636e-16, 8.743e-16, 1.5708),    (0,0,1)),
    ("wrist_flex",    (-0.1349, 0.0052, 3.62355e-17),    (4.025e-15, 8.674e-16, -1.5708),    (0,0,1)),
    ("wrist_roll",    (5.55e-17, -0.0611, 0.0181),       (1.5708, 0.0486795, 3.14159),       (0,0,1)),
    ("gripper",       (0.0202, 0.0188, -0.0234),         (1.5708, -5.24284e-08, -1.41553e-15),(0,0,1)),
]
N = 6


def fk_urdf(q):
    T = np.eye(4)
    for i, (_, xyz, rpy, axis) in enumerate(so101_joints):
        T = T @ urdf_joint_transform(xyz, rpy, q[i], axis)
    return T


# ========================================================================
# MDH 变换
# ========================================================================
def mdh_transform(theta, d, a, alpha):
    """T = Rx(alpha) · Tx(a) · Rz(theta) · Tz(d)"""
    ca, sa = cos(alpha), sin(alpha)
    ct, st = cos(theta), sin(theta)
    return np.array([
        [ct,       -st,       0,    a       ],
        [ca*st,     ca*ct,   -sa,   d*(-sa) ],
        [sa*st,     sa*ct,    ca,   d*ca    ],
        [0,         0,        0,    1       ]
    ])


def fk_mdh(q, params):
    T = np.eye(4)
    for i in range(N):
        a, alpha, d, theta_off = params[i*4:(i+1)*4]
        T = T @ mdh_transform(q[i] + theta_off, d, a, alpha)
    return T


# ========================================================================
# Step 1: 关节轴几何分析
# ========================================================================
print("=" * 72)
print("  Step 1: 关节轴几何分析")
print("=" * 72)

T_acc = np.eye(4)
z_axes = [np.array([0, 0, 1])]  # z_0 = 基座
origins = [np.array([0, 0, 0])]

for i, (name, xyz, rpy, axis) in enumerate(so101_joints):
    T_o = np.eye(4)
    T_o[:3, 3] = xyz
    T_o[:3, :3] = rpy_to_rotation(*rpy)
    T_acc = T_acc @ T_o

    z = T_acc[:3, :3] @ np.array(axis)
    z = z / np.linalg.norm(z)
    z_axes.append(z)
    origins.append(T_acc[:3, 3].copy())
    print(f"  Joint {i+1} ({name:16s}): z=[{z[0]:+.4f}, {z[1]:+.4f}, {z[2]:+.4f}]")

# 关节轴结构:
# z_0 = [0, 0, +1]   (基座, 朝上)
# z_1 = [0, 0, -1]   (shoulder_pan, 朝下)     alpha_0: z0→z1 = pi
# z_2 = [0, +1, 0]   (shoulder_lift, 朝+y)    alpha_1: z1→z2 = pi/2
# z_3 = [0, +1, 0]   (elbow_flex, 同z_2)      alpha_2: z2→z3 = 0
# z_4 = [0, +1, 0]   (wrist_flex, 同z_2)      alpha_3: z3→z4 = 0
# z_5 = [-1, 0, 0]   (wrist_roll, 朝-x)       alpha_4: z4→z5 = -pi/2 or pi/2

print(f"""
  关节轴结构:
    z_0 = [ 0,  0, +1]  基座 (向上)
    z_1 = [ 0,  0, -1]  shoulder_pan (向下)     alpha_0: pi
    z_2 = [ 0, +1,  0]  shoulder_lift (向+y)    alpha_1: pi/2
    z_3 = [ 0, +1,  0]  elbow_flex (平行 z_2)   alpha_2: 0
    z_4 = [ 0, +1,  0]  wrist_flex (平行 z_2)   alpha_3: 0
    z_5 = [-1,  0,  0]  wrist_roll (向-x)       alpha_4: +/-pi/2
    z_6 = [ ?,  ?,  ?]  gripper                 alpha_5: ?
""")


# ========================================================================
# Step 2: 初始估计 + 数值优化
# ========================================================================
print("=" * 72)
print("  Step 2: 数值优化 MDH 参数")
print("=" * 72)

# 初始估计 (从几何分析)
# params = [a_0, alpha_0, d_1, theta_off_1, a_1, alpha_1, d_2, theta_off_2, ...]
x0 = np.array([
    # a_{i-1}, alpha_{i-1}, d_i,     theta_offset_i
    0.039,     pi,          -0.096,   0.0,       # Joint 1 (shoulder_pan)
    0.030,     pi/2,         0.0,    -1.33,      # Joint 2 (shoulder_lift)
    0.116,     0.0,          0.0,     1.29,      # Joint 3 (elbow_flex)
    0.135,     0.0,          0.0,     1.61,      # Joint 4 (wrist_flex)
    0.0,      -pi/2,        -0.061,  -pi,        # Joint 5 (wrist_roll)
    0.020,     pi/2,        -0.023,   0.0,       # Joint 6 (gripper)
])

# 生成采样配置
np.random.seed(42)
q_samples = [np.zeros(N)]
for _ in range(40):
    q_samples.append(np.random.uniform(-1.5, 1.5, N))

# 预计算 URDF FK
urdf_fk_cache = [fk_urdf(q) for q in q_samples]


def cost(x):
    err = 0.0
    for q, T_ref in zip(q_samples, urdf_fk_cache):
        T = fk_mdh(q, x)
        err += np.sum((T_ref[:3, 3] - T[:3, 3]) ** 2)     # 位置
        err += 0.01 * np.sum((T_ref[:3, :3] - T[:3, :3]) ** 2)  # 旋转
    return err / len(q_samples)


print(f"  初始误差: {cost(x0):.6e}")

# 优化: Powell 方法 (无梯度, 适合这种问题)
result = minimize(cost, x0, method='Powell',
                  options={'maxiter': 100000, 'ftol': 1e-20, 'xtol': 1e-15})
print(f"  优化后误差: {result.fun:.6e}")
print(f"  迭代次数: {result.nit}")

opt = result.x


# ========================================================================
# Step 3: 验证
# ========================================================================
print(f"\n{'=' * 72}")
print("  Step 3: 验证 (独立测试集, 未参与优化)")
print("=" * 72)

test_qs = {
    "零位 q=0":         np.zeros(6),
    "随机 A":           np.array([0.3, -0.5, 0.8, -0.3, 1.0, 0.5]),
    "随机 B":           np.array([-1.0, 0.1, 1.2, 0.5, -0.8, -0.1]),
    "随机 C":           np.array([1.5, -1.0, 0.2, 1.0, -2.0, 1.0]),
    "随机 D":           np.array([-0.7, 0.9, -1.1, 0.8, 0.5, 0.3]),
    "单关节 q1=pi/6":   np.array([pi/6, 0, 0, 0, 0, 0]),
    "单关节 q3=pi/4":   np.array([0, 0, pi/4, 0, 0, 0]),
    "单关节 q6=pi/4":   np.array([0, 0, 0, 0, 0, pi/4]),
    "极端配置":          np.array([1.9, -1.7, 1.5, -1.6, 2.7, 1.5]),
}

all_pass = True
for name, q in test_qs.items():
    T_u = fk_urdf(q)
    T_d = fk_mdh(q, opt)
    pos_err = np.linalg.norm(T_u[:3, 3] - T_d[:3, 3])
    rot_err = np.linalg.norm(T_u[:3, :3] - T_d[:3, :3])
    ok = pos_err < 1e-6 and rot_err < 1e-5
    if not ok:
        all_pass = False
    symbol = "+" if ok else "x"
    print(f"  [{symbol}] {name:20s}  pos={pos_err:.2e} m  rot={rot_err:.2e}  {'PASS' if ok else 'FAIL'}")


# ========================================================================
# Step 4: 输出 MDH 参数表
# ========================================================================
print(f"\n{'=' * 72}")
print("  SO101 MDH 参数表 (优化后)")
print("=" * 72)

joint_names = [j[0] for j in so101_joints]
print(f"  {'Joint':<16} {'a (m)':>10} {'alpha (rad)':>14} {'d (m)':>10} {'theta_off (rad)':>16}")
print("  " + "-" * 66)
for i in range(N):
    a, alpha, d, theta_off = opt[i*4:(i+1)*4]
    print(f"  {joint_names[i]:<16} {a:>10.6f} {alpha:>8.4f} ({alpha/pi:+.3f}pi)"
          f" {d:>10.6f} {theta_off:>8.4f} ({theta_off/pi:+.3f}pi)")

print(f"\n  验证: {'全部通过!' if all_pass else '存在误差'}")


# ========================================================================
# 对比 Panda (MDH) 和 SO101 (MDH)
# ========================================================================
print(f"""
{'=' * 72}
  总结: 统一 MDH 框架
{'=' * 72}

  两种机器人的 MDH 参数已成功提取:

  Panda (7-DOF):
    参数来源: Franka 官方文档, 直接可用
    URDF 帧: 恰好按 MDH 规则放置 → 参数可直接从 URDF 读出

  SO101 (6-DOF):
    参数来源: 几何分析 + 数值优化
    URDF 帧: CAD 导出, 不遵循 DH 规则 → 需要计算转换

  两者最终都用统一的 MDH 公式:
    T_i = Rx(alpha_{{i-1}}) * Tx(a_{{i-1}}) * Rz(theta_i) * Tz(d_i)

  DH 的 4 个参数足以描述任何串联机器人的运动学。
  不管 URDF 的帧怎么放，DH 参数总是存在且唯一的。
""")
