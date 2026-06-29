# LeRobot v0.5.0 源码修复记录

## `backbone_cfg`问题
###  错误内容

导入 `lerobot.envs.factory` 时报错：

```
TypeError: non-default argument 'backbone_cfg' follows default argument
```

完整调用链：
```
lerobot.envs.factory
  → lerobot.policies.__init__
    → lerobot.policies.groot.modeling_groot
      → lerobot.policies.groot.groot_n1  ← 报错位置 (line 176)
```

### 原因

`groot_n1.py` 中 `GR00TN15Config` 同时继承了 `PretrainedConfig` 并使用了 `@dataclass` 装饰器。`PretrainedConfig` 自身已具备 dataclass 行为，多余的 `@dataclass` 导致 Python 3.12 对字段顺序做严格检查——`backbone_cfg` 等无默认值字段排在有默认值的 `compute_dtype` 前面，违反了 dataclass 规则。

此 bug 存在于 v0.5.0 release（2026-03-09），上游已在 2026-03-27 通过 PR #3231（commit `07502868`）修复，但未包含在 v0.5.0 tag 中。

### 修改方式

文件：`src/lerobot/policies/groot/groot_n1.py`

1. 移除 `@dataclass` 装饰器（第 176 行）
2. import 中移除 `dataclass`：`from dataclasses import dataclass, field` → `from dataclasses import field`

参考上游 commit：https://github.com/huggingface/lerobot/commit/07502868

---

## torchcodec 版本不兼容

### 错误内容

调用 `dataset[0]` 时报错：

```
RuntimeError: Could not load libtorchcodec.
OSError: libtorchcodec_core6.so: undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev
```

或者 torchcodec 0.8.0 时直接 crash：

```
terminate called after throwing an instance of 'std::bad_alloc'
```

### 原因

lerobot v0.5.0 的 pyproject.toml 中 torchcodec 版本范围为 `>=0.2.1,<0.11.0`，过于宽泛。uv 自动安装了 torchcodec 0.10.0，但本环境 PyTorch 为 2.8.0+cu128。

torchcodec 与 PyTorch 的版本对应关系（每个版本只兼容特定的 PyTorch）：

| torchcodec | torch |
|---|---|
| 0.7 | 2.8 |
| 0.8 | 2.9 |
| 0.9 | 2.10 (nightly) |
| 0.10 | 2.10 |
| 0.11 | 2.11 |

注：`lerobot-dataset-viz` 命令不受影响，因为它硬编码了 `video_backend="pyav"` 绕过了 torchcodec。

### 修改方式

文件：`pyproject.toml`

在 dependencies 中添加 `"torchcodec==0.7.0"` 锁定版本，然后重新 `uv lock && uv sync`。

参考：https://github.com/meta-pytorch/torchcodec/issues/995
