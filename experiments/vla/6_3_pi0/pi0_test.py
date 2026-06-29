from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.io_utils import write_video
import lerobot.policies  # noqa: F401
from libero.libero import benchmark


# MuJoCo offscreen rendering needs EGL on this machine.
os.environ.setdefault("MUJOCO_GL", "egl")

# Keep pi0 demo/eval startup more predictable for class machines.
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


POLICY_PATH = "lerobot/pi0_libero_finetuned_v044"
DEFAULT_SUITES = ("libero_90", "libero_10")
MAX_STEPS = {
    "libero_10": 520,
    "libero_90": 400,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7
FPS = 10
OUT_PATH = Path("6_3_pi0/output/pi0_white_noise_results.json")
VIDEO_DIR = Path("6_3_pi0/output/pi0_white_noise_videos")
IMAGE_MODES = ("white_noise", "real")


def replace_images_with_white_noise(observation_batch: dict[str, Any]) -> dict[str, Any]:
    """Replace only model image inputs with uniform white noise in [0, 1]."""
    for key, value in observation_batch.items():
        if key.startswith("observation.images.") and torch.is_tensor(value):
            observation_batch[key] = torch.rand_like(value)
    return observation_batch


def apply_image_mode(observation_batch: dict[str, Any], image_mode: str) -> dict[str, Any]:
    if image_mode == "white_noise":
        return replace_images_with_white_noise(observation_batch)
    if image_mode == "real":
        return observation_batch
    raise ValueError(f"Unsupported image_mode: {image_mode}")


def _successes_from_info(info: dict[str, Any], n_envs: int) -> np.ndarray:
    if "final_info" not in info:
        return np.zeros(n_envs, dtype=bool)

    final_info = info["final_info"]
    if not isinstance(final_info, dict) or "is_success" not in final_info:
        return np.zeros(n_envs, dtype=bool)

    successes = final_info["is_success"]
    if torch.is_tensor(successes):
        successes = successes.detach().cpu().numpy()
    else:
        successes = np.asarray(successes)
    return successes.astype(bool).reshape(-1)[:n_envs]


def get_suite_task_ids(suite: str) -> list[int]:
    benchmarks = benchmark.get_benchmark_dict()
    if suite not in benchmarks:
        raise ValueError(f"Unknown LIBERO suite: {suite}")
    task_suite = benchmarks[suite]()
    return list(range(len(task_suite.tasks)))


def render_active_envs(env, frames: list[list[np.ndarray]], done: np.ndarray) -> None:
    for episode_index, inner_env in enumerate(env.envs):
        if not done[episode_index]:
            frames[episode_index].append(inner_env.render())


def get_episode_metadata(env) -> list[dict[str, str]]:
    metadata = []
    for inner_env in env.envs:
        metadata.append(
            {
                "task_name": str(getattr(inner_env, "task", "")),
                "task_description": str(getattr(inner_env, "task_description", "")),
            }
        )
    return metadata


def record_active_actions(
    actions: list[list[list[float]]],
    action: np.ndarray,
    done: np.ndarray,
) -> None:
    for episode_index, episode_action in enumerate(action):
        if not done[episode_index]:
            actions[episode_index].append(episode_action.astype(float).tolist())


def write_episode_artifacts(
    *,
    frames: list[list[np.ndarray]],
    actions: list[list[list[float]]],
    successes: np.ndarray,
    metadata: list[dict[str, str]],
    suite: str,
    task_id: int,
    image_mode: str,
    video_dir: Path,
    fps: int,
) -> list[dict[str, Any]]:
    artifacts = []
    for episode_index, episode_frames in enumerate(frames):
        episode_dir = video_dir / suite / f"task_{task_id:02d}"
        video_path = episode_dir / f"episode_{episode_index:03d}.mp4"
        json_path = episode_dir / f"episode_{episode_index:03d}.json"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(video_path), episode_frames, fps=fps)

        episode_payload = {
            "suite": suite,
            "task_id": task_id,
            "episode_index": episode_index,
            "task_name": metadata[episode_index]["task_name"],
            "task_description": metadata[episode_index]["task_description"],
            "image_mode": image_mode,
            "success": bool(successes[episode_index]),
            "steps": len(actions[episode_index]),
            "action_dim": len(actions[episode_index][0]) if actions[episode_index] else 0,
            "actions": actions[episode_index],
            "video": str(video_path),
        }
        json_path.write_text(json.dumps(episode_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        artifacts.append(
            {
                "episode_index": episode_index,
                "video": str(video_path),
                "json": str(json_path),
                "task_name": episode_payload["task_name"],
                "task_description": episode_payload["task_description"],
                "success": episode_payload["success"],
                "steps": episode_payload["steps"],
            }
        )
    return artifacts


def evaluate_task(
    *,
    policy,
    policy_cfg,
    preprocessor,
    postprocessor,
    suite: str,
    task_id: int,
    episodes: int,
    seed: int,
    image_mode: str,
    video_dir: Path,
    fps: int,
) -> dict[str, Any]:
    env_cfg = LiberoEnvConfig(
        task=suite,
        task_ids=[task_id],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        episode_length=MAX_STEPS.get(suite),
    )
    env = make_env(env_cfg, n_envs=episodes)[suite][task_id]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)

    max_steps = env.call("_max_episode_steps")[0]
    successes = np.zeros(episodes, dtype=bool)
    done = np.zeros(episodes, dtype=bool)
    frames: list[list[np.ndarray]] = [[] for _ in range(episodes)]
    actions: list[list[list[float]]] = [[] for _ in range(episodes)]

    try:
        policy.reset()
        observation, _ = env.reset(seed=[seed + i for i in range(episodes)])
        metadata = get_episode_metadata(env)
        render_active_envs(env, frames, done)

        for step in range(max_steps):
            observation_batch = preprocess_observation(observation)
            observation_batch = apply_image_mode(observation_batch, image_mode)
            observation_batch = add_envs_task(env, observation_batch)
            observation_batch = env_preprocessor(observation_batch)
            observation_batch = preprocessor(observation_batch)

            with torch.inference_mode():
                action = policy.select_action(observation_batch)

            action = postprocessor(action)
            action = env_postprocessor({"action": action})["action"].cpu().numpy()
            record_active_actions(actions, action, done)

            observation, _, terminated, truncated, info = env.step(action)
            successes |= _successes_from_info(info, episodes)
            done |= terminated | truncated
            render_active_envs(env, frames, done)

            if np.all(done):
                break

        artifacts = write_episode_artifacts(
            frames=frames,
            actions=actions,
            successes=successes,
            metadata=metadata,
            suite=suite,
            task_id=task_id,
            image_mode=image_mode,
            video_dir=video_dir,
            fps=fps,
        )

        return {
            "suite": suite,
            "task_id": task_id,
            "episodes": episodes,
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "steps": int(step + 1),
            "artifacts": artifacts,
        }
    finally:
        env.close()


def parse_task_ids(value: str | None) -> list[int] | None:
    if value is None or value.strip().lower() == "all":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pi0 on LIBERO with all image observations replaced by white noise."
    )
    parser.add_argument("--policy-path", default=POLICY_PATH)
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--task-ids", default="all", help="Comma separated ids, or 'all'. Applies to every suite.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of init states per task.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--image-mode", choices=IMAGE_MODES, default="white_noise")
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.device = DEVICE
    policy = get_policy_class(policy_cfg.type).from_pretrained(
        args.policy_path,
        config=policy_cfg,
        strict=False,
    )
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=args.policy_path)

    task_ids = parse_task_ids(args.task_ids)
    all_results: list[dict[str, Any]] = []

    for suite in args.suites:
        suite_task_ids = task_ids
        if suite_task_ids is None:
            suite_task_ids = get_suite_task_ids(suite)

        print(f"\n=== {suite}: {len(suite_task_ids)} tasks, {args.episodes} episodes/task ===")
        for task_id in suite_task_ids:
            result = evaluate_task(
                policy=policy,
                policy_cfg=policy_cfg,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                suite=suite,
                task_id=task_id,
                episodes=args.episodes,
                seed=args.seed,
                image_mode=args.image_mode,
                video_dir=args.video_dir,
                fps=args.fps,
            )
            all_results.append(result)
            print(
                f"{suite} task {task_id:02d}: "
                f"{result['successes']}/{result['episodes']} "
                f"success_rate={result['success_rate']:.3f}"
            )

    summary = {}
    for suite in args.suites:
        suite_results = [r for r in all_results if r["suite"] == suite]
        total_successes = sum(r["successes"] for r in suite_results)
        total_episodes = sum(r["episodes"] for r in suite_results)
        summary[suite] = {
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        }

    payload = {
        "policy_path": args.policy_path,
        "image_input": "uniform_white_noise_[0,1]" if args.image_mode == "white_noise" else "real_camera_images",
        "image_mode": args.image_mode,
        "seed": args.seed,
        "episodes_per_task": args.episodes,
        "summary": summary,
        "results": all_results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    for suite, item in summary.items():
        print(f"{suite}: {item['successes']}/{item['episodes']} success_rate={item['success_rate']:.3f}")
    print(args.out)


if __name__ == "__main__":
    main()
