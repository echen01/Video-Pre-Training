from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch as th
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import ENV_KWARGS, MineRLAgent
from lib.gymnasium_compat import reset_env, step_env


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "unknown"


def default_run_id() -> str:
    parts = [
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        safe_part(socket.gethostname().split(".")[0]),
    ]
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value:
            parts.append(f"{name.lower()}-{safe_part(value)}")
    return "_".join(parts)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def load_agent(env: Any, model_path: Path, weights_path: Path, device: str | None) -> MineRLAgent:
    print("---Loading model---", flush=True)
    with model_path.open("rb") as f:
        agent_parameters = pickle.load(f)

    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = dict(agent_parameters["model"]["args"]["pi_head_opts"])
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    agent = MineRLAgent(
        env,
        device=device,
        policy_kwargs=policy_kwargs,
        pi_head_kwargs=pi_head_kwargs,
    )
    agent.load_weights(str(weights_path))
    agent.policy.eval()
    return agent


def make_video_writer(path: Path, first_frame: np.ndarray, fps: float, codec: str) -> cv2.VideoWriter:
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def write_video_frame(writer: cv2.VideoWriter | None, obs: dict[str, Any]) -> None:
    if writer is None:
        return
    writer.write(np.ascontiguousarray(obs["pov"][:, :, ::-1]))


def collect_episode(
    *,
    env: Any,
    agent: MineRLAgent,
    episode_dir: Path,
    episode_index: int,
    max_steps: int,
    seed: int | None,
    save_video: bool,
    video_fps: float,
    video_codec: str,
    log_every: int,
) -> dict[str, Any]:
    episode_dir.mkdir(parents=True, exist_ok=True)
    agent.reset()
    obs, reset_info = reset_env(env, seed=seed)

    writer = None
    video_path = episode_dir / "pov.mp4"
    if save_video:
        writer = make_video_writer(video_path, obs["pov"], video_fps, video_codec)

    actions_path = episode_dir / "actions.jsonl"
    summary_path = episode_dir / "summary.json"
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    try:
        write_video_frame(writer, obs)
        with actions_path.open("w", encoding="utf-8") as actions_file:
            for step in range(max_steps):
                with th.no_grad():
                    action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = step_env(env, action)
                total_reward += float(reward)
                steps = step + 1

                actions_file.write(
                    json.dumps(
                        {
                            "episode": episode_index,
                            "step": step,
                            "action": jsonable(action),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "info": jsonable(info),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                write_video_frame(writer, obs)

                if log_every > 0 and steps % log_every == 0:
                    print(
                        f"episode={episode_index} steps={steps} total_reward={total_reward:.3f}",
                        flush=True,
                    )

                if terminated or truncated:
                    break
    finally:
        if writer is not None:
            writer.release()

    summary = {
        "episode": episode_index,
        "steps": steps,
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "seed": seed,
        "reset_info": jsonable(reset_info),
        "actions": str(actions_path),
        "video": str(video_path) if save_video else None,
        "video_frame_convention": "frame 0 is the reset observation; action step N maps frame N to frame N+1",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"finished episode={episode_index} steps={steps} total_reward={total_reward:.3f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Collect VPT policy rollouts in MineRL without opening a GUI.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the .model file.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the .weights file.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Directory for rollout output.")
    parser.add_argument("--run-id", type=str, default=None, help="Stable run directory name. Defaults to job metadata.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to collect.")
    parser.add_argument("--max-steps", type=int, default=12000, help="Maximum environment steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed. Episode N uses seed+N.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--save-video", action="store_true", help="Write an MP4 of the pov observation stream.")
    parser.add_argument("--video-fps", type=float, default=20.0, help="FPS metadata for --save-video output.")
    parser.add_argument("--video-codec", type=str, default="mp4v", help="FourCC codec for --save-video.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress interval in steps. Use 0 to disable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if len(args.video_codec) != 4:
        raise ValueError("--video-codec must be a four-character FourCC code")

    set_seed(args.seed)
    run_dir = args.out_dir / (args.run_id or default_run_id())
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": str(args.model),
        "weights": str(args.weights),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "device_arg": args.device,
        "save_video": args.save_video,
        "env_kwargs": jsonable(ENV_KWARGS),
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_PROCID", "SLURM_NODELIST")
            if os.environ.get(key) is not None
        },
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("---Launching MineRL environment---", flush=True)
    env = HumanSurvival(**ENV_KWARGS).make()
    try:
        agent = load_agent(env, args.model, args.weights, args.device)
        summaries = []
        for episode in range(args.episodes):
            episode_seed = None if args.seed is None else args.seed + episode
            summaries.append(
                collect_episode(
                    env=env,
                    agent=agent,
                    episode_dir=run_dir / f"episode_{episode:05d}",
                    episode_index=episode,
                    max_steps=args.max_steps,
                    seed=episode_seed,
                    save_video=args.save_video,
                    video_fps=args.video_fps,
                    video_codec=args.video_codec,
                    log_every=args.log_every,
                )
            )
        (run_dir / "summary.json").write_text(
            json.dumps({"episodes": summaries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
