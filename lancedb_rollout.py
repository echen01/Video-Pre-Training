from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import socket
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import ENV_KWARGS, MineRLAgent
from inverse_dynamics_model import IDMAgent
from lib.gymnasium_compat import reset_env, step_env


_NAME_RE = re.compile(r"[^0-9a-zA-Z_]+")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str
    width: int | None = None

    def arrow_type(self):
        import pyarrow as pa

        if self.kind == "binary":
            return pa.binary()
        if self.kind == "string":
            return pa.string()
        if self.kind == "bool":
            return pa.bool_()
        if self.kind == "int":
            return pa.int64()
        if self.kind == "float":
            return pa.float32()
        if self.kind == "int_list":
            return pa.list_(pa.int64(), self.width)
        if self.kind == "float_list":
            return pa.list_(pa.float32(), self.width)
        raise ValueError(f"Unsupported column kind: {self.kind}")


def _safe_name(name: str) -> str:
    name = _NAME_RE.sub("_", name).strip("_")
    if not name:
        name = "value"
    if name[0].isdigit():
        name = f"v_{name}"
    return name


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _is_image(arr: np.ndarray) -> bool:
    return (
        arr.dtype == np.uint8
        and arr.ndim == 3
        and (arr.shape[-1] in (1, 3, 4) or arr.shape[0] in (1, 3, 4))
    )


def _to_hwc_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _encode_jpeg(arr: np.ndarray, jpeg_quality: int) -> bytes:
    import cv2

    frame = _to_hwc_image(arr)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = frame[..., ::-1]
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise ValueError(f"Could not JPEG-encode frame with shape {arr.shape}")
    return encoded.tobytes()


def write_mp4(path: Path, frames: list[np.ndarray], fps: float, codec: str) -> None:
    import cv2

    if not frames:
        raise ValueError("Cannot write MP4 with no frames")
    if len(codec) != 4:
        raise ValueError("video codec must be a four-character FourCC code")

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Video frame shape changed from {(height, width)} to {frame.shape[:2]}"
                )
            writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
    finally:
        writer.release()


def write_episode_mp4(
    video_dir: Path,
    *,
    collector_id: str,
    episode_idx: int,
    frames: list[np.ndarray],
    fps: float,
    codec: str,
) -> Path:
    video_path = video_dir / f"{collector_id}_episode_{episode_idx:06d}.mp4"
    tmp_path = video_path.with_name(f"{video_path.stem}.partial.mp4")
    if video_path.exists() or tmp_path.exists():
        raise FileExistsError(f"Video already exists or is in progress: {video_path}")

    write_mp4(tmp_path, frames, fps, codec)
    os.replace(tmp_path, video_path)
    return video_path


def _flatten(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _flatten(f"{prefix}_{key}", child)
    else:
        yield _safe_name(prefix), value


def _cell_to_storage(
    name: str,
    value: Any,
    jpeg_quality: int,
    *,
    numeric_as_float: bool = False,
) -> tuple[str, Any, ColumnSpec]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return name, bytes(value), ColumnSpec(name, "binary")

    if isinstance(value, str):
        return name, value, ColumnSpec(name, "string")

    arr = _as_numpy(value)
    if _is_image(arr):
        col = f"{name}_jpeg"
        return col, _encode_jpeg(arr, jpeg_quality), ColumnSpec(col, "binary")

    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, (bool, np.bool_)):
            return name, bool(item), ColumnSpec(name, "bool")
        if isinstance(item, (int, np.integer)):
            if numeric_as_float:
                return name, float(item), ColumnSpec(name, "float")
            return name, int(item), ColumnSpec(name, "int")
        if isinstance(item, (float, np.floating)):
            return name, float(item), ColumnSpec(name, "float")
        if isinstance(item, (str, bytes)):
            return name, str(item), ColumnSpec(name, "string")
        if item is None:
            return name, "", ColumnSpec(name, "string")

    if arr.dtype.kind in "SU":
        return name, json.dumps(arr.tolist()), ColumnSpec(name, "string")
    if arr.dtype.kind == "O":
        return name, json.dumps(arr.tolist(), default=str), ColumnSpec(name, "string")

    flat = arr.reshape(-1)
    if numeric_as_float and flat.dtype.kind in "iu":
        return name, flat.astype(np.float32).tolist(), ColumnSpec(name, "float_list", len(flat))
    if flat.dtype.kind in "biu":
        return name, flat.astype(np.int64).tolist(), ColumnSpec(name, "int_list", len(flat))
    return name, flat.astype(np.float32).tolist(), ColumnSpec(name, "float_list", len(flat))


def _transition_to_storage(transition: Mapping[str, Any], jpeg_quality: int) -> dict[str, tuple[Any, ColumnSpec]]:
    out: dict[str, tuple[Any, ColumnSpec]] = {}
    for root_key in ("obs", "next_obs", "action", "idm_action"):
        if root_key not in transition:
            continue
        numeric_as_float = root_key in ("obs", "next_obs")
        for name, value in _flatten(root_key, transition[root_key]):
            col, stored, spec = _cell_to_storage(
                name,
                value,
                jpeg_quality,
                numeric_as_float=numeric_as_float,
            )
            if col in out:
                raise ValueError(f"Column name collision after sanitizing: {col}")
            out[col] = (stored, spec)

    for name in ("reward", "terminated", "truncated", "policy_idm_cross_entropy"):
        if name not in transition:
            continue
        col, stored, spec = _cell_to_storage(name, transition[name], jpeg_quality)
        out[col] = (stored, spec)

    return out


def _infer_specs(first_episode: list[Mapping[str, Any]], jpeg_quality: int) -> list[ColumnSpec]:
    if not first_episode:
        raise ValueError("Cannot infer Lance schema from an empty episode")
    stored = _transition_to_storage(first_episode[0], jpeg_quality)
    return [spec for _, spec in stored.values()]


def _episode_to_batch(
    episode: list[Mapping[str, Any]],
    episode_idx: int,
    specs: list[ColumnSpec],
    jpeg_quality: int,
):
    import pyarrow as pa

    columns: dict[str, list[Any]] = {spec.name: [] for spec in specs}
    for transition in episode:
        stored = _transition_to_storage(transition, jpeg_quality)
        missing = set(columns) - set(stored)
        extra = set(stored) - set(columns)
        if missing or extra:
            raise ValueError(f"Episode schema changed. Missing={missing}, extra={extra}")
        for spec in specs:
            value, value_spec = stored[spec.name]
            if value_spec != spec:
                raise ValueError(f"Column {spec.name} changed type from {spec} to {value_spec}")
            columns[spec.name].append(value)

    arrays = [
        pa.array(np.full(len(episode), episode_idx, dtype=np.int32), type=pa.int32()),
        pa.array(np.arange(len(episode), dtype=np.int32), type=pa.int32()),
    ]
    fields = [
        pa.field("episode_idx", pa.int32()),
        pa.field("step_idx", pa.int32()),
    ]
    for spec in specs:
        arrow_type = spec.arrow_type()
        arrays.append(pa.array(columns[spec.name], type=arrow_type))
        fields.append(pa.field(spec.name, arrow_type))
    return pa.record_batch(arrays, schema=pa.schema(fields))


def write_lance_shard(
    episodes: list[list[Mapping[str, Any]]],
    out_dir: str | Path,
    *,
    collector_id: str,
    shard_idx: int,
    table_name: str = "transitions",
    jpeg_quality: int = 95,
) -> Path:
    """Write complete episodes to an immutable LanceDB shard.

    The shard is first written to ``*.partial`` and then renamed to
    ``*.lancedb``. A ``*.done.json`` marker is written last so trainers can
    safely ignore in-flight shards.
    """
    import lancedb
    import pyarrow as pa

    if not episodes:
        raise ValueError("No episodes to write")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_id = f"{collector_id}_{shard_idx:06d}"
    partial_dir = out_dir / f"{shard_id}.partial"
    final_dir = out_dir / f"{shard_id}.lancedb"
    done_path = out_dir / f"{shard_id}.done.json"

    if partial_dir.exists() or final_dir.exists() or done_path.exists():
        raise FileExistsError(f"Shard already exists or is in progress: {shard_id}")

    partial_dir.mkdir(parents=True)
    specs = _infer_specs(episodes[0], jpeg_quality)

    first_batch = _episode_to_batch(episodes[0], 0, specs, jpeg_quality)
    schema = first_batch.schema

    def batches_with_first():
        yield first_batch
        for episode_idx, episode in enumerate(episodes[1:], start=1):
            yield _episode_to_batch(episode, episode_idx, specs, jpeg_quality)

    db = lancedb.connect(str(partial_dir))
    reader = pa.RecordBatchReader.from_batches(schema, batches_with_first())
    db.create_table(table_name, data=reader, schema=schema)

    os.replace(partial_dir, final_dir)

    num_steps = sum(len(ep) for ep in episodes)
    metadata = {
        "shard_id": shard_id,
        "uri": str(final_dir),
        "table_name": table_name,
        "num_episodes": len(episodes),
        "num_steps": num_steps,
    }
    tmp_done = done_path.with_suffix(".done.json.tmp")
    tmp_done.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(tmp_done, done_path)
    return done_path


def _safe_id_part(value: str) -> str:
    value = _safe_name(value)
    return value or "unknown"


def default_collector_id() -> str:
    parts = [datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")]
    parts.append(_safe_id_part(socket.gethostname().split(".")[0]))
    for name in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value:
            parts.append(f"{name.lower()}_{_safe_id_part(value)}")
    parts.append(uuid.uuid4().hex[:8])
    return "_".join(parts)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def launch_minerl_env():
    print("---Launching MineRL environment---", flush=True)
    return HumanSurvival(**ENV_KWARGS).make()


def close_minerl_env(env: Any | None, *, context: str) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception as exc:
        print(f"MineRL env close failed during {context}: {exc!r}", flush=True)


def load_expert_policy(
    env: Any,
    model_path: str | Path,
    weights_path: str | Path,
    device: str | None,
) -> MineRLAgent:
    model_path = Path(model_path)
    weights_path = Path(weights_path)

    print("---Loading VPT expert policy---", flush=True)
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


def load_idm_model(
    model_path: str | Path,
    weights_path: str | Path,
    device: str | None,
) -> IDMAgent:
    model_path = Path(model_path)
    weights_path = Path(weights_path)

    print("---Loading IDM policy---", flush=True)
    with model_path.open("rb") as f:
        agent_parameters = pickle.load(f)

    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = dict(agent_parameters["model"]["args"]["pi_head_opts"])
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs, device=device)
    agent.load_weights(str(weights_path))
    agent.policy.eval()
    return agent


def _slice_env_action(action: Mapping[str, Any], idx: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, value in action.items():
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[0] == 1:
            current = arr[0, idx]
        elif arr.ndim >= 1 and arr.shape[0] > idx:
            current = arr[idx]
        else:
            raise ValueError(f"Cannot slice IDM action {name!r} with shape {arr.shape} at index {idx}")

        if name == "camera":
            camera = np.asarray(current, dtype=np.float32).reshape(-1)
            if camera.size != 2:
                raise ValueError(f"IDM camera action at index {idx} has shape {np.asarray(current).shape}")
            out[name] = camera.reshape(1, 2)
        else:
            out[name] = np.array([int(np.asarray(current).reshape(-1)[0])], dtype=np.int64)
    return out


def sanitize_env_action(action: Mapping[str, Any], *, disable_inventory_action: bool) -> dict[str, Any]:
    sanitized = dict(action)
    if disable_inventory_action and "inventory" in sanitized:
        sanitized["inventory"] = np.zeros_like(np.asarray(sanitized["inventory"]))
    return sanitized


def summarize_action(action: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in action.items():
        arr = np.asarray(value)
        if key == "camera":
            summary[key] = arr.reshape(-1).astype(float).tolist()
        elif arr.size == 1:
            summary[key] = int(arr.reshape(-1)[0])
        elif np.any(arr):
            summary[key] = arr.reshape(-1).tolist()
    return summary


def _policy_cross_entropy_for_env_action(
    agent: MineRLAgent,
    obs: Mapping[str, Any],
    env_action: Mapping[str, Any],
) -> float:
    agent_input = agent._env_obs_to_agent(obs)
    agent_action = agent._env_action_to_agent(env_action, to_torch=True)
    with torch.no_grad():
        _, agent.hidden_state, result = agent.policy.act(
            agent_input,
            agent._dummy_first,
            agent.hidden_state,
            stochastic=False,
            taken_action=agent_action,
        )
    log_prob = result["log_prob"].detach().cpu().numpy().reshape(-1)[0]
    return -float(log_prob)


def annotate_episode_with_idm(
    episode: list[dict[str, Any]],
    *,
    policy_agent: MineRLAgent,
    idm_agent: IDMAgent,
    idm_batch_frames: int,
) -> None:
    if idm_batch_frames < 1:
        raise ValueError("idm_batch_frames must be at least 1")
    if not episode:
        return

    idm_agent.reset()
    policy_agent.reset()

    for start in range(0, len(episode), idm_batch_frames):
        end = min(start + idm_batch_frames, len(episode))
        frames = np.stack([transition["obs"]["pov"] for transition in episode[start:end]])
        predicted_actions = idm_agent.predict_actions(frames)

        for local_idx, transition in enumerate(episode[start:end]):
            idm_action = _slice_env_action(predicted_actions, local_idx)
            transition["idm_action"] = idm_action
            transition["policy_idm_cross_entropy"] = _policy_cross_entropy_for_env_action(
                policy_agent,
                transition["obs"],
                idm_action,
            )


def collect_vpt_to_lance_shards(
    out_dir: str | Path,
    *,
    model_path: str | Path,
    weights_path: str | Path,
    idm_model_path: str | Path,
    idm_weights_path: str | Path,
    total_episodes: int,
    max_steps: int,
    shard_episodes: int = 32,
    collector_id: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    idm_batch_frames: int = 128,
    jpeg_quality: int = 95,
    table_name: str = "transitions",
    store_next_obs: bool = False,
    save_video: bool = False,
    video_dir: str | Path | None = None,
    video_fps: float = 20.0,
    video_codec: str = "mp4v",
    disable_inventory_action: bool = False,
    render: bool = False,
    log_every: int = 100,
) -> None:
    if total_episodes < 1:
        raise ValueError("total_episodes must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if shard_episodes < 1:
        raise ValueError("shard_episodes must be at least 1")
    if idm_batch_frames < 1:
        raise ValueError("idm_batch_frames must be at least 1")
    if save_video:
        if video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if len(video_codec) != 4:
            raise ValueError("video_codec must be a four-character FourCC code")

    set_seed(seed)
    collector_id = collector_id or default_collector_id()
    video_output_dir = Path(video_dir) if video_dir is not None else Path(out_dir) / "videos"

    shard: list[list[dict[str, Any]]] = []
    shard_idx = 0
    episodes_collected = 0
    consecutive_empty_failures = 0

    env = launch_minerl_env()
    try:
        agent = load_expert_policy(env, model_path, weights_path, device)
        idm_agent = load_idm_model(idm_model_path, idm_weights_path, device)

        while episodes_collected < total_episodes:
            if env is None:
                env = launch_minerl_env()
            episode_seed = None if seed is None else seed + episodes_collected
            obs, _ = reset_env(env, seed=episode_seed)
            agent.reset()

            episode: list[dict[str, Any]] = []
            video_frames = [obs["pov"]] if save_video else []
            total_reward = 0.0
            terminated = False
            truncated = False
            env_error = None
            last_action: dict[str, Any] | None = None
            last_success_info: Mapping[str, Any] | None = None

            for step_idx in range(max_steps):
                with torch.no_grad():
                    action = agent.get_action(obs)
                action = sanitize_env_action(
                    action,
                    disable_inventory_action=disable_inventory_action,
                )
                last_action = action

                next_obs, reward, env_terminated, env_truncated, info = step_env(env, action)
                if isinstance(info, Mapping) and "error" in info:
                    env_error = info["error"]
                    if episode:
                        episode[-1]["terminated"] = False
                        episode[-1]["truncated"] = True
                    print(
                        f"collector={collector_id} episode={episodes_collected} "
                        f"step={step_idx + 1} minerl_step_error={env_error!r}; "
                        "dropped failed step and truncated valid prefix",
                        flush=True,
                    )
                    print(
                        f"collector={collector_id} episode={episodes_collected} "
                        f"last_action={json.dumps(summarize_action(last_action), sort_keys=True)} "
                        "previous_is_gui_open="
                        f"{last_success_info.get('isGuiOpen') if isinstance(last_success_info, Mapping) else None}",
                        flush=True,
                    )
                    break
                last_success_info = info

                max_step_truncated = step_idx + 1 >= max_steps and not env_terminated and not env_truncated
                terminated = bool(env_terminated)
                truncated = bool(env_truncated or max_step_truncated)
                total_reward += float(reward)

                transition: dict[str, Any] = {
                    "obs": {"pov": obs["pov"]},
                    "action": action,
                    "reward": float(reward),
                    "terminated": terminated,
                    "truncated": truncated,
                }
                if store_next_obs:
                    transition["next_obs"] = {"pov": next_obs["pov"]}
                episode.append(transition)
                if save_video:
                    video_frames.append(next_obs["pov"])

                obs = next_obs
                if render:
                    env.render()

                steps = step_idx + 1
                if log_every > 0 and steps % log_every == 0:
                    print(
                        f"collector={collector_id} episode={episodes_collected} "
                        f"steps={steps} total_reward={total_reward:.3f}",
                        flush=True,
                    )

                if terminated or truncated:
                    break

            if env_error is not None:
                close_minerl_env(
                    env,
                    context=f"step error collector={collector_id} episode={episodes_collected}",
                )
                env = None

            if not episode:
                consecutive_empty_failures += 1
                if consecutive_empty_failures >= 3:
                    raise RuntimeError(
                        "MineRL failed before producing any valid transitions "
                        f"{consecutive_empty_failures} times in a row"
                    )
                print(
                    f"collector={collector_id} episode={episodes_collected} "
                    f"skipped empty failed episode env_error={env_error!r}",
                    flush=True,
                )
                continue

            if save_video:
                video_path = write_episode_mp4(
                    video_output_dir,
                    collector_id=str(collector_id),
                    episode_idx=episodes_collected,
                    frames=video_frames,
                    fps=video_fps,
                    codec=video_codec,
                )
                print(f"Wrote video {video_path}", flush=True)

            print(
                f"collector={collector_id} episode={episodes_collected} "
                f"running_idm steps={len(episode)}",
                flush=True,
            )
            annotate_episode_with_idm(
                episode,
                policy_agent=agent,
                idm_agent=idm_agent,
                idm_batch_frames=idm_batch_frames,
            )

            consecutive_empty_failures = 0
            shard.append(episode)
            episodes_collected += 1
            print(
                f"finished episode={episodes_collected}/{total_episodes} "
                f"steps={len(episode)} total_reward={total_reward:.3f}",
                flush=True,
            )

            if len(shard) >= shard_episodes or episodes_collected >= total_episodes:
                done_path = write_lance_shard(
                    shard,
                    out_dir,
                    collector_id=str(collector_id),
                    shard_idx=shard_idx,
                    table_name=table_name,
                    jpeg_quality=jpeg_quality,
                )
                print(f"Wrote {done_path}", flush=True)
                shard = []
                shard_idx += 1
    finally:
        close_minerl_env(env, context="collector shutdown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect VPT expert MineRL rollouts into LanceDB shards.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the VPT .model file.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the matching VPT .weights file.")
    parser.add_argument("--idm-model", type=Path, required=True, help="Path to the IDM .model file.")
    parser.add_argument("--idm-weights", type=Path, required=True, help="Path to the matching IDM .weights file.")
    parser.add_argument("--out-dir", required=True, help="Directory for *.lancedb shards and *.done.json markers.")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=12000, help="Maximum environment steps per collected episode.")
    parser.add_argument("--shard-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--collector-id", default=None)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--idm-batch-frames", type=int, default=128, help="Video frames per IDM forward pass.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--table-name", default="transitions")
    parser.add_argument("--store-next-obs", action="store_true", help="Also store next_obs columns for each transition.")
    parser.add_argument("--save-video", action="store_true", help="Write an MP4 for each collected episode.")
    parser.add_argument("--video-dir", type=Path, default=None, help="Directory for MP4 outputs. Defaults to <out-dir>/videos.")
    parser.add_argument("--video-fps", type=float, default=20.0, help="FPS metadata for --save-video output.")
    parser.add_argument("--video-codec", type=str, default="mp4v", help="FourCC codec for --save-video output.")
    parser.add_argument(
        "--disable-inventory-action",
        action="store_true",
        help="Force inventory=0 before env.step to avoid opening the Minecraft GUI during collection.",
    )
    parser.add_argument("--render", action="store_true", help="Open the MineRL render window while collecting.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress interval in steps. Use 0 to disable.")
    args = parser.parse_args()

    collect_vpt_to_lance_shards(
        out_dir=args.out_dir,
        model_path=args.model,
        weights_path=args.weights,
        idm_model_path=args.idm_model,
        idm_weights_path=args.idm_weights,
        total_episodes=args.episodes,
        max_steps=args.max_steps,
        shard_episodes=args.shard_episodes,
        collector_id=args.collector_id,
        seed=args.seed,
        device=args.device,
        idm_batch_frames=args.idm_batch_frames,
        jpeg_quality=args.jpeg_quality,
        table_name=args.table_name,
        store_next_obs=args.store_next_obs,
        save_video=args.save_video,
        video_dir=args.video_dir,
        video_fps=args.video_fps,
        video_codec=args.video_codec,
        disable_inventory_action=args.disable_inventory_action,
        render=args.render,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
