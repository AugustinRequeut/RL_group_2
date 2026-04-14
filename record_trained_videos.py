import argparse
import json
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import torch
from stable_baselines3 import DQN as SB3DQN

from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TRAINING_CONFIG
from src.dqn import DQN
from src.utils import (
    record_policy_video_from_config,
    record_policy_video_with_overlay_from_config,
)


def _build_dqn_cfg_for_loading() -> dict:
    return {
        "gamma": TRAINING_CONFIG["gamma"],
        "batch_size": TRAINING_CONFIG["batch_size"],
        "buffer_capacity": TRAINING_CONFIG["buffer_capacity"],
        "update_target_every": TRAINING_CONFIG["update_target_every"],
        "epsilon_start": TRAINING_CONFIG["epsilon_start"],
        "decrease_epsilon_factor": TRAINING_CONFIG["decrease_epsilon_factor"],
        "epsilon_min": TRAINING_CONFIG["epsilon_min"],
        "learning_rate": TRAINING_CONFIG["learning_rate"],
        "gradient_clip_norm": TRAINING_CONFIG["gradient_clip_norm"],
        "epsilon_warmup_episodes": 0,
    }


def _infer_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    # Final custom checkpoint: <run_dir>/custom_dqn_qnet.pt
    # Intermediate custom checkpoint: <run_dir>/checkpoints/custom_dqn_qnet_ep_XXXXXX.pt
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _load_custom_model_config(checkpoint_path: Path) -> dict:
    run_dir = _infer_run_dir_from_checkpoint(checkpoint_path)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {"network_type": "flat_mlp", "pooling": "mean"}

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "network_type": metrics.get("custom_network", "flat_mlp"),
        "pooling": metrics.get("pooling", "mean"),
    }


def load_custom_agent(checkpoint_path: Path) -> DQN:
    env = gym.make_vec(
        SHARED_CORE_ENV_ID,
        num_envs=1,
        config=SHARED_CORE_CONFIG,
    )
    action_space = env.single_action_space
    observation_space = env.single_observation_space
    dqn_cfg = _build_dqn_cfg_for_loading()
    dqn_cfg.update(_load_custom_model_config(checkpoint_path))
    agent = DQN(action_space=action_space, observation_space=observation_space, **dqn_cfg)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    agent.q_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    env.close()
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record videos from a trained custom DQN or SB3 DQN checkpoint."
    )
    parser.add_argument("--algo", choices=["custom", "sb3"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--n-videos", type=int, default=3, help="Number of rollout videos.")
    parser.add_argument("--seed", type=int, default=30_000, help="Base seed for rollouts.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/recorded_rollouts",
        help="Directory where MP4 files are saved.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Enable headless/offscreen rendering (safer on servers, can produce black videos).",
    )
    parser.add_argument(
        "--overlay-reward",
        action="store_true",
        help="Overlay per-step reward and cumulative total reward on video frames.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help=(
            "Playback speed multiplier for output videos. "
            "For overlay mode, speed is applied by increasing output FPS without dropping frames."
        ),
    )
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError(f"--speed must be > 0, got {args.speed}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = Path(args.output_dir) / args.algo
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.algo == "custom":
        agent = load_custom_agent(checkpoint_path)
        predictor = lambda obs: agent.get_action(obs, epsilon=0.0)
    else:
        model = SB3DQN.load(str(checkpoint_path))
        predictor = lambda obs: model.predict(obs, deterministic=True)[0]

    rewards = []
    saved_paths: list[Path] = []
    for i in range(args.n_videos):
        run_seed = args.seed + i
        if args.overlay_reward:
            output_path = out_dir / f"{args.algo}_rollout_{i}_annotated.mp4"
            rewards.append(
                record_policy_video_with_overlay_from_config(
                    policy_fn=predictor,
                    env_id=SHARED_CORE_ENV_ID,
                    env_config=SHARED_CORE_CONFIG,
                    save_path=str(output_path),
                    seed=run_seed,
                    headless=args.headless,
                    speed=args.speed,
                )
            )
            saved_paths.append(output_path)
        else:
            rewards.append(
                record_policy_video_from_config(
                    policy_fn=predictor,
                    env_id=SHARED_CORE_ENV_ID,
                    env_config=SHARED_CORE_CONFIG,
                    save_dir=str(out_dir),
                    name_prefix=f"{args.algo}_rollout_{i}",
                    seed=run_seed,
                    headless=args.headless,
                )
            )

    print(f"Saved {args.n_videos} videos to: {out_dir}")
    if saved_paths:
        print("Annotated video paths:")
        for path in saved_paths:
            print(f"- {path}")
    print(f"Rewards: {[round(r, 2) for r in rewards]}")


if __name__ == "__main__":
    main()
