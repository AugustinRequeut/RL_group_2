import argparse
import os
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import torch
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN as SB3DQN

from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TRAINING_CONFIG
from src.dqn import DQN


def load_custom_agent(checkpoint_path: Path) -> DQN:
    env = gym.make_vec(
        SHARED_CORE_ENV_ID,
        num_envs=1,
        config=SHARED_CORE_CONFIG,
    )
    action_space = env.single_action_space
    observation_space = env.single_observation_space
    dqn_cfg = {k: v for k, v in TRAINING_CONFIG.items() if k != "num_envs"}
    agent = DQN(action_space=action_space, observation_space=observation_space, **dqn_cfg)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    agent.q_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    env.close()
    return agent


def record_one_episode(
    algo: str,
    predictor,
    out_dir: Path,
    base_seed: int,
    video_idx: int,
    headless: bool,
) -> float:
    video_config = dict(SHARED_CORE_CONFIG)
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        video_config["offscreen_rendering"] = True

    env = gym.make(
        SHARED_CORE_ENV_ID,
        render_mode="rgb_array",
        config=video_config,
    )
    wrapped_env = RecordVideo(
        env,
        video_folder=str(out_dir),
        episode_trigger=lambda _: True,
        disable_logger=True,
        name_prefix=f"{algo}_rollout_{video_idx}",
    )

    try:
        obs, _ = wrapped_env.reset(seed=base_seed + video_idx)
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action = predictor(obs)
            obs, reward, done, truncated, _ = wrapped_env.step(int(action))
            total_reward += float(reward)
    finally:
        wrapped_env.close()

    return total_reward


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
    args = parser.parse_args()

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
    for i in range(args.n_videos):
        rewards.append(
            record_one_episode(
                algo=args.algo,
                predictor=predictor,
                out_dir=out_dir,
                base_seed=args.seed,
                video_idx=i,
                headless=args.headless,
            )
        )

    print(f"Saved {args.n_videos} videos to: {out_dir}")
    print(f"Rewards: {[round(r, 2) for r in rewards]}")


if __name__ == "__main__":
    main()
