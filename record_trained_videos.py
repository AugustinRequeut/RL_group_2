import argparse
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import torch
from stable_baselines3 import DQN as SB3DQN

from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TRAINING_CONFIG
from src.dqn import DQN
from src.utils import record_policy_video_from_config


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
            record_policy_video_from_config(
                policy_fn=predictor,
                env_id=SHARED_CORE_ENV_ID,
                env_config=SHARED_CORE_CONFIG,
                save_dir=str(out_dir),
                name_prefix=f"{args.algo}_rollout_{i}",
                seed=args.seed + i,
                headless=args.headless,
            )
        )

    print(f"Saved {args.n_videos} videos to: {out_dir}")
    print(f"Rewards: {[round(r, 2) for r in rewards]}")


if __name__ == "__main__":
    main()
