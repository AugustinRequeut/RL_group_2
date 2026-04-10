import argparse
import json
import random
from pathlib import Path
import sys

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch

from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TRAINING_CONFIG
from src.dqn import DQN
from src.evaluate import evaluate_policy
from src.train import train_agent
from src.utils import record_policy_video_from_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom DQN baseline on highway-v0.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=TRAINING_CONFIG.get("num_envs", 4),
        help="Number of vectorized envs.",
    )
    parser.add_argument(
        "--eval-runs", type=int, default=50, help="Number of evaluation episodes."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/custom_dqn",
        help="Directory to save checkpoints and metrics.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast preset for iteration (20k timesteps, 10 eval episodes).",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record one deterministic evaluation episode as MP4.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Where to save videos (default: <output-dir>/seed_<seed>/video).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation episodes to save time.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    run_dir = Path(args.output_dir) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    timesteps = args.timesteps
    eval_runs = args.eval_runs
    if args.quick:
        if "--timesteps" not in sys.argv:
            timesteps = 20_000
        if "--eval-runs" not in sys.argv:
            eval_runs = 10

    train_env = gym.make_vec(
        SHARED_CORE_ENV_ID,
        num_envs=args.num_envs,
        config=SHARED_CORE_CONFIG,
    )

    action_space = train_env.single_action_space
    observation_space = train_env.single_observation_space

    dqn_cfg = {k: v for k, v in TRAINING_CONFIG.items() if k != "num_envs"}

    agent = DQN(
        action_space=action_space,
        observation_space=observation_space,
        **dqn_cfg,
    )

    losses, train_rewards = train_agent(
        env=train_env,
        agent=agent,
        total_timesteps=timesteps,
    )

    eval_env_factory = lambda: gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG)
    eval_policy_fn = lambda state: agent.get_action(state, epsilon=0.0)

    eval_rewards: list[float] = []
    if not args.no_eval:
        eval_rewards = evaluate_policy(
            policy_fn=eval_policy_fn,
            env_factory=eval_env_factory,
            n_runs=eval_runs,
            seed_start=10_000 + args.seed * 1_000,
        )

    video_reward = None
    video_error = None
    if args.record_video:
        try:
            video_dir = Path(args.video_dir) if args.video_dir else run_dir / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_reward = record_policy_video_from_config(
                policy_fn=eval_policy_fn,
                env_id=SHARED_CORE_ENV_ID,
                env_config=SHARED_CORE_CONFIG,
                save_dir=str(video_dir),
                name_prefix=f"custom_dqn_seed_{args.seed}",
                seed=20_000 + args.seed * 1_000,
                headless=True,
            )
        except Exception as exc:
            video_error = str(exc)
            print(f"Custom DQN video recording failed: {video_error}")

    metrics = {
        "seed": args.seed,
        "timesteps": timesteps,
        "eval_runs": 0 if args.no_eval else eval_runs,
        "train_completed_episodes": len(train_rewards),
        "mean_reward": float(np.mean(eval_rewards)) if eval_rewards else None,
        "std_reward": float(np.std(eval_rewards)) if eval_rewards else None,
        "video_reward": float(video_reward) if video_reward is not None else None,
        "video_error": video_error,
        "quick_mode": bool(args.quick),
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if eval_rewards:
        np.save(run_dir / "eval_rewards.npy", np.array(eval_rewards, dtype=np.float32))
    np.save(run_dir / "train_rewards.npy", np.array(train_rewards, dtype=np.float32))
    np.save(run_dir / "train_losses.npy", np.array(losses, dtype=np.float32))

    torch.save(agent.q_net.state_dict(), run_dir / "custom_dqn_qnet.pt")

    if eval_rewards:
        print(
            f"Custom DQN | seed={args.seed} | mean={metrics['mean_reward']:.2f} "
            f"+/- {metrics['std_reward']:.2f} over {eval_runs} runs"
        )
    else:
        print(f"Custom DQN | seed={args.seed} | evaluation skipped")
    if video_reward is not None:
        print(f"Custom DQN video reward (seed={args.seed}): {video_reward:.2f}")
    print(f"Metrics saved to: {run_dir / 'metrics.json'}")

    train_env.close()


if __name__ == "__main__":
    main()
