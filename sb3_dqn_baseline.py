import argparse
import json
from pathlib import Path
import sys

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3 import DQN as SB3DQN
from stable_baselines3.common.env_util import make_vec_env

from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID
from src.evaluate import evaluate_policy
from src.utils import record_policy_video_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SB3 DQN baseline on highway-v0.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of vectorized envs.")
    parser.add_argument(
        "--eval-runs", type=int, default=50, help="Number of evaluation episodes."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sb3_dqn",
        help="Directory to save checkpoints and metrics.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable SB3 progress bar (requires `rich` package).",
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

    run_dir = Path(args.output_dir) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    timesteps = args.timesteps
    eval_runs = args.eval_runs
    if args.quick:
        if "--timesteps" not in sys.argv:
            timesteps = 20_000
        if "--eval-runs" not in sys.argv:
            eval_runs = 10

    vec_env = make_vec_env(
        SHARED_CORE_ENV_ID,
        n_envs=args.num_envs,
        seed=args.seed,
        env_kwargs={"config": SHARED_CORE_CONFIG},
    )

    model = SB3DQN(
        "MlpPolicy",
        vec_env,
        seed=args.seed,
        learning_rate=1e-4,
        batch_size=128,
        gamma=0.97,
        buffer_size=50_000,
        target_update_interval=500,
        learning_starts=1_000,
        train_freq=4,
        gradient_steps=1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        verbose=1,
    )

    model.learn(total_timesteps=timesteps, progress_bar=args.progress_bar)
    model_path = run_dir / "sb3_dqn_model"
    model.save(str(model_path))

    eval_env_factory = lambda: gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG)
    eval_policy_fn = lambda state: model.predict(state, deterministic=True)[0]

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
                name_prefix=f"sb3_dqn_seed_{args.seed}",
                seed=20_000 + args.seed * 1_000,
                headless=True,
            )
        except Exception as exc:
            video_error = str(exc)
            print(f"SB3 DQN video recording failed: {video_error}")

    metrics = {
        "seed": args.seed,
        "timesteps": timesteps,
        "eval_runs": 0 if args.no_eval else eval_runs,
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

    if eval_rewards:
        print(
            f"SB3 DQN | seed={args.seed} | mean={metrics['mean_reward']:.2f} "
            f"+/- {metrics['std_reward']:.2f} over {eval_runs} runs"
        )
    else:
        print(f"SB3 DQN | seed={args.seed} | evaluation skipped")
    if video_reward is not None:
        print(f"SB3 DQN video reward (seed={args.seed}): {video_reward:.2f}")
    print(f"Model saved to: {model_path}.zip")
    print(f"Metrics saved to: {run_dir / 'metrics.json'}")

    vec_env.close()


if __name__ == "__main__":
    main()
