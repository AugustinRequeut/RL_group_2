import argparse
import json
import random
import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from stable_baselines3 import DQN as SB3DQN
from stable_baselines3.common.env_util import make_vec_env

from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TRAINING_CONFIG
from src.dqn import DQN, REINFORCEBaseline
from src.evaluate import evaluate_policy
from src.train import train_agent
from src.utils import plot_learning_curves, record_policy_video_from_config


DEFAULT_OUTPUT_DIRS = {
    "custom": "results/custom_dqn",
    "sb3": "results/sb3_dqn",
    "reinforce": "results/reinforce",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_output_dir(model: str, output_dir: str | None) -> Path:
    resolved = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIRS[model]
    return Path(resolved)


def _apply_quick_defaults(args, model: str) -> None:
    if not args.quick:
        return

    if model in {"custom", "sb3"}:
        if "--timesteps" not in sys.argv:
            args.timesteps = 20_000
        if "--eval-runs" not in sys.argv:
            args.eval_runs = 10
    elif model == "reinforce":
        if "--episodes" not in sys.argv:
            args.episodes = 20
        if "--eval-runs" not in sys.argv:
            args.eval_runs = 10


def _evaluate_and_record(
    *,
    policy_fn,
    model_name: str,
    seed: int,
    eval_runs: int,
    no_eval: bool,
    record_video: bool,
    run_dir: Path,
    video_dir_arg: str | None,
    headless_video: bool,
):
    eval_env_factory = lambda: gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG)

    eval_rewards: list[float] = []
    if not no_eval:
        eval_rewards = evaluate_policy(
            policy_fn=policy_fn,
            env_factory=eval_env_factory,
            n_runs=eval_runs,
            seed_start=10_000 + seed * 1_000,
        )

    video_reward = None
    video_error = None
    if record_video:
        try:
            video_dir = Path(video_dir_arg) if video_dir_arg else run_dir / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_reward = record_policy_video_from_config(
                policy_fn=policy_fn,
                env_id=SHARED_CORE_ENV_ID,
                env_config=SHARED_CORE_CONFIG,
                save_dir=str(video_dir),
                name_prefix=f"{model_name}_seed_{seed}",
                seed=20_000 + seed * 1_000,
                headless=headless_video,
            )
        except Exception as exc:
            video_error = str(exc)
            print(f"{model_name} video recording failed: {video_error}")

    return eval_rewards, video_reward, video_error


def _run_custom(args) -> None:
    seed_everything(args.seed)

    root_dir = _resolve_output_dir("custom", args.output_dir)
    run_dir = root_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

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
        total_timesteps=args.timesteps,
    )

    if args.plot_curves:
        plot_learning_curves(losses, train_rewards, save_dir=str(run_dir))

    eval_policy_fn = lambda state: agent.get_action(state, epsilon=0.0)
    eval_rewards, video_reward, video_error = _evaluate_and_record(
        policy_fn=eval_policy_fn,
        model_name="custom_dqn",
        seed=args.seed,
        eval_runs=args.eval_runs,
        no_eval=args.no_eval,
        record_video=args.record_video,
        run_dir=run_dir,
        video_dir_arg=args.video_dir,
        headless_video=args.headless_video,
    )

    metrics = {
        "model": "custom",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "eval_runs": 0 if args.no_eval else args.eval_runs,
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
            f"+/- {metrics['std_reward']:.2f} over {args.eval_runs} runs"
        )
    else:
        print(f"Custom DQN | seed={args.seed} | evaluation skipped")
    if video_reward is not None:
        print(f"Custom DQN video reward (seed={args.seed}): {video_reward:.2f}")
    print(f"Metrics saved to: {run_dir / 'metrics.json'}")

    train_env.close()


def _run_sb3(args) -> None:
    seed_everything(args.seed)

    root_dir = _resolve_output_dir("sb3", args.output_dir)
    run_dir = root_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

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
        learning_rate=TRAINING_CONFIG["learning_rate"],
        batch_size=TRAINING_CONFIG["batch_size"],
        gamma=TRAINING_CONFIG["gamma"],
        buffer_size=TRAINING_CONFIG["buffer_capacity"],
        target_update_interval=TRAINING_CONFIG["update_target_every"],
        learning_starts=1_000,
        train_freq=4,
        gradient_steps=1,
        exploration_initial_eps=TRAINING_CONFIG["epsilon_start"],
        exploration_final_eps=TRAINING_CONFIG["epsilon_min"],
        exploration_fraction=0.3,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar)
    model_path = run_dir / "sb3_dqn_model"
    model.save(str(model_path))

    eval_policy_fn = lambda state: model.predict(state, deterministic=True)[0]
    eval_rewards, video_reward, video_error = _evaluate_and_record(
        policy_fn=eval_policy_fn,
        model_name="sb3_dqn",
        seed=args.seed,
        eval_runs=args.eval_runs,
        no_eval=args.no_eval,
        record_video=args.record_video,
        run_dir=run_dir,
        video_dir_arg=args.video_dir,
        headless_video=args.headless_video,
    )

    metrics = {
        "model": "sb3",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "eval_runs": 0 if args.no_eval else args.eval_runs,
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
            f"+/- {metrics['std_reward']:.2f} over {args.eval_runs} runs"
        )
    else:
        print(f"SB3 DQN | seed={args.seed} | evaluation skipped")
    if video_reward is not None:
        print(f"SB3 DQN video reward (seed={args.seed}): {video_reward:.2f}")
    print(f"Model saved to: {model_path}.zip")
    print(f"Metrics saved to: {run_dir / 'metrics.json'}")

    vec_env.close()


def _run_reinforce(args) -> None:
    seed_everything(args.seed)

    root_dir = _resolve_output_dir("reinforce", args.output_dir)
    run_dir = root_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = gym.make_vec(
        SHARED_CORE_ENV_ID,
        num_envs=args.num_envs,
        config=SHARED_CORE_CONFIG,
    )

    action_space = train_env.single_action_space
    observation_space = train_env.single_observation_space
    dqn_cfg = {k: v for k, v in TRAINING_CONFIG.items() if k != "num_envs"}
    agent = REINFORCEBaseline(
        action_space=action_space,
        observation_space=observation_space,
        **dqn_cfg,
    )

    losses, train_rewards = train_agent(
        env=train_env,
        agent=agent,
        n_episodes=args.episodes,
    )

    if args.plot_curves:
        plot_learning_curves(losses, train_rewards, save_dir=str(run_dir))

    eval_policy_fn = lambda state: agent.get_action(state, epsilon=0.0)
    eval_rewards, video_reward, video_error = _evaluate_and_record(
        policy_fn=eval_policy_fn,
        model_name="reinforce",
        seed=args.seed,
        eval_runs=args.eval_runs,
        no_eval=args.no_eval,
        record_video=args.record_video,
        run_dir=run_dir,
        video_dir_arg=args.video_dir,
        headless_video=args.headless_video,
    )

    metrics = {
        "model": "reinforce",
        "seed": args.seed,
        "episodes": args.episodes,
        "eval_runs": 0 if args.no_eval else args.eval_runs,
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
    torch.save(agent.q_net.state_dict(), run_dir / "reinforce_qnet.pt")

    if eval_rewards:
        print(
            f"REINFORCE | seed={args.seed} | mean={metrics['mean_reward']:.2f} "
            f"+/- {metrics['std_reward']:.2f} over {args.eval_runs} runs"
        )
    else:
        print(f"REINFORCE | seed={args.seed} | evaluation skipped")
    if video_reward is not None:
        print(f"REINFORCE video reward (seed={args.seed}): {video_reward:.2f}")
    print(f"Metrics saved to: {run_dir / 'metrics.json'}")

    train_env.close()


def _build_parser(model_override: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified RL runner for custom, SB3 and REINFORCE.")
    if model_override is None:
        parser.add_argument(
            "--model",
            choices=["custom", "sb3", "reinforce"],
            default="custom",
            help="Model to train/evaluate.",
        )

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=TRAINING_CONFIG.get("num_envs", 4),
        help="Number of vectorized environments.",
    )
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps.")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes.")
    parser.add_argument("--eval-runs", type=int, default=50, help="Evaluation episodes.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory (default depends on model).",
    )
    parser.add_argument("--quick", action="store_true", help="Use quick default budget.")
    parser.add_argument("--record-video", action="store_true", help="Record one rollout video.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Video directory (default: <output-dir>/seed_<seed>/video).",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation.")
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable SB3 progress bar (requires rich).",
    )
    parser.add_argument(
        "--plot-curves",
        action="store_true",
        help="Save training curves for custom/reinforce.",
    )
    parser.add_argument(
        "--headless-video",
        action="store_true",
        default=True,
        help="Use headless/offscreen rendering for video recording.",
    )
    parser.add_argument(
        "--no-headless-video",
        action="store_false",
        dest="headless_video",
        help="Disable headless rendering (better visuals locally, can fail on servers).",
    )
    return parser


def run_experiment_cli(model_override: str | None = None) -> None:
    parser = _build_parser(model_override=model_override)
    args = parser.parse_args()
    model = model_override if model_override is not None else args.model

    _apply_quick_defaults(args, model=model)

    if model == "custom":
        _run_custom(args)
    elif model == "sb3":
        _run_sb3(args)
    else:
        _run_reinforce(args)
