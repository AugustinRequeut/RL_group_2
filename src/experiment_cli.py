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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TRAINING_CONFIG
from src.dqn import DQN, REINFORCEBaseline
from src.evaluate import evaluate_policy
from src.train import train_agent
from src.utils import (
    export_eval_rewards_dict,
    export_episode_rewards_dict,
    export_train_losses_dict,
    plot_learning_curves,
)


DEFAULT_OUTPUT_DIRS = {
    "custom": "results/custom_dqn",
    "sb3": "results/sb3_dqn",
    "reinforce": "results/reinforce",
}

CUSTOM_NETWORK_CHOICES = ["flat_mlp", "shared_pool", "pairwise_ego"]


class EpisodeRewardCallback(BaseCallback):
    def __init__(
        self,
        log_every_episodes: int = 20,
        checkpoint_every_episodes: int = 0,
        checkpoint_dir: Path | None = None,
        checkpoint_prefix: str = "sb3_dqn_model",
        save_json_every_episodes: int = 0,
        run_dir: Path | None = None,
        train_losses: list[float] | None = None,
    ):
        super().__init__()
        self.episode_rewards = []
        self.log_every_episodes = log_every_episodes
        self.checkpoint_every_episodes = checkpoint_every_episodes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.save_json_every_episodes = save_json_every_episodes
        self.run_dir = run_dir
        self.train_losses = train_losses
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is not None and "r" in episode_info:
                reward = float(episode_info["r"])
                self.episode_rewards.append(reward)
                n = len(self.episode_rewards)
                if self.log_every_episodes > 0 and n % self.log_every_episodes == 0:
                    recent = self.episode_rewards[-self.log_every_episodes :]
                    recent_mean = float(np.mean(recent))
                    print(
                        f"[SB3 train] Ep {n}: Last Reward = {reward:.2f} | "
                        f"Mean(last {self.log_every_episodes}) = {recent_mean:.2f}"
                    )
                if (
                    self.checkpoint_every_episodes > 0
                    and self.checkpoint_dir is not None
                    and n % self.checkpoint_every_episodes == 0
                ):
                    checkpoint_path = (
                        self.checkpoint_dir / f"{self.checkpoint_prefix}_ep_{n:06d}"
                    )
                    self.model.save(str(checkpoint_path))
                    print(f"[checkpoint] Saved: {checkpoint_path}.zip")
                if (
                    self.save_json_every_episodes > 0
                    and self.run_dir is not None
                    and n % self.save_json_every_episodes == 0
                ):
                    _save_training_json_artifacts(
                        run_dir=self.run_dir,
                        train_rewards=self.episode_rewards,
                        train_losses=self.train_losses,
                    )
                    print(
                        f"[train-json] Saved training JSON snapshots at episode {n}: "
                        f"{self.run_dir / 'train_episode_rewards.json'}"
                    )
        return True


def _attach_sb3_loss_collector(model: SB3DQN, loss_store: list[float]) -> None:
    original_train = model.train

    def _train_with_capture(*args, **kwargs):
        result = original_train(*args, **kwargs)
        logger_values = getattr(model.logger, "name_to_value", {})
        maybe_loss = logger_values.get("train/loss")
        if maybe_loss is not None and np.isfinite(maybe_loss):
            loss_store.append(float(maybe_loss))
        return result

    model.train = _train_with_capture


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
            args.timesteps = 5_000
        if "--eval-runs" not in sys.argv:
            args.eval_runs = 10
    elif model == "reinforce":
        if "--episodes" not in sys.argv:
            args.episodes = 20
        if "--eval-runs" not in sys.argv:
            args.eval_runs = 10


def _evaluate_policy_only(
    *,
    policy_fn,
    seed: int,
    eval_runs: int,
    no_eval: bool,
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

    return eval_rewards


def _save_training_json_artifacts(
    *,
    run_dir: Path,
    train_rewards: list[float],
    train_losses: list[float] | None,
) -> None:
    export_episode_rewards_dict(train_rewards, str(run_dir / "train_episode_rewards.json"))
    safe_losses = [] if train_losses is None else train_losses
    export_train_losses_dict(safe_losses, str(run_dir / "train_losses.json"))


def _save_training_episode_artifacts(
    run_dir: Path,
    train_rewards: list[float],
    train_losses: list[float] | None,
) -> None:
    safe_losses = [] if train_losses is None else train_losses
    _save_training_json_artifacts(
        run_dir=run_dir,
        train_rewards=train_rewards,
        train_losses=train_losses,
    )
    plot_learning_curves(
        losses=safe_losses,
        rewards=train_rewards,
        save_dir=str(run_dir),
        filename="training_curves.png",
    )


def _save_eval_json_artifacts(run_dir: Path, eval_rewards: list[float]) -> None:
    export_eval_rewards_dict(eval_rewards, str(run_dir / "eval_rewards.json"))


def _make_torch_training_callback(
    *,
    agent,
    run_dir: Path,
    checkpoint_prefix: str,
    checkpoint_every_episodes: int,
    save_json_every_episodes: int,
):
    if checkpoint_every_episodes <= 0 and save_json_every_episodes <= 0:
        return None

    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_every_episodes > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_episode_end(
        episode_idx: int,
        _episode_reward: float,
        train_rewards: list[float],
        train_losses: list[float],
    ) -> None:
        if checkpoint_every_episodes > 0 and episode_idx % checkpoint_every_episodes == 0:
            checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_ep_{episode_idx:06d}.pt"
            torch.save(agent.q_net.state_dict(), checkpoint_path)
            print(f"[checkpoint] Saved: {checkpoint_path}")

        if save_json_every_episodes > 0 and episode_idx % save_json_every_episodes == 0:
            _save_training_json_artifacts(
                run_dir=run_dir,
                train_rewards=train_rewards,
                train_losses=train_losses,
            )
            print(
                f"[train-json] Saved training JSON snapshots at episode {episode_idx}: "
                f"{run_dir / 'train_episode_rewards.json'}"
            )

    return _on_episode_end


def _run_torch_algo(
    args,
    *,
    model_key: str,
    model_label: str,
    agent_cls,
    checkpoint_prefix: str,
    final_checkpoint_name: str,
    use_timesteps: bool,
) -> None:
    seed_everything(args.seed)

    root_dir = _resolve_output_dir(model_key, args.output_dir)
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
    dqn_cfg["network_type"] = args.custom_network
    dqn_cfg["pooling"] = args.pooling

    agent = agent_cls(
        action_space=action_space,
        observation_space=observation_space,
        **dqn_cfg,
    )
    on_episode_end = _make_torch_training_callback(
        agent=agent,
        run_dir=run_dir,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_every_episodes=args.checkpoint_every_episodes,
        save_json_every_episodes=args.save_json_every_episodes,
    )

    train_kwargs = {
        "env": train_env,
        "agent": agent,
        "eval_every": args.log_train_every,
        "on_episode_end": on_episode_end,
    }
    if use_timesteps:
        train_kwargs["total_timesteps"] = args.timesteps
    else:
        train_kwargs["n_episodes"] = args.episodes
    losses, train_rewards = train_agent(**train_kwargs)

    _save_training_episode_artifacts(
        run_dir=run_dir,
        train_rewards=train_rewards,
        train_losses=losses,
    )

    eval_policy_fn = lambda state: agent.get_action(state, epsilon=0.0)
    eval_rewards = _evaluate_policy_only(
        policy_fn=eval_policy_fn,
        seed=args.seed,
        eval_runs=args.eval_runs,
        no_eval=args.no_eval,
    )

    metrics = {
        "model": model_key,
        "seed": args.seed,
        "eval_runs": 0 if args.no_eval else args.eval_runs,
        "train_completed_episodes": len(train_rewards),
        "mean_reward": float(np.mean(eval_rewards)) if eval_rewards else None,
        "std_reward": float(np.std(eval_rewards)) if eval_rewards else None,
        "quick_mode": bool(args.quick),
        "custom_network": args.custom_network,
        "pooling": args.pooling,
        "checkpoint_every_episodes": int(args.checkpoint_every_episodes),
        "save_json_every_episodes": int(args.save_json_every_episodes),
    }
    if use_timesteps:
        metrics["timesteps"] = args.timesteps
    else:
        metrics["episodes"] = args.episodes

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if eval_rewards:
        _save_eval_json_artifacts(run_dir, eval_rewards)
    torch.save(agent.q_net.state_dict(), run_dir / final_checkpoint_name)

    if eval_rewards:
        print(
            f"{model_label} | seed={args.seed} | mean={metrics['mean_reward']:.2f} "
            f"+/- {metrics['std_reward']:.2f} over {args.eval_runs} runs"
        )
    else:
        print(f"{model_label} | seed={args.seed} | evaluation skipped")
    print(f"Metrics saved to: {run_dir / 'metrics.json'}")

    train_env.close()


def _run_custom(args) -> None:
    _run_torch_algo(
        args,
        model_key="custom",
        model_label="Custom DQN",
        agent_cls=DQN,
        checkpoint_prefix="custom_dqn_qnet",
        final_checkpoint_name="custom_dqn_qnet.pt",
        use_timesteps=True,
    )


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
        policy_kwargs={"net_arch": [256, 256]},
        learning_rate=TRAINING_CONFIG["learning_rate"],
        batch_size=TRAINING_CONFIG["batch_size"],
        gamma=TRAINING_CONFIG["gamma"],
        buffer_size=TRAINING_CONFIG["buffer_capacity"],
        target_update_interval=TRAINING_CONFIG["update_target_every"],
        learning_starts=TRAINING_CONFIG["batch_size"],
        train_freq=4,
        gradient_steps=1,
        exploration_initial_eps=TRAINING_CONFIG["epsilon_start"],
        exploration_final_eps=TRAINING_CONFIG["epsilon_min"],
        exploration_fraction=0.3,
        verbose=1,
    )
    train_losses: list[float] = []
    _attach_sb3_loss_collector(model, train_losses)

    reward_callback = EpisodeRewardCallback(
        log_every_episodes=args.log_train_every,
        checkpoint_every_episodes=args.checkpoint_every_episodes,
        checkpoint_dir=run_dir / "checkpoints",
        checkpoint_prefix="sb3_dqn_model",
        save_json_every_episodes=args.save_json_every_episodes,
        run_dir=run_dir,
        train_losses=train_losses,
    )
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=args.progress_bar,
        callback=reward_callback,
    )
    model_path = run_dir / "sb3_dqn_model"
    model.save(str(model_path))
    train_rewards = reward_callback.episode_rewards

    _save_training_episode_artifacts(
        run_dir=run_dir,
        train_rewards=train_rewards,
        train_losses=train_losses,
    )

    eval_policy_fn = lambda state: model.predict(state, deterministic=True)[0]
    eval_rewards = _evaluate_policy_only(
        policy_fn=eval_policy_fn,
        seed=args.seed,
        eval_runs=args.eval_runs,
        no_eval=args.no_eval,
    )

    metrics = {
        "model": "sb3",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "eval_runs": 0 if args.no_eval else args.eval_runs,
        "train_completed_episodes": len(train_rewards),
        "mean_reward": float(np.mean(eval_rewards)) if eval_rewards else None,
        "std_reward": float(np.std(eval_rewards)) if eval_rewards else None,
        "quick_mode": bool(args.quick),
        "checkpoint_every_episodes": int(args.checkpoint_every_episodes),
        "save_json_every_episodes": int(args.save_json_every_episodes),
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if eval_rewards:
        _save_eval_json_artifacts(run_dir, eval_rewards)

    if eval_rewards:
        print(
            f"SB3 DQN | seed={args.seed} | mean={metrics['mean_reward']:.2f} "
            f"+/- {metrics['std_reward']:.2f} over {args.eval_runs} runs"
        )
    else:
        print(f"SB3 DQN | seed={args.seed} | evaluation skipped")
    print(f"Model saved to: {model_path}.zip")
    print(f"Metrics saved to: {run_dir / 'metrics.json'}")

    vec_env.close()


def _run_reinforce(args) -> None:
    _run_torch_algo(
        args,
        model_key="reinforce",
        model_label="REINFORCE",
        agent_cls=REINFORCEBaseline,
        checkpoint_prefix="reinforce_qnet",
        final_checkpoint_name="reinforce_qnet.pt",
        use_timesteps=False,
    )


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
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation.")
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable SB3 progress bar (requires rich).",
    )
    parser.add_argument(
        "--log-train-every",
        type=int,
        default=50,
        help="Print training reward statistics every N completed episodes.",
    )
    parser.add_argument(
        "--checkpoint-every-episodes",
        type=int,
        default=100,
        help="Save intermediate checkpoints every N completed episodes (0 disables).",
    )
    parser.add_argument(
        "--save-json-every-episodes",
        type=int,
        default=100,
        help="Update training JSON snapshots every N completed episodes (0 disables).",
    )
    parser.add_argument(
        "--custom-network",
        choices=CUSTOM_NETWORK_CHOICES,
        default="flat_mlp",
        help=(
            "Network architecture for custom/reinforce: "
            "flat_mlp | shared_pool | pairwise_ego."
        ),
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "max"],
        default="mean",
        help="Pooling mode for shared_pool and pairwise_ego architectures.",
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
