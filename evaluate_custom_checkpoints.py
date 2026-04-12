import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
from collections import Counter
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN as SB3DQN
from tqdm import tqdm

from record_trained_videos import load_custom_agent
from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


CHECKPOINT_PATTERNS = {
    "custom": re.compile(r"custom_dqn_qnet_ep_(\d+)\.pt$"),
    "sb3": re.compile(r"sb3_dqn_model_ep_(\d+)\.zip$"),
}


def _extract_episode_index(checkpoint_path: Path, algo: str) -> int | None:
    match = CHECKPOINT_PATTERNS[algo].match(checkpoint_path.name)
    if match is None:
        return None
    return int(match.group(1))


def _discover_checkpoints(run_dir: Path, algo: str) -> list[dict]:
    if algo == "custom":
        checkpoint_glob = "custom_dqn_qnet_ep_*.pt"
        final_name = "custom_dqn_qnet.pt"
    else:
        checkpoint_glob = "sb3_dqn_model_ep_*.zip"
        final_name = "sb3_dqn_model.zip"

    checkpoint_dir = run_dir / "checkpoints"
    intermediate_paths = []
    if checkpoint_dir.exists():
        intermediate_paths = sorted(
            checkpoint_dir.glob(checkpoint_glob),
            key=lambda p: (_extract_episode_index(p, algo) or -1, p.name),
        )

    checkpoints = []
    for path in intermediate_paths:
        episode_idx = _extract_episode_index(path, algo)
        checkpoints.append(
            {
                "path": path,
                "label": f"ep_{episode_idx:06d}" if episode_idx is not None else path.stem,
                "episode_index": episode_idx,
                "is_final": False,
            }
        )

    final_path = run_dir / final_name
    if final_path.exists():
        checkpoints.append(
            {
                "path": final_path,
                "label": "final",
                "episode_index": None,
                "is_final": True,
            }
        )

    return checkpoints


def _resolve_run_dir(path: Path) -> Path:
    if path.name == "checkpoints":
        return path.parent
    return path


def _lane_id(env) -> int | None:
    return int(env.unwrapped.vehicle.lane_index[2])


def _termination_reason(terminated: bool, truncated: bool, info: dict, env) -> str:
    if truncated:
        return "timeout"

    if bool(info["crashed"]):
        return "crash"

    if not bool(env.unwrapped.vehicle.on_road):
        return "offroad"

    # Keep only the three categories used in the report.
    _ = terminated
    return "timeout"


def _step_speed(info: dict, env) -> float | None:
    _ = env
    return float(info["speed"])


def _percent_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    counts = Counter(values)
    total = float(len(values))
    return {str(k): (100.0 * v / total) for k, v in sorted(counts.items(), key=lambda x: x[0])}


def _nearest_target_speed(speed: float, target_speeds: list[float]) -> float | None:
    return min(target_speeds, key=lambda t: abs(speed - t))


def _summarize_episode_records(records: list[dict], all_step_speeds: list[float]) -> dict:
    rewards = np.array([r["total_reward"] for r in records], dtype=np.float32)
    lengths = np.array([r["episode_length"] for r in records], dtype=np.float32)
    lane_changes = np.array([r["lane_changes"] for r in records], dtype=np.float32)
    speeds = np.array(all_step_speeds, dtype=np.float32)

    reason_counts = Counter(r["termination_reason"] for r in records)
    known_order = ["crash", "offroad", "timeout"]
    counts = {k: int(reason_counts.get(k, 0)) for k in known_order}
    rates = {k: (counts[k] / len(records) if records else 0.0) for k in known_order}

    rounded_speed_1dp = [round(float(s), 1) for s in all_step_speeds]
    target_speeds = [float(s) for s in SHARED_CORE_CONFIG.get("action", {}).get("target_speeds", [])]
    nearest_targets = [_nearest_target_speed(float(s), target_speeds) for s in all_step_speeds]

    return {
        "n_episodes": int(len(records)),
        "mean_reward": float(rewards.mean()) if len(rewards) else None,
        "std_reward": float(rewards.std()) if len(rewards) else None,
        "mean_episode_length": float(lengths.mean()) if len(lengths) else None,
        "std_episode_length": float(lengths.std()) if len(lengths) else None,
        "mean_lane_changes": float(lane_changes.mean()) if len(lane_changes) else None,
        "std_lane_changes": float(lane_changes.std()) if len(lane_changes) else None,
        "termination_reason_counts": counts,
        "termination_reason_rates": rates,
        "speed_stats": {
            "num_step_samples": int(len(all_step_speeds)),
            "mean_speed": float(speeds.mean()) if len(speeds) else None,
            "std_speed": float(speeds.std()) if len(speeds) else None,
            "min_speed": float(speeds.min()) if len(speeds) else None,
            "max_speed": float(speeds.max()) if len(speeds) else None,
        },
        "speed_time_percent_by_speed_rounded_1dp": _percent_distribution(rounded_speed_1dp),
        "speed_time_percent_by_nearest_target_speed": _percent_distribution(nearest_targets),
    }


def _mean_ci95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean())
    if n < 2:
        return mean, mean, mean
    stderr = float(arr.std(ddof=1) / np.sqrt(n))
    delta = 1.96 * stderr
    return mean, mean - delta, mean + delta


def _proportion_ci95(successes: int, n: int) -> tuple[float, float, float]:
    p = successes / n
    z = 1.96
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return float(p), float(lo), float(hi)


def _checkpoint_evolution_stats(episode_records: list[dict]) -> dict:
    n = len(episode_records)
    rewards = [float(r["total_reward"]) for r in episode_records]
    mean_speeds = [float(r["mean_speed"]) for r in episode_records]
    crashes = sum(1 for r in episode_records if r["termination_reason"] == "crash")

    reward_mean, reward_lo, reward_hi = _mean_ci95(rewards)
    speed_mean, speed_lo, speed_hi = _mean_ci95(mean_speeds)
    crash_rate, crash_lo, crash_hi = _proportion_ci95(crashes, n)

    return {
        "n_episodes": int(n),
        "crash_rate": crash_rate,
        "crash_rate_ci95_low": crash_lo,
        "crash_rate_ci95_high": crash_hi,
        "mean_reward": reward_mean,
        "mean_reward_ci95_low": reward_lo,
        "mean_reward_ci95_high": reward_hi,
        "mean_speed": speed_mean,
        "mean_speed_ci95_low": speed_lo,
        "mean_speed_ci95_high": speed_hi,
    }


def _plot_metric_with_ci(
    ax,
    x: list[float],
    y: list[float],
    lo: list[float],
    hi: list[float],
    title: str,
    ylabel: str,
    color: str,
) -> None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    lo_arr = np.asarray(lo, dtype=np.float64)
    hi_arr = np.asarray(hi, dtype=np.float64)
    ax.plot(x_arr, y_arr, marker="o", linewidth=2.0, color=color)
    ax.fill_between(x_arr, lo_arr, hi_arr, color=color, alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def _save_evolution_plots(
    evolution_rows: list[dict],
    output_dir: Path,
) -> Path | None:
    if not evolution_rows:
        return None

    rows_with_epoch = [r for r in evolution_rows if r["episode_index"] is not None]
    if len(rows_with_epoch) >= 2:
        rows = sorted(rows_with_epoch, key=lambda r: int(r["episode_index"]))
        x = [int(r["episode_index"]) for r in rows]
        xlabel = "Checkpoint episode index"
    else:
        rows = evolution_rows
        x = list(range(1, len(rows) + 1))
        xlabel = "Checkpoint order"

    crash = [100.0 * float(r["stats"]["crash_rate"]) for r in rows]
    crash_lo = [100.0 * float(r["stats"]["crash_rate_ci95_low"]) for r in rows]
    crash_hi = [100.0 * float(r["stats"]["crash_rate_ci95_high"]) for r in rows]

    reward = [float(r["stats"]["mean_reward"]) for r in rows]
    reward_lo = [float(r["stats"]["mean_reward_ci95_low"]) for r in rows]
    reward_hi = [float(r["stats"]["mean_reward_ci95_high"]) for r in rows]

    speed = [float(r["stats"]["mean_speed"]) for r in rows]
    speed_lo = [float(r["stats"]["mean_speed_ci95_low"]) for r in rows]
    speed_hi = [float(r["stats"]["mean_speed_ci95_high"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    _plot_metric_with_ci(
        axes[0], x, crash, crash_lo, crash_hi, "Crash Rate by Checkpoint", "Crash rate (%)", "#c0392b"
    )
    _plot_metric_with_ci(
        axes[1], x, speed, speed_lo, speed_hi, "Mean Speed by Checkpoint", "Speed (m/s)", "#2980b9"
    )
    _plot_metric_with_ci(
        axes[2], x, reward, reward_lo, reward_hi, "Mean Reward by Checkpoint", "Reward", "#16a085"
    )
    axes[2].set_xlabel(xlabel)
    fig.tight_layout()

    plot_path = output_dir / "checkpoint_eval_evolution_ci95.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _evaluate_checkpoint(
    checkpoint_path: Path,
    n_episodes: int,
    seed_start: int,
    algo: str,
    show_episode_tqdm: bool = True,
) -> tuple[list[dict], dict, list[float]]:
    if algo == "custom":
        agent = load_custom_agent(checkpoint_path)
        policy_fn = lambda obs: int(agent.get_action(obs, epsilon=0.0))
    else:
        model = SB3DQN.load(str(checkpoint_path), device="cpu")
        policy_fn = lambda obs: int(np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[0])

    episode_records: list[dict] = []
    all_step_speeds: list[float] = []

    episode_iter = range(n_episodes)
    if show_episode_tqdm:
        episode_iter = tqdm(
            episode_iter,
            total=n_episodes,
            desc=f"Episodes {checkpoint_path.stem}",
            unit="ep",
            leave=False,
        )

    for episode_idx in episode_iter:
        seed = seed_start + episode_idx
        env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG)
        try:
            state, info = env.reset(seed=seed)
            _ = info

            total_reward = 0.0
            episode_length = 0
            lane_changes = 0
            lane_prev = _lane_id(env)
            episode_step_speeds: list[float] = []

            terminated = truncated = False
            final_info: dict = {}

            while not (terminated or truncated):
                action = policy_fn(state)
                state, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                episode_length += 1

                speed_now = _step_speed(info, env)
                all_step_speeds.append(float(speed_now))
                episode_step_speeds.append(float(speed_now))

                lane_now = _lane_id(env)
                if lane_now != lane_prev:
                    lane_changes += abs(lane_now - lane_prev)
                lane_prev = lane_now

                final_info = info

            episode_records.append(
                {
                    "total_reward": float(total_reward),
                    "episode_length": int(episode_length),
                    "lane_changes": int(lane_changes),
                    "mean_speed": float(np.mean(episode_step_speeds)),
                    "termination_reason": _termination_reason(
                        bool(terminated), bool(truncated), final_info, env
                    ),
                }
            )
        finally:
            env.close()

    return (
        episode_records,
        _summarize_episode_records(episode_records, all_step_speeds),
        all_step_speeds,
    )


def _evaluate_checkpoint_task(task: dict) -> dict:
    checkpoint_path = Path(task["checkpoint_path"])
    episode_records, summary, step_speeds = _evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        n_episodes=int(task["episodes_per_checkpoint"]),
        seed_start=int(task["seed_start"]),
        algo=str(task["algo"]),
        show_episode_tqdm=True,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": task["checkpoint_label"],
        "episode_index": task["episode_index"],
        "summary": summary,
        "episode_records": episode_records,
        "step_speeds": step_speeds,
        "index": int(task["index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple custom or SB3 checkpoints and export "
            "aggregated diagnostics."
        )
    )
    parser.add_argument(
        "--algo",
        choices=["custom", "sb3"],
        default="custom",
        help="Which checkpoint family to evaluate.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="results/custom_dqn/flat_mlp/seed_0",
        help=(
            "Run directory containing final checkpoint and optional intermediate "
            "checkpoint files in <run-dir>/checkpoints."
        ),
    )
    parser.add_argument(
        "--episodes-per-checkpoint",
        type=int,
        default=20,
        help="How many episodes to evaluate for each checkpoint.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=40_000,
        help="Base seed used for evaluation episodes.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes to evaluate checkpoints in parallel. "
            "Use 1 for sequential mode."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: <run-dir>/checkpoint_eval_diagnostics.json",
    )
    args = parser.parse_args()
    if args.episodes_per_checkpoint <= 0:
        raise ValueError("--episodes-per-checkpoint must be >= 1")

    run_dir = _resolve_run_dir(Path(args.run_dir))
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    discovered = _discover_checkpoints(run_dir, args.algo)
    if not discovered:
        if args.algo == "custom":
            expected = "custom_dqn_qnet.pt and/or checkpoints/custom_dqn_qnet_ep_*.pt"
        else:
            expected = "sb3_dqn_model.zip and/or checkpoints/sb3_dqn_model_ep_*.zip"
        raise FileNotFoundError(
            f"No checkpoints found for algo='{args.algo}'. Expected {expected}"
        )

    selected = list(discovered)

    output_path = (
        Path(args.output)
        if args.output is not None
        else run_dir / "checkpoint_eval_diagnostics.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Algo={args.algo} | discovered {len(discovered)} checkpoints, "
        f"selected {len(selected)} (parallel_workers={args.parallel_workers})."
    )

    results: list[dict] = []
    evolution_rows: list[dict] = []
    total_step_speeds_all_selected: list[float] = []
    total_episode_records_all_selected: list[dict] = []
    if args.parallel_workers <= 1:
        for idx, item in enumerate(selected, start=1):
            checkpoint_path = item["path"]
            print(
                f"[{idx}/{len(selected)}] Evaluating {item['label']} "
                f"({checkpoint_path})..."
            )
            episode_records, summary, step_speeds = _evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                n_episodes=args.episodes_per_checkpoint,
                seed_start=args.seed_start,
                algo=args.algo,
                show_episode_tqdm=True,
            )
            total_episode_records_all_selected.extend(episode_records)
            total_step_speeds_all_selected.extend(step_speeds)
            evolution_rows.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_label": item["label"],
                    "episode_index": item["episode_index"],
                    "stats": _checkpoint_evolution_stats(episode_records),
                }
            )
            results.append(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_label": item["label"],
                    "episode_index": item["episode_index"],
                    "summary": summary,
                }
            )
            print(
                f"  mean_reward={summary['mean_reward']:.2f} | "
                f"mean_len={summary['mean_episode_length']:.2f} | "
                f"mean_lane_changes={summary['mean_lane_changes']:.2f} | "
                f"reasons={summary['termination_reason_counts']}"
            )
    else:
        tasks = [
            {
                "index": idx,
                "checkpoint_path": str(item["path"]),
                "checkpoint_label": item["label"],
                "episode_index": item["episode_index"],
                "episodes_per_checkpoint": int(args.episodes_per_checkpoint),
                "seed_start": int(args.seed_start),
                "algo": str(args.algo),
            }
            for idx, item in enumerate(selected, start=1)
        ]
        results_buffer: list[dict | None] = [None] * len(tasks)
        evolution_buffer: list[dict | None] = [None] * len(tasks)

        with ProcessPoolExecutor(max_workers=int(args.parallel_workers)) as executor:
            future_to_idx = {
                executor.submit(_evaluate_checkpoint_task, task): int(task["index"])
                for task in tasks
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                task_result = future.result()
                summary = task_result["summary"]
                print(
                    f"[{idx}/{len(selected)}] Done {task_result['checkpoint_label']} | "
                    f"mean_reward={summary['mean_reward']:.2f} | "
                    f"mean_len={summary['mean_episode_length']:.2f} | "
                    f"mean_lane_changes={summary['mean_lane_changes']:.2f} | "
                    f"reasons={summary['termination_reason_counts']}"
                )

                total_episode_records_all_selected.extend(task_result["episode_records"])
                total_step_speeds_all_selected.extend(task_result["step_speeds"])
                evolution_buffer[idx - 1] = {
                    "checkpoint_path": task_result["checkpoint_path"],
                    "checkpoint_label": task_result["checkpoint_label"],
                    "episode_index": task_result["episode_index"],
                    "stats": _checkpoint_evolution_stats(task_result["episode_records"]),
                }
                results_buffer[idx - 1] = {
                    "checkpoint_path": task_result["checkpoint_path"],
                    "checkpoint_label": task_result["checkpoint_label"],
                    "episode_index": task_result["episode_index"],
                    "summary": task_result["summary"],
                }

        results = [r for r in results_buffer if r is not None]
        evolution_rows = [r for r in evolution_buffer if r is not None]

    payload = {
        "algo": str(args.algo),
        "run_dir": str(run_dir),
        "num_discovered_checkpoints": len(discovered),
        "num_selected_checkpoints": len(selected),
        "episodes_per_checkpoint": int(args.episodes_per_checkpoint),
        "seed_start": int(args.seed_start),
        "selected_checkpoints": [
            {
                "path": str(item["path"]),
                "label": item["label"],
                "episode_index": item["episode_index"],
                "is_final": bool(item["is_final"]),
            }
            for item in selected
        ],
        "overall_selected_checkpoints_summary": _summarize_episode_records(
            total_episode_records_all_selected,
            total_step_speeds_all_selected,
        ),
        "checkpoint_evolution_ci95": evolution_rows,
        "results": results,
    }

    plot_path = _save_evolution_plots(evolution_rows=evolution_rows, output_dir=output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved diagnostics to: {output_path}")
    if plot_path is not None:
        print(f"Saved evolution plot to: {plot_path}")


if __name__ == "__main__":
    main()
