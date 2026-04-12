import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import TRAINING_CONFIG


SB3_EXPLORATION_FRACTION_DEFAULT = 0.3
NUM_SUFFIX_RE = re.compile(r"(\d+)$")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dict_series_to_list(data: dict) -> list[float]:
    ordered = sorted(
        data.items(),
        key=lambda kv: int(NUM_SUFFIX_RE.search(str(kv[0])).group(1)),
    )
    return [float(v) for _, v in ordered]


def _moving_average(values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(arr) < window:
        x = np.arange(1, len(arr) + 1)
        return x, arr
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smooth = np.convolve(arr, kernel, mode="valid")
    x = np.arange(window, len(arr) + 1)
    return x, smooth


def _epsilon_curve_custom(
    n_episodes: int,
    epsilon_start: float,
    epsilon_min: float,
    decrease_epsilon_factor: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    x = np.arange(1, n_episodes + 1, dtype=np.float64)
    y = epsilon_min + (epsilon_start - epsilon_min) * np.exp(
        -1.0 * x / float(decrease_epsilon_factor)
    )
    return x, y, "Completed episodes"


def _epsilon_curve_sb3(
    timesteps: int,
    epsilon_start: float,
    epsilon_final: float,
    exploration_fraction: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    x = np.arange(0, timesteps + 1, dtype=np.float64)
    ramp = max(1.0, float(exploration_fraction) * float(timesteps))
    frac = np.clip(x / ramp, 0.0, 1.0)
    y = epsilon_start + frac * (epsilon_final - epsilon_start)
    return x, y, "Timesteps"


def _build_title(metrics: dict) -> str:
    model = str(metrics.get("model", "unknown")).upper()
    seed = metrics.get("seed", "NA")
    network = metrics.get("custom_network")
    pooling = metrics.get("pooling")
    if network is None:
        return f"{model} | seed={seed}"
    return f"{model} | seed={seed} | net={network} | pooling={pooling}"


def _plot_one_run(
    run_dir: Path,
    output_name: str,
    ma_window: int,
    epsilon_start: float,
    epsilon_final: float,
    custom_decay: float,
    sb3_exploration_fraction: float,
) -> Path:
    metrics = _load_json(run_dir / "metrics.json")
    rewards = _dict_series_to_list(_load_json(run_dir / "train_episode_rewards.json"))
    losses = _dict_series_to_list(_load_json(run_dir / "train_losses.json"))
    model = str(metrics.get("model", "")).lower()

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    # Loss curve
    ax = axes[0]
    if losses:
        x = np.arange(1, len(losses) + 1, dtype=np.float64)
        ax.plot(x, losses, color="#c0392b", alpha=0.35, linewidth=1.0, label="raw")
        x_ma, y_ma = _moving_average(losses, ma_window)
        ax.plot(
            x_ma,
            y_ma,
            color="#e74c3c",
            linewidth=2.0,
            label=f"moving avg ({ma_window})",
        )
    else:
        ax.text(0.5, 0.5, "No training losses", transform=ax.transAxes, ha="center")
    ax.set_title("Training Loss")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    # Reward curve
    ax = axes[1]
    if rewards:
        x = np.arange(1, len(rewards) + 1, dtype=np.float64)
        ax.plot(x, rewards, color="#2471a3", alpha=0.35, linewidth=1.0, label="raw")
        x_ma, y_ma = _moving_average(rewards, ma_window)
        ax.plot(
            x_ma,
            y_ma,
            color="#3498db",
            linewidth=2.0,
            label=f"moving avg ({ma_window})",
        )
    else:
        ax.text(0.5, 0.5, "No training rewards", transform=ax.transAxes, ha="center")
    ax.set_title("Training Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    # Epsilon curve
    ax = axes[2]
    if model == "custom":
        n = len(rewards) if rewards else int(metrics.get("train_completed_episodes", 0))
        x_eps, y_eps, x_label = _epsilon_curve_custom(
            n_episodes=n,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_final,
            decrease_epsilon_factor=custom_decay,
        )
        curve_label = "custom epsilon (exp by completed episodes)"
    elif model == "sb3":
        timesteps = int(metrics.get("timesteps", 0))
        x_eps, y_eps, x_label = _epsilon_curve_sb3(
            timesteps=timesteps,
            epsilon_start=epsilon_start,
            epsilon_final=epsilon_final,
            exploration_fraction=sb3_exploration_fraction,
        )
        curve_label = "sb3 epsilon (linear by timesteps)"
    else:
        raise ValueError(f"Unsupported model in {run_dir / 'metrics.json'}: {model}")

    ax.plot(x_eps, y_eps, color="#8e44ad", linewidth=2.0, label=curve_label)
    ax.set_title("Epsilon Schedule (Reconstructed)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("epsilon")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle(_build_title(metrics), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = run_dir / output_name
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _discover_run_dirs(input_path: Path) -> list[Path]:
    if (input_path / "metrics.json").exists():
        return [input_path]
    run_dirs = [p.parent for p in sorted(input_path.rglob("metrics.json"))]
    return run_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate training curves from JSON artifacts in results directories "
            "(loss, reward, epsilon)."
        )
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="results",
        help="Either a single run dir (containing metrics.json) or a root directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="training_curves_from_json.png",
        help="Output PNG filename inside each run directory.",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=50,
        help="Moving-average window for reward/loss curves.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=float(TRAINING_CONFIG["epsilon_start"]),
        help="Epsilon start value.",
    )
    parser.add_argument(
        "--epsilon-final",
        type=float,
        default=float(TRAINING_CONFIG["epsilon_min"]),
        help="Epsilon final/min value.",
    )
    parser.add_argument(
        "--custom-decay",
        type=float,
        default=float(TRAINING_CONFIG["decrease_epsilon_factor"]),
        help="Custom DQN exponential epsilon decay factor.",
    )
    parser.add_argument(
        "--sb3-exploration-fraction",
        type=float,
        default=SB3_EXPLORATION_FRACTION_DEFAULT,
        help="SB3 DQN exploration_fraction used during training.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    run_dirs = _discover_run_dirs(input_path)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directory found under {input_path} (expected metrics.json)."
        )

    print(f"Found {len(run_dirs)} run(s).")
    for run_dir in run_dirs:
        output_path = _plot_one_run(
            run_dir=run_dir,
            output_name=args.output_name,
            ma_window=max(1, int(args.ma_window)),
            epsilon_start=float(args.epsilon_start),
            epsilon_final=float(args.epsilon_final),
            custom_decay=float(args.custom_decay),
            sb3_exploration_fraction=float(args.sb3_exploration_fraction),
        )
        print(f"[ok] {run_dir} -> {output_path.name}")


if __name__ == "__main__":
    main()
