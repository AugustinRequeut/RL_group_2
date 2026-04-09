import argparse
import json
from pathlib import Path

import numpy as np


def load_metrics_by_seed(root_dir: Path) -> dict[int, dict]:
    metrics_by_seed: dict[int, dict] = {}

    if not root_dir.exists():
        return metrics_by_seed

    for metrics_file in sorted(root_dir.glob("seed_*/metrics.json")):
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        seed = int(data["seed"])
        metrics_by_seed[seed] = data

    return metrics_by_seed


def summarize_algo(name: str, metrics_by_seed: dict[int, dict]) -> dict:
    valid_items = [(s, m) for s, m in sorted(metrics_by_seed.items()) if m.get("mean_reward") is not None]

    if not valid_items:
        return {
            "name": name,
            "n_seeds": 0,
            "seed_mean_rewards": np.array([], dtype=np.float32),
            "overall_mean": float("nan"),
            "overall_std": float("nan"),
        }

    seed_mean_rewards = np.array([m["mean_reward"] for _, m in valid_items], dtype=np.float32)

    return {
        "name": name,
        "n_seeds": int(len(seed_mean_rewards)),
        "seed_mean_rewards": seed_mean_rewards,
        "overall_mean": float(seed_mean_rewards.mean()),
        "overall_std": float(seed_mean_rewards.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare custom DQN and SB3 DQN metrics.")
    parser.add_argument(
        "--custom-dir",
        type=str,
        default="results/custom_dqn",
        help="Directory containing custom DQN seed folders.",
    )
    parser.add_argument(
        "--sb3-dir",
        type=str,
        default="results/sb3_dqn",
        help="Directory containing SB3 DQN seed folders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dqn_comparison_summary.json",
        help="Path for the comparison summary JSON.",
    )
    args = parser.parse_args()

    custom_metrics = load_metrics_by_seed(Path(args.custom_dir))
    sb3_metrics = load_metrics_by_seed(Path(args.sb3_dir))

    common_seeds = sorted(
        s
        for s in (set(custom_metrics.keys()) & set(sb3_metrics.keys()))
        if custom_metrics[s].get("mean_reward") is not None
        and sb3_metrics[s].get("mean_reward") is not None
    )

    custom_summary = summarize_algo("custom_dqn", custom_metrics)
    sb3_summary = summarize_algo("sb3_dqn", sb3_metrics)

    paired_diff = None
    if common_seeds:
        custom_common = np.array([custom_metrics[s]["mean_reward"] for s in common_seeds])
        sb3_common = np.array([sb3_metrics[s]["mean_reward"] for s in common_seeds])
        paired_diff = {
            "common_seeds": common_seeds,
            "custom_minus_sb3_mean": float((custom_common - sb3_common).mean()),
            "custom_minus_sb3_std": float((custom_common - sb3_common).std()),
        }

    report = {
        "custom": {
            "n_seeds": custom_summary["n_seeds"],
            "overall_mean_of_seed_means": custom_summary["overall_mean"],
            "overall_std_of_seed_means": custom_summary["overall_std"],
            "seeds": sorted(
                s for s, m in custom_metrics.items() if m.get("mean_reward") is not None
            ),
        },
        "sb3": {
            "n_seeds": sb3_summary["n_seeds"],
            "overall_mean_of_seed_means": sb3_summary["overall_mean"],
            "overall_std_of_seed_means": sb3_summary["overall_std"],
            "seeds": sorted(
                s for s, m in sb3_metrics.items() if m.get("mean_reward") is not None
            ),
        },
        "paired_difference": paired_diff,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== DQN Comparison ===")
    print(
        f"Custom DQN: n={report['custom']['n_seeds']} | "
        f"mean={report['custom']['overall_mean_of_seed_means']:.2f} | "
        f"std={report['custom']['overall_std_of_seed_means']:.2f}"
    )
    print(
        f"SB3 DQN:    n={report['sb3']['n_seeds']} | "
        f"mean={report['sb3']['overall_mean_of_seed_means']:.2f} | "
        f"std={report['sb3']['overall_std_of_seed_means']:.2f}"
    )

    if paired_diff is None:
        print("No common seeds found yet between custom and SB3 results.")
    else:
        print(
            "Paired (custom - sb3) over common seeds "
            f"{paired_diff['common_seeds']}: "
            f"mean={paired_diff['custom_minus_sb3_mean']:.2f}, "
            f"std={paired_diff['custom_minus_sb3_std']:.2f}"
        )

    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()
