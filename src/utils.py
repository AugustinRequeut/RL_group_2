import matplotlib.pyplot as plt
import os
import gymnasium as gym
import json
from gymnasium.wrappers import RecordVideo

def plot_learning_curves(losses, rewards, save_dir="results", filename="training_curves.png"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    if losses is None or len(losses) == 0:
        ax1.plot([], [], color="red")
        ax1.text(
            0.5,
            0.5,
            "No training losses recorded",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    else:
        ax1.plot(losses, color="red")
    ax1.set_title("Training Loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    ax2.plot(rewards, color="blue")
    ax2.set_title("Rewards per Episode")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Reward")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close(fig)


def export_episode_rewards_dict(rewards, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    episode_rewards = {f"episode_{i+1}": float(r) for i, r in enumerate(rewards)}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(episode_rewards, f, indent=2)


def export_train_losses_dict(losses, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_losses = {f"update_{i+1}": float(v) for i, v in enumerate(losses)}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(train_losses, f, indent=2)


def export_eval_rewards_dict(rewards, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    eval_rewards = {f"run_{i+1}": float(r) for i, r in enumerate(rewards)}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(eval_rewards, f, indent=2)


def record_policy_video(
    render_env,
    policy_fn,
    save_dir="results/video",
    name_prefix="rollout",
    seed=None,
):
    video_env = RecordVideo(
        render_env,
        video_folder=save_dir,
        episode_trigger=lambda _: True,
        disable_logger=True,
        name_prefix=name_prefix,
    )

    try:
        if seed is None:
            state, _ = video_env.reset()
        else:
            state, _ = video_env.reset(seed=seed)
        done = truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action = int(policy_fn(state))
            state, reward, done, truncated, _ = video_env.step(action)
            total_reward += reward
    finally:
        video_env.close()

    return float(total_reward)


def make_render_env(env_id, env_config, headless=True):
    video_config = dict(env_config)
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        video_config["offscreen_rendering"] = True

    return gym.make(
        env_id,
        render_mode="rgb_array",
        config=video_config,
    )


def record_policy_video_from_config(
    policy_fn,
    env_id,
    env_config,
    save_dir="results/video",
    name_prefix="rollout",
    seed=None,
    headless=True,
):
    render_env = make_render_env(
        env_id=env_id,
        env_config=env_config,
        headless=headless,
    )
    return record_policy_video(
        render_env=render_env,
        policy_fn=policy_fn,
        save_dir=save_dir,
        name_prefix=name_prefix,
        seed=seed,
    )
