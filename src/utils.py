import matplotlib.pyplot as plt
import os
import gymnasium as gym
import json
import numpy as np
import imageio.v2 as imageio
from gymnasium.wrappers import RecordVideo

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

def plot_learning_curves(
    losses,
    rewards,
    epsilon_values=None,
    epsilon_x=None,
    epsilon_xlabel="Episodes",
    save_dir="results",
    filename="training_curves.png",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    has_epsilon = epsilon_values is not None
    if has_epsilon:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11))
    else:
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

    if has_epsilon:
        eps_arr = np.asarray(epsilon_values, dtype=np.float32).reshape(-1)
        if epsilon_x is None:
            x_arr = np.arange(1, len(eps_arr) + 1)
        else:
            x_arr = np.asarray(epsilon_x, dtype=np.float32).reshape(-1)

        if len(eps_arr) == 0:
            ax3.plot([], [], color="purple")
            ax3.text(
                0.5,
                0.5,
                "No epsilon data",
                transform=ax3.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
        else:
            ax3.plot(x_arr, eps_arr, color="purple")
        ax3.set_title("Exploration Epsilon")
        ax3.set_xlabel(epsilon_xlabel)
        ax3.set_ylabel("Epsilon")
        ax3.set_ylim(-0.02, 1.02)
        ax3.grid(True)

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


def make_render_env(env_id, env_config, headless=True, use_dummy_driver=True):
    video_config = dict(env_config)
    if headless:
        # Some platforms (notably macOS) can produce black frames with SDL dummy.
        # Keep this behavior configurable for callers that need a virtual display.
        if use_dummy_driver:
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


def _to_uint8_frame(frame):
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _measure_text(draw, text, font):
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _overlay_lines(frame, lines):
    frame_u8 = _to_uint8_frame(frame)
    if Image is None:
        return frame_u8

    image = Image.fromarray(frame_u8)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    padding = 8
    line_spacing = 4
    x0 = 8
    y0 = 8

    line_sizes = [_measure_text(draw, line, font) for line in lines]
    max_w = max((w for w, _ in line_sizes), default=0)
    total_h = 0
    for _, h in line_sizes:
        total_h += h
    if len(line_sizes) > 1:
        total_h += line_spacing * (len(line_sizes) - 1)

    box_w = max_w + 2 * padding
    box_h = total_h + 2 * padding
    draw.rectangle(
        (x0, y0, x0 + box_w, y0 + box_h),
        fill=(0, 0, 0, 150),
    )

    y_text = y0 + padding
    for line, (_, h) in zip(lines, line_sizes):
        draw.text((x0 + padding, y_text), line, fill=(255, 255, 255, 255), font=font)
        y_text += h + line_spacing

    return np.asarray(image)


def _resolve_render_fps(render_env):
    metadata = getattr(render_env, "metadata", {}) or {}
    fps = metadata.get("render_fps", None)
    if fps is None:
        fps = 2.0
    fps = float(fps)
    return max(1.0, fps)


def record_policy_video_with_overlay(
    render_env,
    policy_fn,
    save_path,
    seed=None,
    speed=1.0,
    freeze_final_seconds=1.2,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    base_fps = _resolve_render_fps(render_env)
    out_fps = max(1.0, base_fps * float(speed))

    writer = imageio.get_writer(
        save_path,
        fps=out_fps,
        codec="libx264",
        macro_block_size=1,
    )

    try:
        if seed is None:
            state, _ = render_env.reset()
        else:
            state, _ = render_env.reset(seed=seed)

        done = truncated = False
        total_reward = 0.0
        step_idx = 0

        frame = render_env.render()
        last_frame = frame
        if frame is not None:
            lines = [
                f"Step: {step_idx}",
                "Reward: +0.000",
                f"Total: {total_reward:.3f}",
            ]
            writer.append_data(_overlay_lines(frame, lines))

        while not (done or truncated):
            action = int(policy_fn(state))
            state, reward, done, truncated, _ = render_env.step(action)

            step_idx += 1
            reward_f = float(reward)
            total_reward += reward_f

            frame = render_env.render()
            if frame is None:
                continue
            last_frame = frame

            lines = [
                f"Step: {step_idx}",
                f"Reward: {reward_f:+.3f}",
                f"Total: {total_reward:.3f}",
            ]
            if done or truncated:
                lines.append("Episode: done")

            writer.append_data(_overlay_lines(frame, lines))

        if last_frame is not None and freeze_final_seconds > 0:
            final_lines = [
                "Episode finished",
                f"Total reward: {total_reward:.3f}",
                f"Steps: {step_idx}",
            ]
            final_frame = _overlay_lines(last_frame, final_lines)
            n_freeze = max(1, int(round(out_fps * float(freeze_final_seconds))))
            for _ in range(n_freeze):
                writer.append_data(final_frame)
    finally:
        writer.close()
        render_env.close()

    return float(total_reward)


def record_policy_video_with_overlay_from_config(
    policy_fn,
    env_id,
    env_config,
    save_path,
    seed=None,
    headless=True,
    speed=1.0,
    freeze_final_seconds=1.2,
):
    render_env = make_render_env(
        env_id=env_id,
        env_config=env_config,
        headless=headless,
        use_dummy_driver=False,
    )
    return record_policy_video_with_overlay(
        render_env=render_env,
        policy_fn=policy_fn,
        save_path=save_path,
        seed=seed,
        speed=speed,
        freeze_final_seconds=freeze_final_seconds,
    )
