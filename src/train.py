from tqdm import tqdm
import numpy as np


def train_agent(env, agent, n_episodes=None, total_timesteps=None, eval_every=20):
    if (n_episodes is None) == (total_timesteps is None):
        raise ValueError("Specify exactly one of `n_episodes` or `total_timesteps`.")

    all_losses = []
    all_rewards = []
    episode_rewards = np.zeros(env.num_envs, dtype=np.float32)

    states, _ = env.reset()
    completed_episodes = 0
    steps_collected = 0

    if n_episodes is not None:
        progress_total = n_episodes
        progress_desc = "Training Progress (episodes)"
    else:
        progress_total = total_timesteps
        progress_desc = "Training Progress (timesteps)"

    with tqdm(total=progress_total, desc=progress_desc) as pbar:
        while True:
            if n_episodes is not None and completed_episodes >= n_episodes:
                break
            if total_timesteps is not None and steps_collected >= total_timesteps:
                break

            actions = np.array([agent.get_action(s) for s in states])
            next_states, rewards, terminateds, truncateds, _ = env.step(actions)

            for i in range(env.num_envs):
                loss = agent.update(
                    states[i],
                    int(actions[i]),
                    float(rewards[i]),
                    bool(terminateds[i]),
                    bool(truncateds[i]),
                    next_states[i],
                )
                if loss is not None:
                    loss_arr = np.asarray(loss, dtype=np.float32).reshape(-1)
                    if loss_arr.size > 0:
                        loss_value = float(loss_arr[0])
                        if np.isfinite(loss_value):
                            all_losses.append(loss_value)

            episode_rewards += rewards
            dones = terminateds | truncateds
            for i, done in enumerate(dones):
                if done:
                    all_rewards.append(float(episode_rewards[i]))
                    episode_rewards[i] = 0.0
                    completed_episodes += 1
                    if n_episodes is not None:
                        pbar.update(1)
                        if completed_episodes % eval_every == 0:
                            tqdm.write(
                                f" Ep {completed_episodes}: Last Reward = {all_rewards[-1]:.2f}"
                            )

            states = next_states

            if total_timesteps is not None:
                step_inc = min(env.num_envs, total_timesteps - steps_collected)
                steps_collected += env.num_envs
                pbar.update(step_inc)

    return all_losses, all_rewards
