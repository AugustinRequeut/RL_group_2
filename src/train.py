from tqdm import tqdm
import numpy as np

def train_agent(env, agent, n_episodes, eval_every=20):
    all_losses = []
    all_rewards = []
    episode_rewards = np.zeros(env.num_envs)  # track reward per env

    states, _ = env.reset()
    completed_episodes = 0

    with tqdm(total=n_episodes, desc="Training Progress") as pbar:
        while completed_episodes < n_episodes:
            actions = np.array([agent.get_action(s) for s in states])
            next_states, rewards, terminateds, truncateds, _ = env.step(actions)

            # Push one transition per parallel env
            for i in range(env.num_envs):
                loss = agent.update(
                    states[i], actions[i], rewards[i],
                    terminateds[i], truncateds[i], next_states[i]
                )
                if loss is not None:
                    all_losses.append(loss)

            episode_rewards += rewards

            # Detect completed episodes (any env that terminated or truncated)
            dones = terminateds | truncateds
            for i, done in enumerate(dones):
                if done:
                    all_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0
                    completed_episodes += 1
                    pbar.update(1)
                    if completed_episodes % eval_every == 0:
                        tqdm.write(f" Ep {completed_episodes}: Last Reward = {all_rewards[-1]:.2f}")

            states = next_states

    return all_losses, all_rewards