import numpy as np


def evaluate_policy(
    policy_fn,
    env_factory,
    n_runs=10,
    seed_start=None,
):
    all_rewards = []
    for i in range(n_runs):
        env = env_factory()
        try:
            if seed_start is None:
                state, _ = env.reset()
            else:
                state, _ = env.reset(seed=seed_start + i)

            done = truncated = False
            total_reward = 0.0
            while not (done or truncated):
                action = int(policy_fn(state))
                state, reward, done, truncated, _ = env.step(action)
                total_reward += float(reward)
            all_rewards.append(total_reward)
        finally:
            env.close()

    print(f"Mean: {np.mean(all_rewards):.2f} | Std: {np.std(all_rewards):.2f}")
    return all_rewards
