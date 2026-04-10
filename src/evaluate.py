import numpy as np


def evaluate_policy(
    agent=None,
    env=None,
    n_runs=10,
    policy_fn=None,
    env_factory=None,
    seed_start=None,
):
    if policy_fn is None:
        if agent is None:
            raise ValueError("Provide either `agent` or `policy_fn`.")
        policy_fn = lambda state: agent.get_action(state, epsilon=0)

    if env_factory is None:
        if env is None:
            raise ValueError("Provide either `env` or `env_factory`.")
        env_factory = lambda: env
        reuse_single_env = True
    else:
        reuse_single_env = False

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
            if not reuse_single_env:
                env.close()

    print(f"Mean: {np.mean(all_rewards):.2f} | Std: {np.std(all_rewards):.2f}")
    return all_rewards
