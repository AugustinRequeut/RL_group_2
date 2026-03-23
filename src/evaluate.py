import numpy as np

def evaluate_policy(agent, env, n_runs=10):
    all_rewards = []
    for _ in range(n_runs):
        state, _ = env.reset()
        done = truncated = False
        total_reward = 0
        while not (done or truncated):
            action = agent.get_action(state, epsilon=0)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        all_rewards.append(total_reward)
    
    print(f"Mean: {np.mean(all_rewards):.2f} | Std: {np.std(all_rewards):.2f}")
    return all_rewards