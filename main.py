import gymnasium as gym
import highway_env
import numpy as np
from src.dqn import DQN
from src.evaluate import evaluate_policy
from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TRAINING_CONFIG
from src.train import train_agent
from src.utils import plot_learning_curves, record_final_agent_video

def main():
    training_env = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG)
    render_env = gym.make(SHARED_CORE_ENV_ID, render_mode='rgb_array', config=SHARED_CORE_CONFIG)

    action_space = training_env.action_space
    observation_space = training_env.observation_space

    agent = DQN(action_space,observation_space, **TRAINING_CONFIG)

    losses, rewards = train_agent(training_env, agent, n_episodes=1)

    plot_learning_curves(losses, rewards)
    
    final_scores = evaluate_policy(agent, training_env, n_runs=1)
    print(f"Final Reward : {np.mean(final_scores):.2f} (+/- {np.std(final_scores):.2f})")

    record_final_agent_video(agent, render_env=render_env)

if __name__ == "__main__":
    main()