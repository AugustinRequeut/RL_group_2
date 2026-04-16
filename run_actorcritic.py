import gymnasium as gym
import highway_env
import numpy as np
from src.agents import DQN, REINFORCEBaseline, ActorCriticBasic, ActorCriticGAE
from src.evaluate import evaluate_policy
from src.config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, TRAINING_CONFIG, ACTOR_CRITIC_TRAINING_CONFIG
from src.train import train_agent
from src.utils import plot_learning_curves, record_policy_video

num_envs = TRAINING_CONFIG["num_envs"]

training_env = gym.make_vec(
    SHARED_CORE_ENV_ID,
    num_envs=num_envs,
    config=SHARED_CORE_CONFIG
)
eval_env  = gym.make(SHARED_CORE_ENV_ID, config=SHARED_CORE_CONFIG)
render_env = gym.make(SHARED_CORE_ENV_ID, render_mode='rgb_array', config=SHARED_CORE_CONFIG)

# Use single-env spaces for the agent
action_space = training_env.single_action_space
observation_space = training_env.single_observation_space

# Strip num_envs before passing to DQN
training_config = {k: v for k, v in ACTOR_CRITIC_TRAINING_CONFIG.items() if k != "num_envs"}
agent = ActorCriticGAE(action_space, observation_space, **training_config)

losses, rewards = train_agent(training_env, agent, n_episodes=500)

plot_learning_curves(losses, rewards, agent.epsilon_history)

final_scores = evaluate_policy(agent, eval_env, n_runs=5)
print(f"Final Reward : {np.mean(final_scores):.2f} (+/- {np.std(final_scores):.2f})")