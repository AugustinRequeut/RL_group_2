import matplotlib.pyplot as plt
import os
from gymnasium.wrappers import RecordVideo

def plot_learning_curves(losses, rewards, epsilon_history, save_dir="results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Graphique de la perte (Loss)
    ax1.plot(losses, color='red')
    ax1.set_title('Training Loss')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Graphique des récompenses (Version simple)
    ax2.plot(rewards, color='blue')
    ax2.set_title('Rewards per Episode')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Reward')
    ax2.grid(True)

    # Graphique de epsilon
    ax3.plot(epsilon_history, color='green')
    ax3.set_title('Epsilon over Updates')
    ax3.set_xlabel('Update steps')
    ax3.set_ylabel('Epsilon')
    ax3.set_ylim(0, 1)
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_performance.png")
    plt.show()

def record_final_agent_video(agent, render_env, save_dir="results/video"):

    video_env = RecordVideo(
        render_env, 
        video_folder=save_dir,
        episode_trigger=lambda x: True,
        disable_logger=True,
        name_prefix="final_agent_run"
    )
    
    try:
        state, _ = video_env.reset()
        done = truncated = False
        total_reward = 0

        while not (done or truncated):
            action = agent.get_action(state, epsilon=0)
            state, reward, done, truncated, _ = video_env.step(action)
            total_reward += reward

        print(f"Video episode total reward: {total_reward:.2f}")
    finally:
        video_env.close()