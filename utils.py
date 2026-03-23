import matplotlib.pyplot as plt
import os
from gymnasium.wrappers import RecordVideo

def plot_learning_curves(losses, rewards, save_dir="results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

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
    
    state, _ = video_env.reset()
    video_env.render()
    
    done = truncated = False
    total_reward = 0
    
    while not (done or truncated):
        action = agent.get_action(state, epsilon=0) 
        state, reward, terminated, truncated, _ = video_env.step(action)
        total_reward += reward
        
    video_env.close()