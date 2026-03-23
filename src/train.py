from tqdm import tqdm

def train_agent(env, agent, n_episodes, eval_every=20):
    all_losses = []
    all_rewards = []
    
    for ep in tqdm(range(n_episodes), desc="Training Progress"):
        state, _ = env.reset()
        done = truncated = False
        ep_reward = 0
        
        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            loss = agent.update(state, action, reward, terminated, next_state)
            
            state = next_state
            ep_reward += reward
            if loss is not None:
                all_losses.append(loss)
        
        all_rewards.append(ep_reward)

        if (ep + 1) % eval_every == 0:
            tqdm.write(f" Ep {ep+1}: Last Reward = {ep_reward:.2f}")
                        
    return all_losses, all_rewards