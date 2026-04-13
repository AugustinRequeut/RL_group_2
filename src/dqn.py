# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (
            np.array(state, dtype=np.float32),
            action,
            float(reward),
            bool(terminated),
            np.array(next_state, dtype=np.float32),
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_shape, hidden_size, n_actions):
        super(Net, self).__init__()
        input_size = np.prod(obs_shape)

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())


def _masked_pool(x, mask, mode="mean"):
    """
    x: (B, N, D)
    mask: (B, N) with 1 for valid entities, 0 for padded/missing entities.
    """
    mask = mask.float().unsqueeze(-1)  # (B, N, 1)
    if mode == "mean":
        denom = mask.sum(dim=1).clamp_min(1.0)  # (B, 1)
        return (x * mask).sum(dim=1) / denom

    if mode == "max":
        neg_inf = torch.finfo(x.dtype).min
        masked = x.masked_fill(mask == 0, neg_inf)
        pooled = masked.max(dim=1).values
        # If no valid entity, max is -inf: replace by zeros.
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    raise ValueError(f"Unknown pooling mode '{mode}'. Use 'mean' or 'max'.")


class SharedPoolNet(nn.Module):
    """
    Architecture 1:
    - shared MLP on each non-ego vehicle: phi(5 -> 128 -> 128)
    - pooling over non-ego embeddings (mean or max)
    - concat with ego features
    - output head: (5 + 128) -> 128 -> 128 -> n_actions
    """

    def __init__(self, obs_shape, n_actions, pooling="mean"):
        super().__init__()
        if len(obs_shape) != 2:
            raise ValueError(
                "SharedPoolNet expects obs shape (n_vehicles, n_features), "
                f"got {obs_shape}"
            )
        n_vehicles, n_features = int(obs_shape[0]), int(obs_shape[1])
        if n_vehicles < 2:
            raise ValueError("SharedPoolNet needs at least 2 vehicles (ego + non-ego).")

        self.n_vehicles = n_vehicles
        self.n_features = n_features
        self.pooling = pooling
        self.embed_dim = 128

        self.phi = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(n_features + self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def _ensure_batched(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")
        return x

    def forward(self, x):
        x = self._ensure_batched(x.float())
        ego = x[:, 0, :]  # (B, F)
        others = x[:, 1:, :]  # (B, N-1, F)

        bsz, n_other, _ = others.shape
        flat = others.reshape(-1, self.n_features)
        emb = self.phi(flat).view(bsz, n_other, self.embed_dim)

        # "presence" is feature index 0 in this environment config.
        presence_mask = others[:, :, 0] > 0
        pooled = _masked_pool(emb, presence_mask, mode=self.pooling)

        fused = torch.cat([ego, pooled], dim=1)
        return self.head(fused)


class PairwiseEgoNet(nn.Module):
    """
    Architecture 2:
    - shared MLP on each non-ego vehicle: phi(5 -> 128 -> 128)
    - for each non-ego i: concat(ego, phi_i)
    - shared pairwise MLP psi: (5 + 128) -> 128 -> 128
    - pooling over pairwise embeddings (mean or max)
    - output head: (5 + 128) -> 128 -> 128 -> n_actions
    """

    def __init__(self, obs_shape, n_actions, pooling="mean"):
        super().__init__()
        if len(obs_shape) != 2:
            raise ValueError(
                "PairwiseEgoNet expects obs shape (n_vehicles, n_features), "
                f"got {obs_shape}"
            )
        n_vehicles, n_features = int(obs_shape[0]), int(obs_shape[1])
        if n_vehicles < 2:
            raise ValueError("PairwiseEgoNet needs at least 2 vehicles (ego + non-ego).")

        self.n_vehicles = n_vehicles
        self.n_features = n_features
        self.pooling = pooling
        self.embed_dim = 128
        pair_in = n_features + self.embed_dim

        self.phi = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.psi = nn.Sequential(
            nn.Linear(pair_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(pair_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def _ensure_batched(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")
        return x

    def forward(self, x):
        x = self._ensure_batched(x.float())
        ego = x[:, 0, :]  # (B, F)
        others = x[:, 1:, :]  # (B, N-1, F)

        bsz, n_other, _ = others.shape
        flat = others.reshape(-1, self.n_features)
        other_emb = self.phi(flat).view(bsz, n_other, self.embed_dim)

        ego_tiled = ego.unsqueeze(1).expand(-1, n_other, -1)
        pair_input = torch.cat([ego_tiled, other_emb], dim=-1)
        pair_emb = self.psi(pair_input.reshape(-1, self.n_features + self.embed_dim)).view(
            bsz, n_other, self.embed_dim
        )

        presence_mask = others[:, :, 0] > 0
        pooled = _masked_pool(pair_emb, presence_mask, mode=self.pooling)

        fused = torch.cat([ego, pooled], dim=1)
        return self.head(fused)


class DQN:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        batch_size,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
        learning_rate,
        epsilon_warmup_episodes=0,
        gradient_clip_norm=100.0,
        network_type="flat_mlp",
        pooling="mean",
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = (
            decrease_epsilon_factor  # larger -> more exploration
        )
        self.epsilon_min = epsilon_min
        self.epsilon_warmup_episodes = max(0, int(epsilon_warmup_episodes))

        self.learning_rate = learning_rate
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.network_type = network_type
        self.pooling = pooling

        self.reset()

    def update(self, state, action, reward, terminated, truncated, next_state):
        # add data to replay buffer
        self.buffer.push(state, action, reward, terminated, next_state)

        if len(self.buffer) < self.batch_size:
            return None

        # get batch
        transitions = self.buffer.sample(self.batch_size)
        states, actions, rewards, terminateds, next_states = zip(*transitions)

        state_batch = torch.tensor(np.array(states), dtype=torch.float32)
        action_batch = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(np.array(rewards), dtype=torch.float32)
        terminated_batch = torch.tensor(np.array(terminateds), dtype=torch.int64)
        next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float32)

        values = self.q_net.forward(state_batch).gather(1, action_batch)

        # Compute the ideal Q values
        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(
                next_state_batch
            ).max(1)[0]
            targets = next_state_values * self.gamma + reward_batch

        loss = self.loss_function(values, targets.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), max_norm=self.gradient_clip_norm
        )
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()

        self.n_steps += 1
        if terminated or truncated:
            self.n_eps += 1

        return loss.detach().numpy()

    def get_action(self, state, epsilon=None):
        """
        Return action according to an epsilon-greedy exploration policy
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.get_q(state))
        
    def get_q(self, state):
        """
        Compute Q function for a states
        """
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            output = self.q_net.forward(state_tensor) # shape (1,  n_actions)
        return output.numpy()[0]

    def decrease_epsilon(self):
        if self.n_eps < self.epsilon_warmup_episodes:
            self.epsilon = self.epsilon_start
            return

        decayed_episodes = self.n_eps - self.epsilon_warmup_episodes
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * decayed_episodes / self.decrease_epsilon_factor)
        )

    def reset(self):
        hidden_size = 256

        obs_shape = self.observation_space.shape
        n_actions = self.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        if self.network_type == "flat_mlp":
            self.q_net = Net(obs_shape, hidden_size, n_actions)
            self.target_net = Net(obs_shape, hidden_size, n_actions)
        elif self.network_type == "shared_pool":
            self.q_net = SharedPoolNet(
                obs_shape=obs_shape,
                n_actions=n_actions,
                pooling=self.pooling,
            )
            self.target_net = SharedPoolNet(
                obs_shape=obs_shape,
                n_actions=n_actions,
                pooling=self.pooling,
            )
        elif self.network_type == "pairwise_ego":
            self.q_net = PairwiseEgoNet(
                obs_shape=obs_shape,
                n_actions=n_actions,
                pooling=self.pooling,
            )
            self.target_net = PairwiseEgoNet(
                obs_shape=obs_shape,
                n_actions=n_actions,
                pooling=self.pooling,
            )
        else:
            raise ValueError(
                f"Unknown network_type='{self.network_type}'. "
                "Use one of: flat_mlp, shared_pool, pairwise_ego."
            )

        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0

class REINFORCEBaseline(DQN):
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        batch_size,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
        learning_rate,
        epsilon_warmup_episodes=0,
        gradient_clip_norm=100.0,
        network_type="flat_mlp",
        pooling="mean",
    ):
        super().__init__(action_space,
        observation_space,
        gamma,
        batch_size,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
        learning_rate,
        epsilon_warmup_episodes,
        gradient_clip_norm,
        network_type,
        pooling)
        
        self.current_episode = []

    def get_action(self, state, epsilon=None):
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            h = self.q_net(state_tensor)
            policy = torch.softmax(h, dim=1)
            return torch.multinomial(policy, 1)[0,0].numpy()
        
    def _gradient_returns(self, rewards, gamma):
        """
        Turns a list of rewards into the list of returns * gamma**t
        """
        G = 0
        returns_list = []
        T = len(rewards)
        full_gamma = np.power(gamma, T)
        for t in range(T):
            G = rewards[T-t-1] + gamma * G
            full_gamma /= gamma
            returns_list.append(full_gamma * G)
        return torch.tensor(returns_list[::-1])
    
    def update(self, state, action, reward, terminated, truncated, next_state):
        
        self.current_episode.append((
            torch.tensor(state).unsqueeze(0),
            torch.tensor([[int(action)]], dtype=torch.int64),
            torch.tensor([reward]),
        )
        )

        if terminated or truncated:

            states, actions, rewards = tuple(
                [torch.cat(data) for data in zip(*self.current_episode)]
            )

            gain_t = self._gradient_returns(rewards, self.gamma).float()

            b_t = gain_t.mean()

            h = self.q_net(states)
            log_probs = torch.log_softmax(h, dim=1)

            full_neg_score = - torch.dot(log_probs.gather(1, actions).squeeze(-1), (gain_t-b_t)).unsqueeze(0)

            self.current_episode = []

            self.optimizer.zero_grad()
            full_neg_score.backward()
            self.optimizer.step()

            return full_neg_score.detach().numpy()
        
        return None
