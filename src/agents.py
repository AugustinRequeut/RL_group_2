# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class AttentionNet(nn.Module):
    """
    Permutation-invariant network for kinematic observations.
    Each vehicle is treated as a token; multi-head attention
    aggregates the fleet before predicting Q-values.
    """

    def __init__(self, obs_shape, hidden_size, n_actions, n_heads=8):
        super(AttentionNet, self).__init__()
        n_vehicles, n_features = obs_shape
        
        self.embedding = nn.Linear(n_features, hidden_size)
        self.register_buffer('position',torch.zeros((n_vehicles,1)))
        self.position[0] = 1.

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):

        tokens = F.relu(self.embedding(x))+self.position

        attended, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm(tokens + attended)          # residual connection

        pooled = tokens.mean(dim=1)

        return self.head(pooled)

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

        self.learning_rate = learning_rate

        self.reset()

    def update(self, state, action, reward, terminated, truncated, next_state):
        # add data to replay buffer
        self.buffer.push(state, action, reward, terminated, next_state)

        if len(self.buffer) < self.batch_size:
            return np.inf

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
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
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
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * self.n_eps / self.decrease_epsilon_factor)
        )
        self.epsilon_history.append(self.epsilon)

    def reset(self):
        hidden_size = 256

        obs_shape = self.observation_space.shape
        n_actions = self.action_space.n

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net = AttentionNet(obs_shape, hidden_size, n_actions)
        self.target_net = AttentionNet(obs_shape, hidden_size, n_actions)

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0
        self.epsilon_history = []

class REINFORCEBaseline:
    """
    Implementation of the REINFORCE algorithm, with a baseline.
    """

    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        batch_size,
        learning_rate,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.episode_batch_size = batch_size
        self.learning_rate = learning_rate

        # Reset
        hidden_size = 256

        n_actions = self.action_space.n

        self.policy_net = AttentionNet(self.observation_space.shape, hidden_size, n_actions)

        self.scores = []
        self.current_episode = []

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0
        self.epsilon_history = []

    def update(self, state, action, reward, terminated, truncated, next_state):
        self.current_episode.append(
            (
                torch.tensor(state).unsqueeze(0),
                torch.tensor(action, dtype=torch.int64).unsqueeze(0).unsqueeze(0),
                torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
            )
        )

        if terminated or truncated:
            self.n_eps += 1

            states, actions, rewards = tuple(
                [torch.cat(data) for data in zip(*self.current_episode)]
            )

            current_episode_returns = self._returns(rewards, self.gamma)
            # this is where the baseline happens
            current_episode_returns = (
                current_episode_returns - current_episode_returns.mean()
            )

            unn_log_probs = self.policy_net.forward(states)
            log_probs = unn_log_probs - torch.log(
                torch.sum(torch.exp(unn_log_probs), dim=1)
            ).unsqueeze(1)

            selected_log_probs = log_probs.gather(1, actions).squeeze()
            discounts = torch.pow(self.gamma, torch.arange(len(rewards)))

            score = discounts * selected_log_probs * current_episode_returns

            self.scores.append(score)

            self.current_episode = []

            full_neg_score = -torch.cat(self.scores).sum() / max(1, len(self.scores))

            if (self.n_eps % self.episode_batch_size) == 0:
                self.optimizer.zero_grad()
                batch_loss = -torch.cat(self.scores).sum() / self.episode_batch_size
                batch_loss.backward()
                self.optimizer.step()
                self.scores = []

            return full_neg_score.detach().numpy()

        return None

    def _returns(self, rewards, gamma):
        """ """
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def get_action(self, state, epsilon=None):
        """
        Return action according to an epsilon-greedy exploration policy
        """
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            unn_log_probs = self.policy_net(state_tensor)
            p = torch.softmax(unn_log_probs, dim=1)[0].numpy()
            return np.random.choice(np.arange(self.action_space.n), p=p)

    def train_reset(self):
        self.scores = []
        self.current_episode = []

    def reset(self):
        hidden_size = 256

        obs_size = self.observation_space.shape[0]
        n_actions = self.action_space.n

        self.policy_net = AttentionNet(obs_size, hidden_size, n_actions)

        self.scores = []
        self.current_episode = []

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0

class ActorCriticBasic:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        actor_learning_rate,
        critic_learning_rate,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        self.loss_function = nn.MSELoss()

        # Reset
        hidden_size = 256

        n_actions = self.action_space.n

        self.actor = AttentionNet(self.observation_space.shape, hidden_size, n_actions)
        self.critic = AttentionNet(self.observation_space.shape, hidden_size, 1)

        self.actor_optimizer = optim.Adam(
            params=self.actor.parameters(), lr=self.actor_learning_rate
        )

        self.critic_optimizer = optim.Adam(
            params=self.critic.parameters(), lr=self.critic_learning_rate
        )

        self.current_episode = []
        self.episode_reward = 0

        self.scores = []

        self.n_eps = 0
        self.total_steps = 0

        self.epsilon_history = []

    def get_action(self, state, epsilon=None):
        """
        Return action according to an epsilon-greedy exploration policy
        """
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            unn_log_probs = self.actor(state_tensor)
            p = torch.softmax(unn_log_probs, dim=1).numpy()[0]
            return np.random.choice(np.arange(self.action_space.n), p=p)

    def compute_gradient_score(self):
        states, actions, rewards, terminals, next_states = tuple(
            [torch.cat(data) for data in zip(*self.current_episode)]
        )

        with torch.no_grad():
            target_values = (
                rewards
                + self.gamma * (1 - terminals) * self.critic(next_states).squeeze()
            )
            values = self.critic(states).squeeze()
            advantages = target_values - values

        logits = self.actor(states)
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        discounts = self.gamma ** torch.arange(len(rewards), dtype=torch.float32)
        weighted_advantages = discounts * advantages
        selected_log_probs = log_probs.gather(1, actions).view(-1)
        score = torch.dot(selected_log_probs, weighted_advantages).unsqueeze(0)

        return score

    def train_reset(self):
        self.current_episode = []
        self.episode_reward = 0
        self.scores = []

    def update_critic(self, transition):
        state, _, reward, terminated, next_state = transition

        values = self.critic.forward(state)
        with torch.no_grad():
            next_state_values = (1 - terminated) * self.critic(next_state)
            targets = next_state_values * self.gamma + reward

        loss = self.loss_function(values, targets)

        self.critic_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.critic.parameters(), 5)
        self.critic_optimizer.step()

    def update(self, state, action, reward, terminated, truncated, next_state):
        transition = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[action]], dtype=torch.int64),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([float(terminated)], dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
        )
        self.current_episode.append(transition)

        self.total_steps += 1
        self.episode_reward += reward

        self.update_critic(transition)

        if terminated or truncated:
            self.episode_reward = 0
            self.n_eps += 1

            self.scores.append(self.compute_gradient_score())
            self.current_episode = []

            self.actor_optimizer.zero_grad()
            full_neg_score = -torch.cat(self.scores).sum()
            full_neg_score.backward()
            self.actor_optimizer.step()

            self.scores = []

            return full_neg_score.detach().numpy()
        
        return None


class ActorCriticGAE(ActorCriticBasic):
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        actor_learning_rate,
        critic_learning_rate,
        lambda_,
    ):
        super().__init__(
            action_space,
            observation_space,
            gamma,
            actor_learning_rate,
            critic_learning_rate,
        )
        self.lambda_ = lambda_

    def compute_GAE(self, rewards, terminateds, advantages):
        GAE = 0
        GAE_list = []
        for t in reversed(range(len(rewards))):
            GAE = (1 - terminateds[t]) * GAE
            GAE = advantages[t] + self.gamma * self.lambda_ * GAE
            GAE_list.append(GAE)
        return torch.tensor(GAE_list[::-1], dtype=torch.float32)

    def compute_gradient_score(self):
        states, actions, rewards, terminateds, next_states = tuple(
            [torch.cat(data) for data in zip(*self.current_episode)]
        )

        with torch.no_grad():
            target_values = (
                rewards
                + self.gamma * (1 - terminateds) * self.critic(next_states).squeeze()
            )
            values = self.critic(states).squeeze()
            advantages = target_values - values
        GAEs = self.compute_GAE(rewards, terminateds, advantages)

        logits = self.actor(states)
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        selected_log_probs = log_probs.gather(1, actions).view(-1)
        GAEs = GAEs.view(-1)
        return torch.dot(selected_log_probs, GAEs).unsqueeze(0)