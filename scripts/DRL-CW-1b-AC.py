# %% [markdown]
# # DRL Coursework 1b: Combined Actor-Critic Variant
# Alternate Actor-Critic version using a combined actor-critic network.
# 

# %% [markdown]
# ## Imports
# Libraries are grouped first so every later cell has the same dependencies available.
# 

# %%
import random
import torch
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


# %% [markdown]
# ## Reproducibility setup
# A fixed seed makes random, NumPy, and PyTorch sampling easier to reproduce.
# 

# %%
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %% [markdown]
# ## Supporting code
# This code supports the surrounding experiment section.
# 

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% [markdown]
# ## Environment definition
# The hospital environment stores bed availability, patient type, actions, rewards, and transitions.
# 

# %%
class HospitalEnv:
    def __init__(self):
        # ICU free: 0,1,2
        # GEN free: 0,1,2,3
        # Patient type: 0=emergency, 1=urgent, 2=elective
        self.reset()

    def reset(self):
        self.icu_free = np.random.randint(0, 3)
        self.gen_free = np.random.randint(0, 4)
        self.ptype = np.random.choice([0,1,2])
        return self._get_state()

    def _get_state(self):
        return (self.icu_free, self.gen_free, self.ptype)

    def step(self, action):
        reward = 0
        icu, gen, p = self.icu_free, self.gen_free, self.ptype

        # APPLY ACTION
        if action == 0:  # allocate ICU
            if icu > 0:
                self.icu_free -= 1
                if p in [0,1]: reward += 5
                else: reward -= 3
            else:
                reward -= 10

        elif action == 1:  # allocate general
            if gen > 0:
                self.gen_free -= 1
                if p == 2: reward += 5
                elif p in [0,1]:
                    if icu > 0: reward -= 3
                    else: reward += 2
            else:
                reward -= 10

        elif action == 2:  # delay
            reward -= 1
            if p == 0:
                reward -= 10

        # DISCHARGES
        if random.random() < 0.3:
            self.icu_free = min(self.icu_free + 1, 2)
        if random.random() < 0.4:
            self.gen_free = min(self.gen_free + 1, 3)

        # EMERGENCY REJECTION LOGIC
        emergency_rejected = 0
        if p == 0:  # emergency patient
            if action == 0 and icu == 0:
                emergency_rejected = 1
            elif action == 1 and gen == 0:
                emergency_rejected = 1
            elif action == 2:
                emergency_rejected = 1

        # ICU utilisation: 1 if at least one ICU bed is occupied
        icu_occupied = 1 if self.icu_free < 2 else 0

        # NEW PATIENT ARRIVAL
        self.ptype = np.random.choice([0,1,2], p=[0.2,0.3,0.5])

        next_state = self._get_state()
        done = False
        return next_state, reward, done, emergency_rejected, icu_occupied


# %% [markdown]
# ## Tabular Q-learning
# These helpers map states to Q-table rows and update values with epsilon-greedy exploration.
# 

# %%
def state_to_index(s):
    icu, gen, p = s
    return icu*12 + gen*3 + p  # 36 states


# %% [markdown]
# ## Tabular Q-learning
# These helpers map states to Q-table rows and update values with epsilon-greedy exploration.
# 

# %%
def run_q_learning(episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.2):
    env = HospitalEnv()
    Q = np.zeros((36, 3))

    rewards_per_episode = []
    emergency_rejections = []
    icu_utilisation = []

    for ep in range(episodes):
        s = env.reset()
        total_reward = 0
        episode_rejections = 0
        episode_icu_occ = 0

        for t in range(50):
            s_idx = state_to_index(s)

            # epsilon-greedy
            if random.random() < epsilon:
                a = np.random.randint(0, 3)
            else:
                a = np.argmax(Q[s_idx])

            s_next, r, done, rej, icu_occ = env.step(a)
            s_next_idx = state_to_index(s_next)

            # Q update
            Q[s_idx, a] += alpha * (r + gamma * np.max(Q[s_next_idx]) - Q[s_idx, a])

            s = s_next
            total_reward += r
            episode_rejections += rej
            episode_icu_occ += icu_occ

        rewards_per_episode.append(total_reward)
        emergency_rejections.append(episode_rejections)
        icu_utilisation.append(episode_icu_occ / 50.0)

    return Q, rewards_per_episode, emergency_rejections, icu_utilisation


# %% [markdown]
# ## State vector representation
# Deep RL models use numeric state vectors instead of integer Q-table indices.
# 

# %%
def state_to_vector(s):
    icu, gen, p = s
    p_onehot = np.zeros(3)
    p_onehot[p] = 1.0
    return np.array([icu, gen] + p_onehot.tolist(), dtype=np.float32)


# %% [markdown]
# ## Replay buffer
# Experience replay stores transitions so DQN updates can sample mixed past experience.
# 

# %%
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_states, dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


# %% [markdown]
# ## DQN network models
# Standard and dueling networks estimate action values for each state.
# 

# %%
class DQN(nn.Module):
    def __init__(self, state_dim=5, action_dim=3, hidden_dim=64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# %% [markdown]
# ## DQN network models
# Standard and dueling networks estimate action values for each state.
# 

# %%
class DuelingDQN(nn.Module):
    def __init__(self, state_dim=5, action_dim=3, hidden_dim=64):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_stream(feat)
        adv = self.adv_stream(feat)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q


# %% [markdown]
# ## DQN training loop
# The training loop supports standard DQN, Double DQN, and Dueling Double DQN.
# 

# %%
def train_dqn(
    episodes=1000,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_capacity=10000,
    min_buffer_size=1000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=500,
    target_update_freq=50,
    double_dqn=True,
    dueling=False
):
    env = HospitalEnv()
    state_dim = 5
    action_dim = 3

    if dueling:
        policy_net = DuelingDQN(state_dim, action_dim).to(device)
        target_net = DuelingDQN(state_dim, action_dim).to(device)
    else:
        policy_net = DQN(state_dim, action_dim).to(device)
        target_net = DQN(state_dim, action_dim).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    rewards_per_episode = []
    steps_done = 0

    for ep in range(episodes):
        s = env.reset()
        s_vec = state_to_vector(s)
        total_reward = 0

        for t in range(50):
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-1.0 * steps_done / epsilon_decay)
            steps_done += 1

            if random.random() < epsilon:
                a = np.random.randint(0, action_dim)
            else:
                with torch.no_grad():
                    state_t = torch.tensor([s_vec], device=device)
                    q_values = policy_net(state_t)
                    a = q_values.argmax(dim=1).item()

            s_next, r, done, _, _ = env.step(a)
            s_next_vec = state_to_vector(s_next)

            replay_buffer.push(s_vec, a, r, s_next_vec, done)

            s_vec = s_next_vec
            total_reward += r

            if len(replay_buffer) >= min_buffer_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if double_dqn:
                        next_q_policy = policy_net(next_states)
                        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                        next_q_target = target_net(next_states).gather(1, next_actions).squeeze(1)
                    else:
                        next_q_target = target_net(next_states).max(dim=1)[0]

                    targets = rewards + gamma * (1 - dones) * next_q_target

                loss = nn.MSELoss()(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        rewards_per_episode.append(total_reward)

        if ep % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return rewards_per_episode


# %% [markdown]
# ## Actor-Critic networks
# Actor-Critic models learn a policy and a value estimate instead of only Q-values.
# 

# %%
class ActorCritic(nn.Module):
    def __init__(self, state_dim=5, action_dim=3, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        feat = self.shared(x)
        logits = self.actor(feat)
        value = self.critic(feat)
        return logits, value


# %% [markdown]
# ## Actor-Critic training loop
# This loop updates policy and value losses from simulated hospital episodes.
# 

# %%
def train_actor_critic(
    episodes=1000,
    gamma=0.99,
    lr=1e-3,
    entropy_coef=0.01
):
    env = HospitalEnv()
    state_dim = 5
    action_dim = 3

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    rewards_per_episode = []
    rejections_per_episode = []
    icu_utilisation_per_episode = []

    for ep in range(episodes):
        s = env.reset()
        s_vec = state_to_vector(s)

        total_reward = 0
        total_rejections = 0
        total_icu_occ = 0

        log_probs = []
        values = []
        rewards = []
        entropies = []

        for t in range(50):
            state_t = torch.tensor([s_vec], dtype=torch.float32, device=device)
            logits, value = model(state_t)

            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            s_next, r, done, rej, icu_occ = env.step(action.item())
            s_next_vec = state_to_vector(s_next)

            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(torch.tensor(r, dtype=torch.float32, device=device))
            entropies.append(entropy)

            total_reward += r
            total_rejections += rej
            total_icu_occ += icu_occ

            s_vec = s_next_vec

            if done:
                break

        returns = []
        G = torch.tensor(0.0, device=device)
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.stack(returns)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        advantages = returns - values.squeeze(-1)

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -entropies.mean()

        loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_per_episode.append(total_reward)
        rejections_per_episode.append(total_rejections)
        icu_utilisation_per_episode.append(total_icu_occ / 50.0)

    return rewards_per_episode, rejections_per_episode, icu_utilisation_per_episode


# %% [markdown]
# ## Q-learning experiment
# This cell runs the tabular baseline and plots reward, rejection, or utilisation traces.
# 

# %%
# 5. RUN EXPERIMENTS

# Q-learning
Q, rewards_q, rejections_q, icu_q = run_q_learning(episodes=1500)

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.plot(rewards_q)
plt.title("Q-Learning: Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(rejections_q)
plt.title("Q-Learning: Emergency Rejections")
plt.xlabel("Episode")
plt.ylabel("Rejections")
plt.grid(True)

plt.subplot(1,3,3)
plt.plot(icu_q)
plt.title("Q-Learning: ICU Utilisation")
plt.xlabel("Episode")
plt.ylabel("Utilisation")
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Deep RL experiment runs
# This cell trains the configured deep RL variants for comparison.
# 

# %%
# DQN variants
rewards_dqn = train_dqn(
    episodes=1500,
    double_dqn=False,
    dueling=False
)

rewards_double = train_dqn(
    episodes=1500,
    double_dqn=True,
    dueling=False
)

rewards_dueling = train_dqn(
    episodes=1500,
    double_dqn=True,
    dueling=True
)

# Actor-Critic
rewards_ac, rejections_ac, icu_ac = train_actor_critic(episodes=1500)


# %% [markdown]
# ## Result plots
# Plots compare reward, emergency rejection, and ICU utilisation trends across methods.
# 

# %%
# 6. PLOTS: COMPARISONS

# Reward comparison across DRL algorithms
plt.figure(figsize=(10,5))
plt.plot(rewards_dqn, label="DQN")
plt.plot(rewards_double, label="Double DQN")
plt.plot(rewards_dueling, label="Dueling Double DQN")
plt.plot(rewards_ac, label="Actor-Critic")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DRL Algorithms on Hospital Bed Allocation")
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## Result plots
# Plots compare reward, emergency rejection, and ICU utilisation trends across methods.
# 

# %%
# Actor-Critic detailed metrics
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.plot(rewards_ac)
plt.title("Actor-Critic: Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(rejections_ac)
plt.title("Actor-Critic: Emergency Rejections")
plt.xlabel("Episode")
plt.ylabel("Rejections")
plt.grid(True)

plt.subplot(1,3,3)
plt.plot(icu_ac)
plt.title("Actor-Critic: ICU Utilisation")
plt.xlabel("Episode")
plt.ylabel("Utilisation")
plt.grid(True)

plt.tight_layout()
plt.show()



