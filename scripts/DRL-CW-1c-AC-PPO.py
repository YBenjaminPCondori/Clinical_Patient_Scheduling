# %% [markdown]
# # DRL Coursework 1b: PPO Actor-Critic Comparison
# This version keeps the old Part 1b bed-allocation environment and uses PPO as the actor-critic method.
# 

# %% [markdown]
# ## Imports
# Libraries are grouped first so every later cell has the same dependencies available.
# 

# %%
import random
from collections import deque
from itertools import product

import gymnasium as gym
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

ACTION_MEANINGS = {
    0: "allocate_icu",
    1: "allocate_general",
    2: "delay_patient",
}


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
# ## Q-learning experiment
# This cell runs the tabular baseline and plots reward, rejection, or utilisation traces.
# 

# %%
# 3. RUN EXPERIMENT + PLOTS

Q, rewards, rejections, icu_util = run_q_learning(episodes=1500)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.plot(rewards)
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(rejections)
plt.title("Emergency Rejections per Episode")
plt.xlabel("Episode")
plt.ylabel("Rejections")
plt.grid(True)

plt.subplot(1,3,3)
plt.plot(icu_util)
plt.title("ICU Utilisation Rate")
plt.xlabel("Episode")
plt.ylabel("Utilisation")
plt.grid(True)

plt.tight_layout()
plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        super().__init__()
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
        super().__init__()
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
        return value + adv - adv.mean(dim=1, keepdim=True)


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
    dueling=False,
    icu_capacity=2
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
    emergency_rej_per_episode = []
    icu_util_per_episode = []

    steps_done = 0

    for ep in range(episodes):
        s = env.reset()
        s_vec = state_to_vector(s)
        total_reward = 0

        emergency_rej = 0
        icu_util_sum = 0.0
        step_count = 0

        for t in range(50):
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-steps_done / epsilon_decay)
            steps_done += 1

            if random.random() < epsilon:
                a = np.random.randint(0, action_dim)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor([s_vec], device=device))
                    a = q_values.argmax(dim=1).item()

            s_next, r, done, rej, icu_occ = env.step(a)

            emergency_rej += rej
            icu_util_sum += icu_occ
            step_count += 1

            s_next_vec = state_to_vector(s_next)

            replay_buffer.push(s_vec, a, r, s_next_vec, done)

            s_vec = s_next_vec
            s = s_next
            total_reward += r

            if len(replay_buffer) >= min_buffer_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    if double_dqn:
                        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
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
        emergency_rej_per_episode.append(emergency_rej)
        icu_util_per_episode.append(icu_util_sum / step_count)

        if ep % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return rewards_per_episode, emergency_rej_per_episode, icu_util_per_episode


# %% [markdown]
# ## Actor-Critic networks
# Actor-Critic models learn a policy and a value estimate instead of only Q-values.
# 

# %%
class Actor(nn.Module):
    def __init__(self, state_dim=5, action_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        logits = self.policy_head(self.net(x))
        return torch.softmax(logits, dim=-1)


# %% [markdown]
# ## Actor-Critic networks
# Actor-Critic models learn a policy and a value estimate instead of only Q-values.
# 

# %%
class Critic(nn.Module):
    def __init__(self, state_dim=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# %% [markdown]
# ## Gymnasium Adapter for Notebook Environment
# This adapter uses the `HospitalEnv` defined above so PPO does not rely on an external environment script.

# %%
class HospitalGymWrapper(gym.Env):
    """Gymnasium adapter around the notebook original HospitalEnv."""

    metadata = {"name": "HospitalGymWrapper"}

    def __init__(self, max_steps=50):
        super().__init__()
        self.max_steps = int(max_steps)
        self.env = HospitalEnv()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.asarray([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.asarray([2, 3, 2, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.current_state = None
        self.timestep = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.env = HospitalEnv()
        self.current_state = self.env.reset()
        self.timestep = 0
        return self._get_observation(), self._get_info()

    def step(self, action):
        action = int(np.clip(action, 0, self.action_space.n - 1))
        self.timestep += 1

        next_state, reward, done, emergency_rejected, icu_occupied = self.env.step(action)
        self.current_state = next_state

        terminated = bool(done)
        truncated = self.timestep >= self.max_steps
        info = self._get_info(
            action=action,
            reward=reward,
            emergency_rejected=emergency_rejected,
            icu_occupied=icu_occupied,
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _get_observation(self):
        return state_to_vector(self.current_state).astype(np.float32)

    def _get_info(self, action=None, reward=0.0, emergency_rejected=0, icu_occupied=0):
        icu_free, gen_free, patient_type = self.current_state
        return {
            "timestep": self.timestep,
            "action": action,
            "action_meaning": ACTION_MEANINGS.get(action) if action is not None else None,
            "reward": float(reward),
            "icu_free": int(icu_free),
            "general_free": int(gen_free),
            "patient_type": int(patient_type),
            "emergency_rejection": int(emergency_rejected),
            "icu_occupied": int(icu_occupied),
            "icu_utilisation": float(icu_occupied),
        }


# %% [markdown]
# ## PPO Actor-Critic Setup
# PPO is used here as the actor-critic method while keeping the old Part 1b environment logic.

# %%
learning_rates = [1e-4, 3e-4, 1e-3]
batch_sizes = [32, 64, 128]
n_steps_values = [128, 256, 512]

PPO_FIXED_PARAMS = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "n_epochs": 25,
}


def make_ppo_old_env(max_steps=50):
    return Monitor(HospitalGymWrapper(max_steps=max_steps))


def train_ppo(learning_rate, batch_size, n_steps, total_timesteps=30000, seed=42, max_steps=50):
    # Train one PPO model for a single grid-search configuration.
    train_env = make_ppo_old_env(max_steps=max_steps)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=PPO_FIXED_PARAMS["n_epochs"],
        gamma=PPO_FIXED_PARAMS["gamma"],
        gae_lambda=PPO_FIXED_PARAMS["gae_lambda"],
        clip_range=PPO_FIXED_PARAMS["clip_range"],
        ent_coef=PPO_FIXED_PARAMS["ent_coef"],
        verbose=0,
        seed=seed,
        device=device,
    )

    model.learn(total_timesteps=total_timesteps)
    return model


def train_ppo_actor_critic(total_timesteps=75000, seed=42, max_steps=50):
    # Backwards-compatible helper using the best default candidate.
    return train_ppo(1e-4, 64, 256, total_timesteps=total_timesteps, seed=seed, max_steps=max_steps)


# %% [markdown]
# ## PPO Evaluation Helpers
# These helpers match the existing reward, emergency rejection, and ICU utilisation outputs.

# %%
def evaluate_ppo(model, episodes=100, seed=42, max_steps=50):
    # Evaluate PPO using reward, safety, utilisation, delay, and action metrics.
    env = HospitalGymWrapper(max_steps=max_steps)
    rows = []
    action_counts = {action: 0 for action in ACTION_MEANINGS}

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        emergency_rejections = 0
        icu_values = []
        delayed_patients = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            emergency_rejections += int(info.get("emergency_rejection", 0))
            icu_values.append(float(info.get("icu_occupied", 0)))
            delayed_patients += int(action == 2)
            steps += 1

        rows.append({
            "episode": ep,
            "total_reward": total_reward,
            "emergency_rejections": emergency_rejections,
            "icu_utilisation": float(np.mean(icu_values)) if icu_values else 0.0,
            "delayed_patients": delayed_patients,
            "steps": steps,
        })

    results_df = pd.DataFrame(rows)
    action_counts_df = pd.DataFrame({
        "action": list(action_counts.keys()),
        "action_name": [ACTION_MEANINGS[action] for action in action_counts],
        "count": list(action_counts.values()),
    })

    summary = {
        "mean_reward": results_df["total_reward"].mean(),
        "std_reward": results_df["total_reward"].std(),
        "emergency_rejections": results_df["emergency_rejections"].mean(),
        "icu_utilisation": results_df["icu_utilisation"].mean(),
        "delayed_patients": results_df["delayed_patients"].mean(),
    }
    return summary, results_df, action_counts_df


def evaluate_ppo_actor_critic(model, episodes=1500, seed=42, max_steps=50):
    # Backwards-compatible output format for the original comparison plots.
    summary, results_df, action_counts_df = evaluate_ppo(model, episodes=episodes, seed=seed, max_steps=max_steps)
    return (
        results_df["total_reward"].tolist(),
        results_df["emergency_rejections"].tolist(),
        results_df["icu_utilisation"].tolist(),
        action_counts_df,
    )


def evaluate_random_old_env(episodes=1500, seed=42, max_steps=50):
    rng = np.random.default_rng(seed)
    env = HospitalGymWrapper(max_steps=max_steps)
    rewards_per_episode = []
    emergency_rej_per_episode = []
    icu_util_per_episode = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        emergency_rej = 0
        icu_util_sum = 0.0
        step_count = 0

        while not done:
            action = int(rng.integers(env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            emergency_rej += int(info.get("emergency_rejection", 0))
            icu_util_sum += float(info.get("icu_occupied", 0))
            step_count += 1

        rewards_per_episode.append(total_reward)
        emergency_rej_per_episode.append(emergency_rej)
        icu_util_per_episode.append(icu_util_sum / step_count if step_count else 0.0)

    return rewards_per_episode, emergency_rej_per_episode, icu_util_per_episode


# %% [markdown]
# ## PPO Grid Search Explanation
# Grid search trains every specified parameter combination and ranks them by validation metrics.

# %% [markdown]
# The tuned parameters control update size and rollout batching; `gamma` and `gae_lambda` are kept fixed to preserve the reward horizon and advantage estimator.

# %%
def valid_grid_combinations(learning_rates, batch_sizes, n_steps_values):
    # Keep rollout batches compatible with Stable-Baselines3 PPO.
    for learning_rate, batch_size, n_steps in product(learning_rates, batch_sizes, n_steps_values):
        if batch_size <= n_steps and n_steps % batch_size == 0:
            yield learning_rate, batch_size, n_steps


def run_grid_search(
    total_timesteps=30000,
    eval_episodes=100,
    seed=42,
    max_steps=50,
    output_csv="ppo_1b_grid_search_results.csv",
):
    # Train and evaluate every PPO hyperparameter combination.
    rows = []
    models = {}
    action_count_frames = {}
    combinations = list(valid_grid_combinations(learning_rates, batch_sizes, n_steps_values))

    for idx, (learning_rate, batch_size, n_steps) in enumerate(tqdm(combinations, desc="PPO grid search")):
        run_seed = seed + idx
        model = train_ppo(
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            total_timesteps=total_timesteps,
            seed=run_seed,
            max_steps=max_steps,
        )
        summary, episode_df, action_counts_df = evaluate_ppo(
            model,
            episodes=eval_episodes,
            seed=run_seed,
            max_steps=max_steps,
        )

        combo_label = f"lr={learning_rate:g}, batch={batch_size}, steps={n_steps}"
        rows.append({
            "combination": combo_label,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_steps": n_steps,
            "seed": run_seed,
            **summary,
        })
        models[combo_label] = model
        action_count_frames[combo_label] = action_counts_df

    results_df = pd.DataFrame(rows).sort_values(
        by=["mean_reward", "emergency_rejections"],
        ascending=[False, True],
    ).reset_index(drop=True)
    results_df["rank"] = np.arange(1, len(results_df) + 1)
    results_df.to_csv(output_csv, index=False)

    best_row = results_df.iloc[0]
    best_label = best_row["combination"]
    print("Best PPO configuration:")
    print(best_row[["learning_rate", "batch_size", "n_steps", "mean_reward", "emergency_rejections"]])

    return results_df, models[best_label], best_row.to_dict(), action_count_frames[best_label]


# %% [markdown]
# ## Grid Search Plotting
# These report-ready figures compare reward and emergency rejection outcomes across combinations.

# %%
def plot_results(grid_results_df):
    ordered = grid_results_df.sort_values("rank")

    plt.figure(figsize=(12, 5))
    plt.bar(ordered["combination"], ordered["mean_reward"], yerr=ordered["std_reward"], capsize=3)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean episode reward")
    plt.title("PPO Grid Search: Mean Reward by Hyperparameter Combination")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.bar(ordered["combination"], ordered["emergency_rejections"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean emergency rejections")
    plt.title("PPO Grid Search: Emergency Rejections by Hyperparameter Combination")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Actor-Critic training loop
# This loop updates policy and value losses from simulated hospital episodes.
# 

# %%
rewards_dqn, emrej_dqn, icu_dqn = train_dqn(
    episodes=1500,
    double_dqn=False,
    dueling=False
)

rewards_double, emrej_double, icu_double = train_dqn(
    episodes=1500,
    double_dqn=True,
    dueling=False
)

rewards_dueling, emrej_dueling, icu_dueling = train_dqn(
    episodes=1500,
    double_dqn=True,
    dueling=True
)

grid_results_df, ppo_model, best_ppo_params, ppo_action_counts_df = run_grid_search(
    total_timesteps=30000,
    eval_episodes=100,
    seed=SEED,
    max_steps=50,
)

plot_results(grid_results_df)

rewards_ppo, emrej_ppo, icu_ppo, ppo_action_counts_df = evaluate_ppo_actor_critic(
    ppo_model,
    episodes=1500,
    seed=SEED,
    max_steps=50,
)

rewards_random, emrej_random, icu_random = evaluate_random_old_env(
    episodes=1500,
    seed=SEED,
    max_steps=50,
)


# %% [markdown]
# ## Result plots
# Plots compare reward, emergency rejection, and ICU utilisation trends across methods.
# 

# %%
plt.figure(figsize=(10,5))
plt.plot(rewards_dueling, label="Dueling Double DQN")
plt.plot(rewards_ppo, label="PPO Actor-Critic", linewidth=2.5)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Comparison")
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## Result plots
# Plots compare reward, emergency rejection, and ICU utilisation trends across methods.
# 

# %%
plt.figure(figsize=(10,5))
plt.plot(emrej_dueling, label="Dueling Double DQN")
plt.plot(emrej_ppo, label="PPO Actor-Critic", linewidth=2.5)
plt.xlabel("Episode")
plt.ylabel("Emergency Rejections")
plt.title("Emergency Rejections per Episode")
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## Result plots
# Plots compare reward, emergency rejection, and ICU utilisation trends across methods.
# 

# %%
plt.figure(figsize=(10,5))
plt.plot(icu_dqn, label="DQN")
plt.plot(icu_double, label="Double DQN")
plt.plot(icu_dueling, label="Dueling Double DQN")
plt.plot(icu_ppo, label="PPO Actor-Critic", linewidth=2.5)
plt.xlabel("Episode")
plt.ylabel("ICU Utilisation")
plt.title("ICU Utilisation per Episode")
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# ## PPO Results Table
# This converts the PPO and random baseline curves into Part C style summary tables.

# %%
ppo_1b_results = pd.DataFrame({
    "episode": np.arange(len(rewards_ppo)),
    "total_reward": rewards_ppo,
    "emergency_rejections": emrej_ppo,
    "icu_utilisation": icu_ppo,
})

random_1b_results = pd.DataFrame({
    "episode": np.arange(len(rewards_random)),
    "total_reward": rewards_random,
    "emergency_rejections": emrej_random,
    "icu_utilisation": icu_random,
})

ppo_1b_summary = pd.DataFrame([
    {
        "metric": metric,
        "ppo_mean": ppo_1b_results[metric].mean(),
        "ppo_std": ppo_1b_results[metric].std(),
        "random_mean": random_1b_results[metric].mean(),
        "random_std": random_1b_results[metric].std(),
    }
    for metric in ["total_reward", "emergency_rejections", "icu_utilisation"]
])

print("Best PPO hyperparameters:")
print(pd.Series(best_ppo_params))
ppo_1b_summary


# %% [markdown]
# ## PPO versus Random Baseline
# These plots give the Part C style PPO/random comparison for the old Part 1b environment.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(ppo_1b_results["episode"], ppo_1b_results["total_reward"], label="PPO")
axes[0].plot(random_1b_results["episode"], random_1b_results["total_reward"], label="Random", alpha=0.7)
axes[0].set_title("Episode Reward")
axes[0].set_xlabel("Episode")
axes[0].legend()

axes[1].plot(ppo_1b_results["episode"], ppo_1b_results["emergency_rejections"], label="PPO")
axes[1].plot(random_1b_results["episode"], random_1b_results["emergency_rejections"], label="Random", alpha=0.7)
axes[1].set_title("Emergency Rejections")
axes[1].set_xlabel("Episode")
axes[1].legend()

axes[2].plot(ppo_1b_results["episode"], ppo_1b_results["icu_utilisation"], label="PPO")
axes[2].plot(random_1b_results["episode"], random_1b_results["icu_utilisation"], label="Random", alpha=0.7)
axes[2].set_title("ICU Utilisation")
axes[2].set_xlabel("Episode")
axes[2].legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# ## PPO Action Frequencies
# This checks whether PPO mainly allocates ICU, allocates general beds, or delays patients.

# %%
plt.figure(figsize=(6, 4))
plt.bar(ppo_action_counts_df["action_name"], ppo_action_counts_df["count"])
plt.title("PPO Actor-Critic Action Frequencies")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

ppo_action_counts_df


# %% [markdown]
# ## Save PPO 1b Outputs
# These files use 1b-specific names so they do not overwrite Part C PPO outputs.

# %%
ppo_model.save("ppo_1b_actor_critic_model")
ppo_1b_results.to_csv("ppo_1b_actor_critic_results.csv", index=False)
random_1b_results.to_csv("random_1b_baseline_results.csv", index=False)
ppo_1b_summary.to_csv("ppo_1b_actor_critic_summary.csv", index=False)
ppo_action_counts_df.to_csv("ppo_1b_action_counts.csv", index=False)
grid_results_df.to_csv("ppo_1b_grid_search_results.csv", index=False)

print("Saved PPO 1b model, grid search, and result CSV files.")



