from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ModuleNotFoundError:
    torch = None
    nn = None
    optim = None


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for OpenArmX policy training/evaluation. "
            "Install torch in the robosuit conda environment first."
        )
    return torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass
class DDPGConfig:
    seed: int = 42
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    buffer_size: int = 200_000
    batch_size: int = 128
    hidden_dim: int = 256
    pretrain_steps: int = 200
    total_steps: int = 1000
    exploration_noise: float = 0.1
    bc_weight_start: float = 1.0
    bc_weight_end: float = 0.2
    actor_q_weight: float = 0.005
    reward_scale: float = 1.0


class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = int(buffer_size)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.a = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.r = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.s_n = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.done = np.zeros((self.buffer_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.s[self.ptr] = np.asarray(state, dtype=np.float32)
        self.a[self.ptr] = np.asarray(action, dtype=np.float32)
        self.r[self.ptr, 0] = float(reward)
        self.s_n[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.done[self.ptr, 0] = float(bool(done))
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        index = np.random.randint(0, self.size, int(batch_size))
        return self.s[index], self.a[index], self.r[index], self.s_n[index], self.done[index]

    def load_dataset(self, data_dir, reward_scale=1.0):
        loaded = 0
        pattern = os.path.join(str(data_dir), "*.npz")
        for file_name in sorted(glob.glob(pattern)):
            data = np.load(file_name)
            obs = data["obs"]
            actions = data["actions"]
            rewards = data["rewards"]
            next_obs = data["next_obs"]
            dones = data["dones"]
            for i in range(len(obs)):
                self.add(
                    obs[i],
                    actions[i],
                    float(rewards[i]) * float(reward_scale),
                    next_obs[i],
                    bool(dones[i]),
                )
                loaded += 1
                if self.size >= self.buffer_size:
                    return loaded
        return loaded


if nn is not None:

    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )

        def forward(self, state):
            return self.net(state)


    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state, action):
            return self.net(torch.cat([state, action], dim=-1))

else:

    class Actor:
        def __init__(self, *args, **kwargs):
            require_torch()


    class Critic:
        def __init__(self, *args, **kwargs):
            require_torch()


class DDPGAgent:
    def __init__(self, state_dim, action_dim, config=None, device=None):
        torch_module = require_torch()
        self.cfg = config or DDPGConfig()
        self.device = device or torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )
        self.rb = ReplayBuffer(self.cfg.buffer_size, state_dim, action_dim)

        self.actor = Actor(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

    def act(self, state, noise=0.0):
        torch_module = require_torch()
        state_t = torch_module.tensor(
            state, dtype=torch_module.float32, device=self.device
        ).unsqueeze(0)
        with torch_module.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if noise > 0:
            action = action + np.random.normal(0.0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, use_bc_loss=True, bc_weight=0.5):
        torch_module = require_torch()
        if self.rb.size < self.cfg.batch_size:
            return None

        s, a, r, sn, d = self.rb.sample(self.cfg.batch_size)
        s = torch_module.tensor(s, dtype=torch_module.float32, device=self.device)
        a_real = torch_module.tensor(a, dtype=torch_module.float32, device=self.device)
        r = torch_module.tensor(r, dtype=torch_module.float32, device=self.device)
        sn = torch_module.tensor(sn, dtype=torch_module.float32, device=self.device)
        d = torch_module.tensor(d, dtype=torch_module.float32, device=self.device)

        with torch_module.no_grad():
            next_a = self.actor_target(sn)
            target_q = self.critic_target(sn, next_a)
            y = r + self.cfg.gamma * (1.0 - d) * target_q

        current_q = self.critic(s, a_real)
        critic_loss = nn.MSELoss()(current_q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a_pred = self.actor(s)
        q_loss = -self.critic(s, a_pred).mean()
        bc_loss = nn.MSELoss()(a_pred, a_real)
        if use_bc_loss:
            bc_weight = float(np.clip(bc_weight, 0.0, 1.0))
            actor_loss = (1.0 - bc_weight) * self.cfg.actor_q_weight * q_loss + bc_weight * bc_loss
        else:
            actor_loss = q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_loss": float(q_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "bc_weight": float(bc_weight),
        }

    def _soft_update(self, source, target):
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(
                    self.cfg.tau * source_param.data
                    + (1.0 - self.cfg.tau) * target_param.data
                )

    def save_actor(self, path):
        require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load_actor(self, path):
        require_torch()
        state_dict = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.actor_target.load_state_dict(self.actor.state_dict())
