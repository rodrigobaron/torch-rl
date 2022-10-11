import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .buffers import ReplayBuffer


def get_optimizer(params, **kwargs):
    return optim.Adam(params, **kwargs)


def get_replay_buffer(buffer_size, env, device, **kwargs):
    return ReplayBuffer(
        buffer_size,
        env.single_observation_space,
        env.single_action_space,
        device,
        **kwargs
    )


class DQN(nn.Module):
    def __init__(self, model_func, device="cpu"):
        super().__init__()
        self.q_network = model_func().to(device)
        self.target_network = model_func().to(device)

        self.update_target_network()

    def get_q_values(self, obs):
        return self.q_network(obs)

    def get_target_values(self, obs):
        return self.target_network(obs)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def forward(self, obs):
        q_values = self.get_q_values(obs)
        return torch.argmax(q_values, dim=1)


class DQNTrainingWrapper(nn.Module):
    def __init__(self, model, env, device):
        super().__init__()
        self.device = device
        self.dqn = model.to(self.device)
        self.env = env

    def get_optimizer(self, **kwargs):
        return get_optimizer(self.dqn.q_network.parameters(), **kwargs)

    def get_replay_buffer(self, buffer_size, **kwargs):
        return get_replay_buffer(buffer_size, self.env, self.device, **kwargs)

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def train_step(self, data, gamma):
        with torch.no_grad():
            target_max, _ = self.dqn.get_target_values(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
        old_val = self.dqn.get_q_values(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        return old_val, loss

    def forward(self):
        raise NotImplementedError()
