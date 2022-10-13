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


class C51(nn.Module):
    def __init__(self, model_func, device="cpu"):
        super().__init__()
        self.q_network = model_func().to(device)
        self.target_network = model_func().to(device)

        self.update_target_network()

    def get_q_action(self, obs, action=None):
        return self.q_network.get_action(obs, action)

    def get_q_atoms(self):
        return self.q_network.atoms

    def get_target_action(self, obs, action=None):
        return self.target_network.get_action(obs, action=None)

    def get_target_atoms(self):
        return self.target_network.atoms

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def forward(self, obs):
        return self.get_q_action(obs)


class C51TrainingWrapper(nn.Module):
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

    def train_step(self, data, gamma, n_atoms, v_min, v_max):
        with torch.no_grad():
            _, next_pmfs = self.dqn.get_target_action(data.next_observations)
            next_atoms = data.rewards + gamma * self.dqn.get_target_atoms() * (1 - data.dones)
            # projection
            delta_z = self.dqn.get_target_atoms()[1] - self.dqn.get_target_atoms()[0]
            tz = next_atoms.clamp(v_min, v_max)

            b = (tz - v_min) / delta_z
            l = b.floor().clamp(0, n_atoms - 1)
            u = b.ceil().clamp(0, n_atoms - 1)
            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = torch.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        _, old_pmfs = self.dqn.get_q_action(data.observations, data.actions.flatten())
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
        old_val = (old_pmfs * self.dqn.get_q_atoms()).sum(1)

        return old_val, loss

    def forward(self):
        raise NotImplementedError()
