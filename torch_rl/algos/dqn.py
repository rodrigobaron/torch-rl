import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import random
import tqdm

import gym

from dataclasses import dataclass


@dataclass
class DQNTrainingArgs:
    total_timesteps: int
    learning_starts: int
    train_frequency: int
    target_network_frequency: int
    batch_size: int
    gamma: int


class DiscreteDQNModel(nn.Module):
    def __init__(self, env, encoder):
        super().__init__()
        self.network = nn.Sequential(
            encoder,
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class DiscreteDQN():
    def __init__(self, encoder, optim, optim_args, scheduler, buffer):
        self.encoder = encoder
        self.optim_func = optim
        self.optim_args = optim_args
        self.scheduler = scheduler
        self.buffer = buffer

        self.q_network = None
        self.target_network = None
        self.optimizer = None


    def initialize(self, env, device):
        self.q_network = DiscreteDQNModel(env, self.encoder).to(device)

        self.target_network = DiscreteDQNModel(env, self.encoder).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = self.optim_func(self.q_network.parameters(), **self.optim_args)
    
    def train(self, env, device, training_args, writer):
        assert isinstance(env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        self.initialize(env, device)
        start_time = time.time()

        obs = env.reset()
        pbar = tqdm.trange(training_args.total_timesteps, desc="DiscreteDQN training", unit="step")
        for global_step in pbar:
            epsilon = self.scheduler(global_step)
            if random.random() < epsilon:
                actions = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
            else:
                logits = self.q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(logits, dim=1).cpu().numpy()

            next_obs, rewards, dones, infos = env.step(actions)

            for info in infos:
                if "episode" in info.keys():
                    pbar.set_postfix(episodic_return=info['episode']['r'])
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)
                    break

            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d and "terminal_observation" in infos[idx].keys():
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            
            self.buffer.add(obs, real_next_obs, actions, rewards, dones)
            obs = next_obs

            if global_step > training_args.learning_starts and global_step % training_args.train_frequency == 0:
                data = self.buffer.sample(training_args.batch_size)
                with torch.no_grad():
                    target_max, _ = self.target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + training_args.gamma * target_max * (1 - data.dones.flatten())
                old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if global_step % training_args.target_network_frequency == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

        env.close()
        writer.close()
        pbar.close()