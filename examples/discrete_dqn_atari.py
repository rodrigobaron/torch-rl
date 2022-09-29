import argparse
from distutils.util import strtobool

import random
import time

import gym

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torch_rl.utils import set_seed, LinearSchedule
from torch_rl.envs import make_gym_env
from torch_rl.buffers import ReplayBuffer

from torch_rl.core.encoder import mlp_encoder
from torch_rl.algos.dqn import DiscreteDQN, DQNTrainingArgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--log-path", type=str, default=".logs/",
        help="the base log path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__DiscreteDQN__{args.seed}__{int(time.time())}"
    summary_path = f"{args.log_path}/runs/{run_name}"
    record_path = f"{args.log_path}/videos/{run_name}"

    writer = SummaryWriter(summary_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    set_seed(seed=args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_gym_env(args.env_id, args.seed, 0, record_path)]
    )

    scheduler = LinearSchedule(
        args.start_e, 
        args.end_e, 
        args.exploration_fraction * args.total_timesteps
    )

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device
    )

    dqn = DiscreteDQN(
        encoder=mlp_encoder(envs.single_observation_space.shape),
        optim= optim.Adam,
        optim_args = dict(
            lr = args.learning_rate
        ),
        scheduler = scheduler,
        buffer = rb
    )

    training_args = DQNTrainingArgs(
        total_timesteps = args.total_timesteps,
        learning_starts = args.learning_starts,
        train_frequency = args.train_frequency,
        target_network_frequency = args.target_network_frequency,
        batch_size = args.batch_size,
        gamma = args.gamma
    )

    dqn.train(
        env=envs, 
        device=device,
        training_args = training_args,
        writer = writer
    )

    # eval_env = gym.make(args.env_id)
    # env_evaluator = EnvEvaluator(env=eval_env)

    # env_evaluator.play(
    #     eval_agent,
    #     trials=10,
    #     log_path='.logs/', 
    #     record=True, 
    #     use_gpu=True
    # )