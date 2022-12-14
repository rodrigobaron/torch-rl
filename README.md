# torch-rl

TorchRL is a Deep Reinforcement Learning (RL), Offline RL, Imitation Learning and Multi Agent RL algorithms re-implementation using pytorch as simple and standalone python notebooks.

## Algorithms

List of implemented algorithms

| Algorithm      | Notebook |
| ----------- | ----------- |
| [Proximal Policy Gradient (PPO)](https://arxiv.org/pdf/1707.06347.pdf)  |  [`ppo_atari.ipynb`](notebooks/ppo_atari.ipynb) |
| [Deep Q-Learning (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) |  [`dqn.ipynb`](notebooks/dqn.ipynb) |

:TODO:

## Environments

List of environments used.

### Atari

<div align='center'>
  <img alt="Playing on Asterix, Boxing, Breakout, Demon Attack, Freeway, Gopher, Kung Fu Master, Pong" src="assets/atari.gif">
</div>

:TODO:

## Dev Setup

The development docker uses [torch-jupyter](https://github.com/rodrigobaron/torch-jupyter) as base image so running [docker-buid](docker-build.sh) and [docker-run](docker-run.sh) will setup everything.

## License

[MIT License](LICENSE)