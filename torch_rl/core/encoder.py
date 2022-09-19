import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def conv_encoder(obs_shape: int):
    """Build a convolution encoder used to encode 2-dim observations."""
    return nn.Sequential(
        layer_init(nn.Conv2d(4, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU()
    )

def mlp_encoder(obs_shape: int):
    """Build a full connected encoder used to encode raw 1-dim observations."""
    return nn.Sequential(
        nn.Linear(np.array(obs_shape).prod(), 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
    )
