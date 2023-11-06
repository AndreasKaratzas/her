
import sys
sys.path.append("../")

import torch
import numpy as np
import torch.nn as nn

from typing import List
from copy import deepcopy


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# TODO: embed cnn in code
# issue: https://github.com/openai/spinningup/issues/367 
# template: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L15

def cnn(
    # kernel details
    channels: List[int] = ['auto', 32, 64, 64], 
    kernel_size: List[int] = [8, 4, 3], 
    stride: List[int] = [4, 2, 1], 
    # linear layer details
    sizes: List[int] = ['auto', 512, 128, 1], 
    # activation details
    conv_activation: nn.Module = nn.ReLU,
    linear_activation: nn.Module = nn.ReLU, 
    output_activation: nn.Module = nn.Tanh,
    # expected input dimensions
    dims: List[int] = [4, 84, 84]
):
    layers = []

    assert len(channels) - 1 == len(kernel_size) == len(stride)

    for j in range(len(channels)-1):
        if channels[j] == 'auto' and j == 0:
            layers += [
                nn.Conv2d(
                    in_channels=dims[j],
                    out_channels=channels[j+1],
                    kernel_size=kernel_size[j],
                    stride=stride[j]
                ),
                conv_activation()
            ]
        else:
            layers += [
                nn.Conv2d(
                    in_channels=channels[j], 
                    out_channels=channels[j+1], 
                    kernel_size=kernel_size[j], 
                    stride=stride[j]
                ), 
                conv_activation()
            ]
    
    features = deepcopy(layers)
    features += [nn.Flatten()]
    feature_extractor = nn.Sequential(*features)
    dummy_input = torch.zeros(size=[1, *dims])
    dummy_output = feature_extractor(dummy_input)
    flattened = dummy_output.shape[1]
    sizes.insert(0, flattened)

    for j in range(len(sizes) - 1):
        if j == 0:
            act = linear_activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(flattened, sizes[j+1]), act()]
        else:
            act = linear_activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):

    def __init__(self, obs_dim, goal_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim + goal_dim] + list(hidden_sizes) + [act_dim]
        self.actor = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.actor(obs)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, goal_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.act_limit = act_limit

        self.critic = mlp([obs_dim + act_dim + goal_dim] +
                        list(hidden_sizes) + [1], activation)
        

    def forward(self, obs, act):
        q = self.critic(torch.cat([obs, act / self.act_limit], dim=1))
        return q


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, goal_space, action_limit, hidden_sizes=(256, 256),
                 activation=nn.ReLU, extractor='mlp'):
        super().__init__()
        
        # build policy and value functions
        self.actor = MLPActor(
            obs_dim=observation_space, 
            act_dim=action_space, 
            goal_dim=goal_space, 
            hidden_sizes=hidden_sizes, 
            activation=activation, 
            act_limit=action_limit
        )

        self.critic = MLPQFunction(
            obs_dim=observation_space, 
            act_dim=action_space, 
            goal_dim=goal_space, 
            hidden_sizes=hidden_sizes, 
            activation=activation,
            act_limit=action_limit
        )

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs).numpy()