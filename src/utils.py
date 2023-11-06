
import sys
sys.path.append("../")

import os 
import re
import gymnasium as gym
import time
import yaml
import json
import torch
import random
import numpy as np
import torch.nn as nn
import os.path as osp

from typing import List
from collections import defaultdict 

from src.logger import colorstr

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    

def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) 
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) 
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def envs():
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)

    # reading benchmark names directly from retro requires
    # importing retro here, and for some reason that crashes tensorflow
    # in ubuntu
    _game_envs['retro'] = {
        'BubbleBobble-Nes',
        'SuperMarioBros-Nes',
        'TwinBee3PokoPokoDaimaou-Nes',
        'SpaceHarrier-Nes',
        'SonicTheHedgehog-Genesis',
        'Vectorman-Genesis',
        'FinalFight-Snes',
        'SpaceInvaders-Snes',
    }

    return _game_envs


def get_env_type(env: str, game_envs: defaultdict, env_type: str = None):
    env_id = env

    if env_type is not None:
        return env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in game_envs.keys():
        env_type = env_id
        env_id = [g for g in game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, game_envs.keys())

    return env_type, env_id


def get_default_network(env_type: str) -> str:
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_env_details(env: gym.Env):
    dummy_step, _ = env.reset()
    # self.env.observation_space.shape
    obs_dim = dummy_step['observation'].shape[0]
    act_dim = env.action_space.shape[0]
    goal_dim = dummy_step['desired_goal'].shape[0]
    max_ep_len = env._max_episode_steps

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    return obs_dim, act_dim, goal_dim, max_ep_len, act_limit


def _seed(env: gym.Env, device: torch.device, seed: int= 0):
    # set random seeds for reproduce
    random.seed(seed + (MPI.COMM_WORLD.Get_rank() if MPI is not None else 0))
    np.random.seed(seed + (MPI.COMM_WORLD.Get_rank() if MPI is not None else 0))
    torch.manual_seed(seed + (MPI.COMM_WORLD.Get_rank() if MPI is not None else 0))
    if 'cuda' in device.type:
        torch.cuda.manual_seed(seed + (MPI.COMM_WORLD.Get_rank() if MPI is not None else 0))


def configure_os_vars():
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'


def parse_configs(filepath: str):
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def recursive_items(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def update_args(args, settings):
    for key, value in recursive_items(settings):
        if key == 'alias':
            args.name = value
        if key == 'name':
            args.env = value
        if key == 'extractor':
            args.extractor = value
        
        if key == 'arch':
            _game_envs = envs()
            env_type, env_id = get_env_type(env=args.env, game_envs=_game_envs)
            net_arch = get_default_network(env_type=env_type)

            if net_arch != value:
                print(f"{colorstr(['red', 'bold'], list(['Warning']))}: Model "
                      f"configuration was {value} whereas the default value "
                      f"for {args.env} is {net_arch}.")

            args.arch = value
        
        if key == 'epochs':
            args.epochs = value
        if key == 'activation':
            args.activation = value
        if key == 'pi_lr':
            args.lr_actor = value
        if key == 'q_lr':
            args.lr_critic = value
        if key == 'replay_size':
            args.replay_size = value
        if key == 'polyak':
            args.polyak = value
        if key == 'action_l2':
            args.action_l2 = value
        if key == 'clip_obs':
            args.clip_obs = value
        if key == 'gamma':
            args.gamma = value
        if key == 'clip_return':
            args.clip_return = value
        if key == 'cycles':
            args.cycles = value
        if key == 'num_rollouts_per_mpi':
            args.num_rollouts_per_mpi = value
        if key == 'updates':
            args.updates = value
        if key == 'batch_size':
            args.batch_size = value
        if key == 'test_rollouts':
            args.test_rollouts = value
        if key == 'demo_episodes':
            args.demo_episodes = value
        if key == 'random_eps':
            args.random_eps = value
        if key == 'noise_eps':
            args.noise_eps = value
        if key == 'sampling_strategy':
            args.sampling_strategy = value
        if key == 'replay_k':
            args.replay_k = value
        if key == 'norm_eps':
            args.norm_eps = value
        if key == 'norm_clip':
            args.norm_clip = value
        if key == 'checkpoint_freq':
            args.checkpoint_freq = value
        if key == 'seed':
            args.seed = value
        if key == 'checkpoint_dir':
            args.checkpoint_dir = value
        if key == 'device':
            args.device = value
        if key == 'auto_save':
            args.auto_save = value
    return args


def _mlp_configurations(hidden_sizes: List[int], activation: str, extractor: str):
    if activation.lower() == 'relu':
        activation = nn.ReLU
    elif activation.lower() == 'sigmoid':
        activation = nn.Sigmoid
    elif activation.lower() == 'tanh':
        activation = nn.Tanh
    else:
        raise NotImplementedError(f"Activation function {activation} is currently not supported. "
                                  f"Try one of the following:\n\t1. Sigmoid\n\t2. ReLU\n\t3. Tanh")
    return {
        'extractor': 'mlp',
        'activation': activation,
        'hidden_sizes': hidden_sizes
    }


def _logger_configuration(output_dir: str, output_fname: str, exp_name: str):
    return {
        'output_dir': output_dir,
        'output_fname': output_fname,
        'exp_name': exp_name
    }


def _cycle_vars(observation, mode: str):
    assert mode in ['train', 'test']
    
    if mode == 'train':
        return observation['observation'], observation['achieved_goal'], observation['desired_goal']
    if mode == 'test':
        return observation['observation'], observation['desired_goal']
    

def _next_cycle_vars(next_observation, mode: str):
    assert mode in ['train', 'test']

    if mode == 'train':
        return next_observation['observation'], next_observation['achieved_goal']
    if mode == 'test':
        return next_observation['observation'], next_observation['desired_goal']


class EpochVars:
    def __init__(self):
        self.reset()
    
    def update(self, observation, achieved_goal, desired_goal, action):
        self.epoch_observations.append(observation.copy())
        self.epoch_achieved_goals.append(achieved_goal.copy())
        self.epoch_desired_goals.append(desired_goal.copy())
        self.epoch_actions.append(action.copy())
    
    def finalize(self, observation, achieved_goal):
        self.epoch_observations.append(observation.copy())
        self.epoch_achieved_goals.append(achieved_goal.copy())

    def reset(self):
        self.epoch_observations = [] 
        self.epoch_achieved_goals = []
        self.epoch_desired_goals = [] 
        self.epoch_actions = []

class CycleVars:
    def __init__(self):
        self.reset()

    def update(self, epoch_vars):
        self.observations_batch.append(epoch_vars.epoch_observations)
        self.achieved_goals_batch.append(epoch_vars.epoch_achieved_goals)
        self.desired_goals_batch.append(epoch_vars.epoch_desired_goals)
        self.actions_batch.append(epoch_vars.epoch_actions)
    
    def typecast(self):
        self.observations_batch = np.array(self.observations_batch)
        self.achieved_goals_batch = np.array(self.achieved_goals_batch)
        self.desired_goals_batch = np.array(self.desired_goals_batch)
        self.actions_batch = np.array(self.actions_batch)
    
    def reset(self):
        self.observations_batch = [] 
        self.achieved_goals_batch = [] 
        self.desired_goals_batch = [] 
        self.actions_batch = []

    def compile_experience(self):
        return [
            self.observations_batch, 
            self.achieved_goals_batch, 
            self.desired_goals_batch, 
            self.actions_batch
        ]


def info():
    print(f"\n\n"
          f"\t\t        The {colorstr(['red', 'bold'], list(['Odysseus']))} suite serves as a framework for \n"
          f"\t\t    reinforcement learning agents training on OpenAI GYM \n"
          f"\t\t     defined environments. This repository was created \n"
          f"\t\t    created to help in future projects that would require \n"
          f"\t\t     such agents to find solutions for complex problems. \n"
          f"\n")
