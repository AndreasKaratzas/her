
from mujoco_py.generated import const
from mujoco_py import GlfwContext
from gymnasium import wrappers
import sys
sys.path.append('../')

import gymnasium as gym
import time
import torch
import psutil
import numpy as np
import torcheval.metrics as tnt

from tqdm import tqdm
import src.model as model

from src.agent import Agent
from src.args import arguments
from src.logger import HardLogger
from src.utils import configure_os_vars, parse_configs, update_args, _mlp_configurations, _logger_configuration, info, _cycle_vars


try:
    from mpi4py import MPI
except ImportError:
    raise ImportError(f"`MPI` could not be imported from `mpi4py`.")


def process_inputs(observation, goal, observation_mean, observation_std, goal_mean, goal_std, clip_obs, clip_range):
    observation_clip = np.clip(observation, -clip_obs, clip_obs)
    goal_clip = np.clip(goal, -clip_obs, clip_obs)
    observation_norm = np.clip((observation_clip - observation_mean) / (observation_std), -clip_range, clip_range)
    goal_norm = np.clip((goal_clip - goal_mean) / (goal_std), -clip_range, clip_range)
    input = np.concatenate([observation_norm, goal_norm])
    input = torch.tensor(input, dtype=torch.float32)
    return input


if __name__ == '__main__':
    # initialize MPI vars
    if MPI is not None:
        configure_os_vars()
    
    GlfwContext(offscreen=True)

    # parse arguments
    args = arguments()

    if args.info:
        info()

    if args.config:
        settings = parse_configs(filepath=args.config)
        args = update_args(args, settings)
        args.export_configs = False
    else:
        args.export_configs = True

    ac_kwargs = _mlp_configurations(
        hidden_sizes=args.hidden_sizes, activation=args.activation, extractor=args.extractor)
    logger_kwargs = _logger_configuration(
        output_dir=args.checkpoint_dir, output_fname=args.logger_name, exp_name=args.name)

    logger = HardLogger(**logger_kwargs)
    logger.print_test_message(
        agent="DDPG with Hindsight Experience Replay and " + args.arch.upper() + "core",
        env_id=args.env, epochs=args.epochs, device=args.device)
    
    # create RL environment
    env = gym.make(args.env)
    env = wrappers.Monitor(env, logger.demo_dir, video_callable=lambda episode_id: True, force=True)

    # create the DDPG agent with Hindsight Experience Replay
    agent = Agent(env=env, env_id=args.env, actor_critic=model.MLPActorCritic, ac_kwargs=ac_kwargs,
                  seed=args.seed, test_rollouts=args.test_rollouts, cycles=args.cycles,
                  epochs=args.epochs, replay_size=args.replay_size, gamma=args.gamma,
                  updates=args.updates, action_l2=args.action_l2, polyak=args.polyak, auto_save=args.auto_save,
                  lr_actor=args.lr_actor, lr_critic=args.lr_critic, batch_size=args.batch_size,
                  elite_criterion=args.elite_criterion, noise_eps=args.noise_eps, random_eps=args.random_eps,
                  demo_episodes=args.demo_episodes, max_ep_len=args.max_ep_len, logger=logger,
                  checkpoint_freq=args.checkpoint_freq, debug_mode=args.debug_mode, device=args.device,
                  strategy=args.sampling_strategy, replay_k=args.replay_k, clip_return=args.clip_return,
                  clip_obs=args.clip_obs, norm_clip=args.norm_clip, name=args.name, num_rollouts_per_mpi=args.num_rollouts_per_mpi,
                  checkpoint_dir=logger.model_dir, export_configs=args.export_configs, load_checkpoint=args.load_checkpoint)

    agent.load(agent_checkpoint_path=agent.load_checkpoint)

    if MPI.COMM_WORLD.Get_rank() == 0:
        epoch_time = tnt.Mean()
        print(
            f"\n\n{'Episode':>10}{'bench':>11}{'gpu_mem':>9}{'ram_util':>13}{'success':>14}")
    
    for episode in range(agent.demo_episodes):

        if MPI.COMM_WORLD.Get_rank() == 0:
            ep_start = time.time()

        observation, e_info = agent.env.reset(seed=agent.seed)
        
        observation, goal = _cycle_vars(observation=observation, mode='test')

        for step in range(env._max_episode_steps):
            env.render()
            
            inputs = process_inputs(
                observation=observation, 
                goal=goal,
                o_mean=agent.obs_normalizer.mean,
                o_std=agent.obs_normalizer.std, 
                g_mean=agent.goal_normalizer.mean,
                g_std=agent.goal_normalizer.std, 
                clip_obs=agent.clip_obs,
                clip_range=agent.norm_clip
            )

            with torch.no_grad():
                action = agent.online.actor(inputs)
            action = action.cpu().detach().numpy().squeeze()
            
            next_observation, reward, done, truncated, e_info = agent.env.step(action)
            observation = next_observation['observation']
            frame = agent.env.render(mode="human")
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            ep_end = time.time()
            epoch_time.update(torch.tensor(ep_end - ep_start))
        
            # log agent progress
            print(('%10s' + '%11s' + '%9s' + '%13s' + '%14.3g') % (
                f'{episode + 1}/{agent.epochs}',
                f'{epoch_time.compute().numpy().round(3).item()}',
                f'{round(torch.cuda.memory_reserved() / 1E9, 3) if torch.cuda.is_available() else 0:.3g} G',
                f'{psutil.virtual_memory().percent} %',
                round(agent.success, 3)))

    agent.env.close()
