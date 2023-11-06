
import sys
sys.path.append('./')

import gymnasium as gym

import src.model as model

from src.args import arguments
from src.agent import Agent
from src.train import train
from src.logger import HardLogger
from src.utils import configure_os_vars, parse_configs, update_args, _mlp_configurations, _logger_configuration, info

try:
    from mpi4py import MPI
except ImportError:
    raise ImportError(f"`MPI` could not be imported from `mpi4py`.")


if __name__ == '__main__':
    # initialize MPI vars
    if MPI is not None:
        configure_os_vars()

    # parse arguments
    args = arguments()

    if MPI.COMM_WORLD.Get_rank() == 0:
        if args.info:
            info()
    
    if args.config:
        settings = parse_configs(filepath=args.config)
        args = update_args(args, settings)
        args.export_configs = False
    else:
        args.export_configs = True

    # create RL environment
    env = gym.make(args.env)

    ac_kwargs = _mlp_configurations(
        hidden_sizes=args.hidden_sizes, activation=args.activation, extractor=args.arch)
    logger_kwargs = _logger_configuration(
        output_dir=args.checkpoint_dir, output_fname=args.logger_name, exp_name=args.name)
    
    logger = HardLogger(**logger_kwargs)
    logger.print_training_message(
        agent="DDPG with Hindsight Experience Replay and " + args.arch.upper() + " core", 
        env_id=args.env, epochs=args.epochs, device=args.device, elite_metric=args.elite_criterion, 
        auto_save=(args.elite_criterion.lower() != 'none'))

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

    # train agent
    train(agent=agent)
