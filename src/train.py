
import sys
sys.path.append('../')

import time
import torch
import psutil
import torcheval.metrics as tnt

from mpi4py import MPI

from src.test import test
from src.agent import Agent
from src.logger import Metrics
from src.utils import _cycle_vars, _next_cycle_vars, CycleVars, EpochVars


def train(agent: Agent):

    # initialize placeholders
    cycle_vars = CycleVars()
    epoch_vars = EpochVars()

    epoch_time = tnt.Mean()
    cycle_time = tnt.Mean()

    if MPI.COMM_WORLD.Get_rank() == 0:
        metrics = Metrics()
        init_msg = f"{'Epoch':>9}{'epoch_time':>13}{'cycle_time':>13}{'gpu_mem':>9}{'ram_util':>11}{'success':>10}{'avg_q_val':>12}{'max_q_val':>12}{'min_q_val':>12}{'loss_actor':>12}{'loss_critic':>12}"
        print("\n\n" + init_msg)
        agent.logger.log_message(init_msg)

    # Agent training loop
    for epoch in range(agent.epochs):
        
        agent.setup()

        ep_start = time.time()
        agent.epoch = epoch

        for cycle in range(agent.cycles):

            if MPI.COMM_WORLD.Get_rank() == 0:
                c_start = time.time()
            
            # reset cycle batches
            cycle_vars.reset()
            
            for num_rollouts in range(agent.num_rollouts_per_mpi):
                
                # reset the rollouts
                epoch_vars.reset()
                
                # reset the environment
                observation, info = agent.env.reset(seed=agent.seed)
                
                # initialize cycle variables
                observation, achieved_goal, desired_goal = _cycle_vars(observation=observation, mode="train")
            
                # start to collect samples
                for step in range(agent.max_ep_len):
                    
                    with torch.no_grad():
                        obs_preprocessed = agent.preprocess(observation, desired_goal)
                        action_raw = agent.online.actor(obs_preprocessed)
                        action = agent.act(action_raw)
                    
                    # feed the action into the environment
                    next_observation, reward, done, truncated, info = agent.env.step(action)

                    # append rollouts
                    epoch_vars.update(observation, achieved_goal, desired_goal, action)

                    # update cycle variables
                    observation, achieved_goal = _next_cycle_vars(next_observation, mode="train")
                
                epoch_vars.finalize(observation, achieved_goal)
                cycle_vars.update(epoch_vars)
            
            # convert cycle variables into arrays
            cycle_vars.typecast()
            
            # store the episodes
            agent.replay.store(experience=cycle_vars.compile_experience())
            agent.update_normalizer(experience=cycle_vars.compile_experience())
            
            for update in range(agent.updates):
                # train the network
                agent.learn()
            
            # soft update
            agent.sync_target()

            if MPI.COMM_WORLD.Get_rank() == 0:
                c_end = time.time()
                cycle_time.update(torch.tensor(c_end - c_start))
        
        # start the evaluation
        agent.success = test(agent=agent)
        
        if MPI.COMM_WORLD.Get_rank() == 0:
            # checkpoint the agent
            agent.store()
            
            ep_end = time.time()
            epoch_time.update(torch.tensor(ep_end - ep_start))
            
            # log agent progress
            msg = metrics.compile(
                epoch=epoch,
                epochs=agent.epochs,
                epoch_time=epoch_time.compute().numpy().round(3).item(),
                cycle_time=cycle_time.compute().numpy().round(3).item(),
                cuda_mem=round(torch.cuda.memory_reserved() / 1E6, 3) if agent.device.type == 'cuda' else 0, 
                show_cuda=agent.device.type == 'cuda',
                ram_util=psutil.virtual_memory().percent,
                success=round(agent.success, 3), 
                avg_q_val=agent.avg_q_val.compute().numpy().round(3).item(), 
                max_q_val=agent.max_q_val.numpy().round(3).item(),
                min_q_val=agent.min_q_val.numpy().round(3).item(), 
                loss_actor=agent.loss_actor.compute().numpy().round(3).item(), 
                loss_critic=agent.loss_critic.compute().numpy().round(3).item()
            )
    
            agent.logger.log_message(msg)

    agent.logger.compile_plots()
