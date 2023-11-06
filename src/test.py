
import sys
sys.path.append('../')

import torch
import numpy as np

from mpi4py import MPI

from src.utils import _cycle_vars, _next_cycle_vars


def test(agent):

    # declare agent accuracy register
    n_success = []

    # initialize agent demo loop
    for demo in range(agent.demo_episodes):
        
        # monitor agent decision success in each time step 
        is_success = []

        # reset the environment
        observation = agent.test_env.reset(seed=agent.seed)

        # initialize cycle variables
        observation, desired_goal = _cycle_vars(observation=observation, mode="test")
        
        # start agent inference
        for step in range(agent.max_ep_len):
            
            with torch.no_grad():
                input = agent.preprocess(observation, desired_goal)
                action_raw = agent.online.actor(input)
                action = action_raw.detach().cpu().numpy().squeeze()
            
            # feed the action into the environment
            next_observation, _, _, info = agent.test_env.step(action)
            
            # update cycle variables
            observation, desired_goal = _next_cycle_vars(next_observation, mode="test")
            
            # update decision success register
            is_success.append(info['is_success'])
        
        # update agent accuracy register
        n_success.append(is_success)
    
    # convert agent accuracy register into array
    n_success = np.array(n_success)
    
    # compute agent performance
    local_success_rate = np.mean(n_success[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    
    return global_success_rate / MPI.COMM_WORLD.Get_size()
