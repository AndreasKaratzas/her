
import sys
sys.path.append('../')

import os
import torch
import numpy as np
import torch.nn as nn
import torcheval.metrics as tnt

from torch.optim import Adam
from copy import deepcopy

import src.model as model

from src.mpi import sync_networks, sync_grads
from src.her import HindsightExperienceReplay
from src.utils import _seed
from src.replay import ReplayBuffer
from src.normalizer import Normalizer

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Agent:
    def __init__(self, env, env_id, logger, actor_critic=model.MLPActorCritic, ac_kwargs=dict(), seed=0, 
                 cycles=50, epochs=100, replay_size=int(1e6), gamma=0.98, updates=40, action_l2=1.,
                 polyak=0.95, lr_actor=1e-3, lr_critic=1e-3, batch_size=128, auto_save=True,
                 noise_eps=0.2, random_eps=0.3, demo_episodes=10, load_checkpoint='model.pth',
                 max_ep_len=1000, test_rollouts=10, checkpoint_freq=1, debug_mode=False, device=None,
                 strategy='future', replay_k=4, clip_return=False, clip_obs=200., norm_clip=5, name='exp', 
                 checkpoint_dir='.', export_configs=False, elite_criterion='success', num_rollouts_per_mpi=2):

        self.env = deepcopy(env)
        self.test_env = deepcopy(env)
        self.env_id = env_id

        # initialize logger
        self.logger = logger

        # hyper-parameters
        self.epochs = epochs
        self.cycles = cycles
        self.updates = updates
        self.gamma = gamma
        self.polyak = polyak
        self.replay_k = replay_k
        self.demo_episodes = demo_episodes
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.norm_clip = norm_clip
        self.clip_obs = clip_obs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.action_l2 = action_l2
        self.test_rollouts = test_rollouts
        self.batch_size = batch_size
        self.num_rollouts_per_mpi = num_rollouts_per_mpi
        self.checkpoint_freq = checkpoint_freq
        self.clip_return = (1. / (1. - gamma)) if clip_return else np.inf
        self.clip_pos_returns = True
        self.checkpoint_dir = checkpoint_dir
        self.elite_criterion = elite_criterion
        self.name = name
        self.epoch = 0
        self.seed = seed + (MPI.COMM_WORLD.Get_rank() if MPI is not None else 0)
        self.auto_save = auto_save
        self.load_checkpoint = load_checkpoint

        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'cpu') if device is None else torch.device(device=device)

        # seed random number generators for debugging purposes
        if debug_mode:
            _seed(env=self.env, device=self.device, seed=seed)

        dummy_step, _ = self.env.reset(seed=seed)
        # self.env.observation_space.shape
        self.obs_dim = dummy_step['observation'].shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.goal_dim = dummy_step['desired_goal'].shape[0]
        self.max_ep_len = max_ep_len if max_ep_len is not None else self.env._max_episode_steps

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.online = actor_critic(
            observation_space=self.obs_dim, action_space=self.act_dim, goal_space=self.goal_dim, action_limit=self.act_limit, **ac_kwargs)
        self.target = deepcopy(self.online)

        # sync the networks across the device
        sync_networks(self.online.actor)
        sync_networks(self.online.critic)

        # sync target with online
        self.target.actor.load_state_dict(
            self.online.actor.state_dict())
        self.target.critic.load_state_dict(
            self.online.critic.state_dict())

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.target.parameters():
            p.requires_grad = False
        
        # upload models to device
        self.online.actor.to(self.device)
        self.online.critic.to(self.device)
        self.target.actor.to(self.device)
        self.target.critic.to(self.device)

        # Count variables (Pro Tip: try to get a feel for how different size networks behave!)
        var_counts = tuple(model.count_vars(module) for module in [self.online.actor, self.online.critic])
        self.logger.log_message('\nNumber of parameters: \t actor: %d, \t critic: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.actor_optimizer = Adam(self.online.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.online.critic.parameters(), lr=lr_critic)

        # her sampler
        self.her = HindsightExperienceReplay(
            strategy=strategy, replay=replay_k, reward_fn=self.env.compute_reward)

        # Experience buffer
        self.replay = ReplayBuffer(
            obs_dim=self.obs_dim, goal_dim=self.goal_dim, act_dim=self.act_dim, 
            max_episode_steps=self.max_ep_len, buffer_size=replay_size, sampler_fn=self.her.sample)

        # create the normalizer
        self.obs_normalizer = Normalizer(
            size=self.obs_dim, default_clip_range=norm_clip)
        self.goal_normalizer = Normalizer(
            size=self.goal_dim, default_clip_range=norm_clip)

        self.setup()
        
        # save the experiment configuration
        if export_configs:
            self.compile_experiment_configs(replay_size, strategy, seed, **ac_kwargs)
            logger.export_yaml(d=self.config, filename=name)

    # initialize progress metrics
    def setup(self):
        self.chkpt_cntr = 0
        self.success = -1
        self.curr_q_val = -np.inf
        self.avg_q_val = tnt.Mean()
        self.max_q_val = -np.inf
        self.min_q_val = np.inf
        self.loss_actor = tnt.Mean()
        self.loss_critic = tnt.Mean()

        if self.epoch == 0:
            if self.elite_criterion == 'success':
                self.elite_register = -1
            if self.elite_criterion == 'avg_q_val':
                self.elite_register = -np.inf
            if self.elite_criterion == 'max_q_val':
                self.elite_register = -np.inf
            if self.elite_criterion == 'min_q_val':
                self.elite_register = np.inf
            if self.elite_criterion == 'loss_actor':
                self.elite_register = np.inf
            if self.elite_criterion == 'loss_critic':
                self.elite_register = np.inf

    def compile_experiment_configs(self, replay_size, strategy, seed, extractor, activation, hidden_sizes):
        if activation == nn.ReLU:
            activation = 'relu'
        elif activation == nn.Sigmoid:
            activation = 'sigmoid'
        elif activation == nn.Tanh:
            activation = 'tanh'
        else:
            raise NotImplementedError(f"Activation function {activation} is currently not supported. "
                                    f"Try one of the following:\n\t1. Sigmoid\n\t2. ReLU\n\t3. Tanh")
        
        self.config = {
            'experiment':
            {
                'alias': self.name, 
                'logger': str(self.logger.log_f_name)
            },
            'env':
            {
                'name': self.env_id
            },
            'ddpg':
            {
                'extractor': str(hidden_sizes),
                'arch': extractor,
                'activation': activation,
                'pi_lr': self.lr_actor,
                'q_lr': self.lr_critic,
                'replay_size': replay_size,
                'polyak': self.polyak,
                'action_l2': self.action_l2,
                'clip_obs': self.clip_obs,
                'gamma': self.gamma,
                'clip_return': self.clip_return
            },
            'training':
            {
                'epochs': self.epochs,
                'cycles': self.cycles,
                'num_rollouts_per_mpi': self.num_rollouts_per_mpi,
                'updates': self.updates,
                'batch_size': self.batch_size,
                'test_rollouts': self.test_rollouts,
                'demo_episodes': self.demo_episodes
            },
            'exploration':
            {
                'random_eps': self.random_eps,
                'noise_eps': self.noise_eps
            },
            'her':
            {
                'sampling_strategy': strategy,
                'replay_k': self.replay_k
            },
            'normalization':
            {
                'norm_clip': self.norm_clip
            },
            'auxiliary':
            {
                'checkpoint_freq': self.checkpoint_freq,
                'seed': seed,
                'checkpoint_dir': str(self.checkpoint_dir),
                'device': self.device.type,
                'elite_metric': self.elite_criterion,
                'auto_save': self.auto_save
            }
        }

    # pre process the inputs
    def preprocess(self, observation, desired_goal):
        observation_normalized = self.obs_normalizer.normalize(observation)
        desired_goal_normalized = self.goal_normalizer.normalize(desired_goal)
        inputs = np.concatenate([observation_normalized, desired_goal_normalized])
        return torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)

    def preprocess_obs_and_goal(self, observations, desired_goals):
        observations = np.clip(observations, -self.clip_obs, self.clip_obs)
        desired_goals = np.clip(desired_goals, -self.clip_obs, self.clip_obs)
        return observations, desired_goals
    
    def update_q_val_register(self, real_q_value):
        self.curr_q_val = np.mean(real_q_value.cpu().detach().numpy())

        if self.max_q_val < self.curr_q_val:
            self.max_q_val = self.curr_q_val
        if self.min_q_val > self.curr_q_val:
            self.min_q_val = self.curr_q_val
        
        self.avg_q_val.update(torch.tensor(self.curr_q_val))
    
    # this function will choose action for the agent and do the exploration
    def act(self, action):
        action = action.cpu().numpy().squeeze()
        # add noise
        action += self.noise_eps * self.act_limit * np.random.randn(*action.shape)
        # clip actions with respect to action range as defined in given environment
        action = np.clip(action, -self.act_limit, self.act_limit)
        # generate random actions
        random_actions = np.random.uniform(low=-self.act_limit, high=self.act_limit, size=self.act_dim)
        # utilize random action vector
        action += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - action)
        return action
    
    def recall(self):
        transitions = self.replay.sample(self.batch_size)

        observations, next_observations, desired_goals = transitions[
            'observations'], transitions['next_observations'], transitions['desired_goals']
        transitions['observations'], transitions['desired_goals'] = self.preprocess_obs_and_goal(
            observations, desired_goals)
        transitions['next_observations'], transitions['next_desired_goals'] = self.preprocess_obs_and_goal(
            next_observations, desired_goals)

        observation_normalized = self.obs_normalizer.normalize(
            transitions['observations'])
        desired_goal_normalized = self.goal_normalizer.normalize(
            transitions['desired_goals'])
        inputs = np.concatenate(
            [observation_normalized, desired_goal_normalized], axis=1)

        next_observation_normalized = self.obs_normalizer.normalize(
            transitions['next_observations'])
        next_desired_goal_normalized = self.goal_normalizer.normalize(
            transitions['next_desired_goals'])
        next_inputs = np.concatenate(
            [next_observation_normalized, next_desired_goal_normalized], axis=1)
        
        inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        next_inputs = torch.tensor(
            next_inputs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(
            transitions['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(
            transitions['rewards'], dtype=torch.float32).to(self.device)
        
        return inputs, next_inputs, actions, rewards
    
    @torch.no_grad()
    def td_target(self, next_inputs, rewards):
        # calculate the target Q value function
        actions_next = self.target.actor(next_inputs)
        q_next_value = self.target.critic(
            next_inputs, actions_next).detach()
        target_q_value = rewards + self.gamma * q_next_value
        target_q_value = target_q_value.detach()
        target_q_value = torch.clamp(target_q_value, -self.clip_return, 0)
        return target_q_value
    
    def compute_loss(self, inputs, actions, target_q_value):
        # the critic loss
        real_q_value = self.online.critic(inputs, actions)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        self.update_q_val_register(real_q_value=real_q_value)

        # the actor loss
        actions_real = self.online.actor(inputs)
        actor_loss = -self.online.critic(inputs, actions_real).mean()
        actor_loss += self.action_l2 * \
            (actions_real / self.act_limit).pow(2).mean()
        
        self.loss_actor.update(torch.from_numpy(actor_loss.cpu().detach().numpy()))
        self.loss_critic.update(torch.from_numpy(critic_loss.cpu().detach().numpy()))
        
        return actor_loss, critic_loss
    
    def update_agent(self, actor_loss, critic_loss):
        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        sync_grads(self.online.actor)
        self.actor_optimizer.step()
        
        # update the critic_network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        sync_grads(self.online.critic)
        self.critic_optimizer.step()


    def learn(self):
        # Sample from memory
        inputs, next_inputs, actions, rewards = self.recall()

        # project predicted reward to true reward values
        target_q_value = self.td_target(next_inputs=next_inputs, rewards=rewards)
        
        # compute agent loss
        actor_loss, critic_loss = self.compute_loss(
            inputs=inputs, actions=actions, target_q_value=target_q_value)
        
        # update agent
        self.update_agent(actor_loss=actor_loss, critic_loss=critic_loss)
        
    # update the normalizer
    def update_normalizer(self, experience):
        observations_batch, achieved_goals_batch, desired_goals_batch, actions_batch = experience
        next_observations_batch = observations_batch[:, 1:, :]
        next_achieved_goals_batch = achieved_goals_batch[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = actions_batch.shape[1]
        # create the new buffer to store them
        buffer_snapshot = {
            'observations': observations_batch,
            'achieved_goals': achieved_goals_batch,
            'desired_goals': desired_goals_batch,
            'actions': actions_batch,
            'next_observations': next_observations_batch,
            'next_achieved_goals': next_achieved_goals_batch,
        }
        transitions = self.her.sample(buffer_snapshot, num_transitions)
        observations, desired_goals = transitions['observations'], transitions['desired_goals']
        # preprocess observations and desired goals
        transitions['observations'], transitions['desired_goals'] = self.preprocess_obs_and_goal(observations, desired_goals)
        # update normalizers
        self.obs_normalizer.update(torch.from_numpy(transitions['observations']))
        self.goal_normalizer.update(torch.from_numpy(transitions['desired_goals']))
        # recompute the stats
        self.obs_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

    # sync target net
    def sync_target(self):
        # actor params
        for target_param, online_param in zip(self.target.actor.parameters(), self.online.actor.parameters()):
            target_param.data.copy_((1 - self.polyak) * online_param.data + self.polyak * target_param.data)
        # critic params
        for target_param, online_param in zip(self.target.critic.parameters(), self.online.critic.parameters()):
            target_param.data.copy_((1 - self.polyak) * online_param.data + self.polyak * target_param.data)
    
    def elite_criterion_factory(self) -> bool:
        if not self.auto_save:
            if self.chkpt_cntr % self.checkpoint_freq:
                return True
            self.chkpt_cntr += 1

        if self.elite_criterion == 'success':
            if self.success > self.elite_register:
                self.elite_register = self.success
                return True
        
        if self.elite_criterion == 'avg_q_val':
            if self.avg_q_val.compute() > self.elite_register:
                self.elite_register = self.avg_q_val.compute()
                return True
        
        if self.elite_criterion == 'max_q_val':
            if self.max_q_val > self.elite_register:
                self.elite_register = self.max_q_val
                return True

        if self.elite_criterion == 'min_q_val':
            if self.min_q_val < self.elite_register:
                self.elite_register = self.min_q_val
                return True

        if self.elite_criterion == 'loss_actor':
            if self.loss_actor.compute() < self.elite_register:
                self.elite_register = self.loss_actor.compute()
                return True

        if self.elite_criterion == 'loss_critic':
            if self.loss_critic.compute() < self.elite_register:
                self.elite_register = self.loss_critic.compute()
                return True
        
        return False
        
    def load(self, agent_checkpoint_path):
        if not agent_checkpoint_path.exists():
            raise ValueError(f"{agent_checkpoint_path} does not exist")
        
        ckp = torch.load(agent_checkpoint_path, map_location=self.device)

        self.obs_normalizer.mean = ckp.get('obs_normalizer_mean')
        self.obs_normalizer.std = ckp.get('obs_normalizer_std')
        self.goal_normalizer.mean = ckp.get('goal_normalizer_mean')
        self.goal_normalizer.std = ckp.get('goal_normalizer_std')

        print(f"Loading model at {agent_checkpoint_path}")

        self.online.actor.load_state_dict(ckp.get('online_actor'))
        self.online.critic.load_state_dict(ckp.get('online_critic'))
        self.target.actor.load_state_dict(ckp.get('target_actor'))
        self.target.critic.load_state_dict(ckp.get('target_critic'))
        self.actor_optimizer.load_state_dict(ckp.get('actor_optimizer'))
        self.critic_optimizer.load_state_dict(ckp.get('critic_optimizer'))

        self.success = ckp.get('success')
        self.avg_q_val = ckp.get('avg_q_val')
        self.max_q_val = ckp.get('max_q_val')
        self.min_q_val = ckp.get('min_q_val')
        self.loss_actor = ckp.get('loss_actor')
        self.loss_critic = ckp.get('loss_critic')

        print(
            f"Loaded checkpoint with:"
            f"\n\t * {self.success:7.3f} success rate"
            f"\n\t * {self.avg_q_val.compute().numpy().round(3).item():7.3f} mean Q value"
            f"\n\t * {self.max_q_val:7.3f} maximum Q value achieved"
            f"\n\t * {self.loss_actor.compute().numpy().round(3).item():7.3f} actor model loss"
            f"\n\t * {self.loss_critic.compute().numpy().round(3).item():7.3f} critic model loss")
    
    def store(self):

        self.epoch += 1

        if self.elite_criterion_factory():
            torch.save({
                'obs_normalizer_mean': self.obs_normalizer.mean,
                'obs_normalizer_std': self.obs_normalizer.std,
                'goal_normalizer_mean': self.goal_normalizer.mean,
                'goal_normalizer_std': self.goal_normalizer.std,
                'online_actor': self.online.actor.state_dict(),
                'online_critic': self.online.critic.state_dict(),
                'target_actor': self.target.actor.state_dict(),
                'target_critic': self.target.critic.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'success': self.success,
                'avg_q_val': self.avg_q_val,
                'max_q_val': self.max_q_val,
                'min_q_val': self.min_q_val,
                'loss_actor': self.loss_actor,
                'loss_critic': self.loss_critic
            }, os.path.join(
                self.checkpoint_dir, 
                f"epoch_{self.epoch:05d}-success_{self.success:07.3f}-" + f"avg_q_val_{self.avg_q_val.compute().numpy().round(3).item():07.3f}-" +
                f"max_q_val_{self.max_q_val:07.3f}-" + f"min_q_val{self.min_q_val:07.3f}-" +
                f"loss_actor_{self.loss_actor.compute().numpy().round(3).item():07.3f}-" +
                f"loss_critic_{self.loss_critic.compute().numpy().round(3).item():07.3f}.pth"
            ))
