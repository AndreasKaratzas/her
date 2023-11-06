
import sys
sys.path.append("../")

import threading
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, goal_dim, act_dim, max_episode_steps, buffer_size, sampler_fn):
        # parameter declaration
        self.sampler_fn = sampler_fn
        self.experience_stored = 0
        self.current_buffer_size = 0
        self.max_episode_steps = max_episode_steps
        self.max_buffer_size = buffer_size
        self.size = buffer_size // max_episode_steps

        # buffer declaration
        self.obs_buf = np.zeros(
            [self.size, max_episode_steps + 1, obs_dim], dtype=np.float32)
        self.achieved_buf = np.zeros(
            [self.size, max_episode_steps + 1, goal_dim], dtype=np.float32)
        self.desired_buf = np.zeros(
            [self.size, max_episode_steps, goal_dim], dtype=np.float32)
        self.action_buf = np.zeros(
            [self.size, max_episode_steps, act_dim], dtype=np.float32)

        # thread lock
        self.lock = threading.Lock()

    def store(self, experience):
        observations, achieved_goal, desired_goal, actions = experience
        batch_size = observations.shape[0]

        assert observations.shape[0] == achieved_goal.shape[0] == desired_goal.shape[0] == actions.shape[0]
        with self.lock:
            idxs = self._allocate_buffer(size=batch_size)
            # store the experience
            self.obs_buf[idxs] = observations
            self.achieved_buf[idxs] = achieved_goal
            self.desired_buf[idxs] = desired_goal
            self.action_buf[idxs] = actions
            self.experience_stored += self.max_episode_steps * batch_size

    def sample(self, batch_size=32):
        buffer_snapshot = {}

        with self.lock:
            buffer_snapshot['observations'] = self.obs_buf[:self.current_buffer_size]
            buffer_snapshot['achieved_goals'] = self.achieved_buf[:self.current_buffer_size]
            buffer_snapshot['desired_goals'] = self.desired_buf[:self.current_buffer_size]
            buffer_snapshot['actions'] = self.action_buf[:self.current_buffer_size]

        buffer_snapshot['next_observations'] = buffer_snapshot['observations'][:, 1:, :]
        buffer_snapshot['next_achieved_goals'] = buffer_snapshot['achieved_goals'][:, 1:, :]

        # sample transitions
        batch = self.sampler_fn(buffer_snapshot, batch_size)
        return batch

    def _allocate_buffer(self, size=None):
        size = size or 1

        if self.current_buffer_size + size <= self.size:
            idx = np.arange(self.current_buffer_size,
                            self.current_buffer_size + size)
        elif self.current_buffer_size < self.size:
            overflow = size - (self.size - self.current_buffer_size)
            idx_a = np.arange(self.current_buffer_size, self.size)
            idx_b = np.random.randint(0, self.current_buffer_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, size)

        self.current_buffer_size = min(
            self.size, self.current_buffer_size + size)

        if size == 1:
            idx = idx[0]

        return idx
