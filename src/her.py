
import sys
sys.path.append("../")

import numpy as np


class HindsightExperienceReplay:
    def __init__(self, strategy, replay, reward_fn=None):
        # parameter declaration
        self.strategy = strategy
        self.replay = replay
        self.reward_fn = reward_fn

        # initialize sampler probability
        if self.strategy == "future":
            self.probability = 1 - (1. / (1 + replay))
        else:
            self.probability = 0

    def sample(self, batch, exp_size):
        # get sizes
        experiences_shape = batch['actions'].shape[1]
        rollout_shape = batch['actions'].shape[0]

        # select experiences
        episodes = np.random.randint(0, rollout_shape, exp_size)
        samples = np.random.randint(experiences_shape, size=exp_size)
        experiences = {key: batch[key][episodes, samples].copy(
        ) for key in batch.keys()}

        # her experiences
        idxs = np.where(np.random.uniform(
            size=exp_size) < self.probability)
        offset = np.random.uniform(
            size=exp_size) * (experiences_shape - samples)
        offset = offset.astype(int)
        future_experiences = (samples + 1 + offset)[idxs]

        # replace desired goals with achieved goals
        future_achieved_goals = batch['achieved_goals'][episodes[idxs],
                                                        future_experiences]
        experiences['desired_goals'][idxs] = future_achieved_goals

        # compute reward
        experiences['rewards'] = np.expand_dims(self.reward_fn(
            experiences['next_achieved_goals'], experiences['desired_goals'], None), 1)
        experiences = {k: experiences[k].reshape(
            exp_size, *experiences[k].shape[1:]) for k in experiences.keys()}

        return experiences
