"""Future-goal HER, adapted from the hardware policy to four observations."""
import numpy as np
from environment import normalize_goal, distance_reward

class HERBuffer:
    def __init__(self, base_buffer, cfg):
        self.base = base_buffer
        self.cfg = cfg
        self.episode = []

    def __len__(self):
        return len(self.base)

    def add(self, **transition):
        transition = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in transition.items()}
        self.base.add(**transition)
        self.episode.append(transition)

    def finalize_episode(self):
        added = 0
        for t, tr in enumerate(self.episode[:-1]):
            for j in np.random.randint(t+1, len(self.episode), size=self.cfg.her_k):
                goal = self.episode[j]['achieved_goal'].copy()
                state, next_state = tr['state'].copy(), tr['next_state'].copy()
                state[2:4] = next_state[2:4] = normalize_goal(goal)
                reward, success = distance_reward(tr['achieved_goal'], goal, self.cfg)
                self.base.add(**{**tr, 'state': state, 'next_state': next_state,
                                 'reward': reward, 'done': float(success), 'desired_goal': goal})
                added += 1
        self.episode.clear()
        return added

    def sample(self, batch_size, device=None):
        return self.base.sample(batch_size, device)
