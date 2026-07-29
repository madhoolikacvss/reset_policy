"""
collects:
    state
    action
    reward
    done
    value
    log probability
"""

import torch

class RolloutBuffer:

    def __init__(self):

        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def add(self,state,action,reward,done,value,log_prob,): 
        self.states.append(state) 
        self.actions.append(action) 
        self.rewards.append(reward) 
        self.dones.append(done) 
        self.values.append(value) 
        self.log_probs.append(log_prob)

    def get(self):
        """
        Convert everything into tensors.
        """

        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.tensor(
            self.rewards,
            dtype=torch.float32,
        )
        dones = torch.tensor(
            self.dones,
            dtype=torch.float32,
        )
        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)

        return (
            states,
            actions,
            rewards,
            dones,
            values,
            log_probs,
        )