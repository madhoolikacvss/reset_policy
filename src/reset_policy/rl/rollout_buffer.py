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
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.bootstrap_values = []  # NEW
    
    def add(self, state, action, reward, done, value, log_prob, bootstrap_value=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.bootstrap_values.append(bootstrap_value)  # NEW
    
    def update_last_bootstrap_value(self, bootstrap_value):
        """Update bootstrap value for the last transition."""
        if len(self.bootstrap_values) > 0:
            self.bootstrap_values[-1] = bootstrap_value


    def __len__(self):
        """Return the number of stored transitions."""
        return len(self.states)
    
    def get(self):
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        values = torch.stack(self.values).squeeze(-1)
        log_probs = torch.stack(self.log_probs)
        
        # Handle bootstrap values
        if any(bv is not None for bv in self.bootstrap_values):
            # Some bootstrap values exist - convert None to 0
            bootstrap_tensor = torch.tensor([
                0.0 if bv is None else bv.item() 
                for bv in self.bootstrap_values
            ], dtype=torch.float32)
            return (states, actions, rewards, dones, values, log_probs, bootstrap_tensor)
        else:
            # No bootstrap values - return 6 elements
            return (states, actions, rewards, dones, values, log_probs)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.bootstrap_values.clear() 