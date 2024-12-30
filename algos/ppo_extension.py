from .agent_base import BaseAgent
from .ppo_utils import Policy
from .ppo_agent import PPOAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time

class PPOExtension(PPOAgent):
    def __init__(self, config=None):
        super().__init__(config)
        self.c = 1.3

    def ppo_update(self, states, actions, rewards, next_states, dones, old_log_probs, targets):
        action_dists, values = self.policy(states)
        values = values.squeeze()
        new_action_probs = action_dists.log_prob(actions).sum(dim=-1)
        old_log_probs = old_log_probs.sum(dim=-1)
        ratio = torch.exp(new_action_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)

        advantages = targets - values
        advantages -= advantages.mean()
        advantages /= advantages.std()+1e-8
        advantages = advantages.detach()
        
        positive_advantages = lambda adv : -torch.min(ratio*adv, clipped_ratio*adv)
        negative_advantages = lambda adv : torch.max(positive_advantages(adv), self.c * adv)
        
        #advantages_copy = advantages.detach().numpy()
        policy_objective = torch.where(advantages >= 0, positive_advantages(advantages), negative_advantages(advantages))

        value_loss = F.smooth_l1_loss(values, targets, reduction="mean")
         

        policy_objective = policy_objective.mean()
        entropy = action_dists.entropy().mean()
        loss = policy_objective + 0.5*value_loss - 0.01*entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()