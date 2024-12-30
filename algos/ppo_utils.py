import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
import  torch

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, env, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        
        self.actor_logstd = torch.eye(self.action_space)
        
        self.fc1_a = torch.nn.Linear(state_space, hidden_size)
        self.fc2_a = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_a = torch.nn.Linear(hidden_size, action_space)

        self.fc1_c = torch.nn.Linear(state_space, hidden_size)
        self.fc2_c = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_c = torch.nn.Linear(hidden_size, 1)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_a = self.fc1_a(x)
        x_a = F.relu(x_a)
        x_a = self.fc2_a(x_a)
        x_a = F.relu(x_a)
        x_a = self.fc3_a(x_a)

        x_c = self.fc1_c(x)
        x_c = F.relu(x_c)
        x_c = self.fc2_c(x_c)
        x_c = F.relu(x_c)
        x_c = self.fc3_c(x_c)
        
        action_mean = x_a 
        critic_values = x_c
        #print(f"mean dimension: {action_mean.shape} mean: {action_mean}")
        
        action_std = torch.exp(torch.diag(self.actor_logstd)).expand_as(action_mean)
        #print(f"std dimension: {action_std.shape} std: {action_std}")
        
        action_probs = torch.distributions.Normal(action_mean, action_std)
        return action_probs, critic_values
    
    def set_logstd_ratio(self, ratio_of_episodes):
        self.actor_logstd = ratio_of_episodes * torch.eye(self.action_space)