import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResBlock(nn.Module):
    def __init__(self, channels=128):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        return out

class ResNet(nn.module):
    def __init__(self, input_channels=17, n_blocks=10, n_actions=8*8*4):
        super(ResNet, self).__init__()
        self.conv_init = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(128)
        
        self.res_blocks = nn.ModuleList([ResBlock(128) for _ in range(n_blocks-1)])
        
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy = nn.Linear(2*8*8, n_actions)
        
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_linear = nn.Linear(1*8*8, 1, 64)
        self.value = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy(policy)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_linear(value))
        value = self.value(value)
        value = torch.tanh(value)
        return policy, value
    
class ReplayBuffer():
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        
    def add(self, state, action_probs, value):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action_probs, value))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = [data[0] for data in batch]
        action_probs = [data[1] for data in batch]
        values = [data[2] for data in batch]
        return states, action_probs, values
    
class Trainer():
    def __init__(self, model, device, replay_buffer):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.device = device
        self.replay_buffer = replay_buffer
        
    def train(self):
        states, t_action_probs, t_values = self.replay_buffer.sample(64)
        
        states = torch.FloatTensor(states).to(self.device)
        t_action_probs = torch.FloatTensor(t_action_probs).to(self.device)
        t_values = torch.FloatTensor(t_values).to(self.device)
        
        self.optimizer.zero_grad()
        p_action_probs, p_values = self.model(states)
        loss = F.mse_loss(p_action_probs, t_action_probs) + F.mse_loss(p_values, t_values)
        loss.backward()
        self.optimizer.step()
                   
                   
        
        