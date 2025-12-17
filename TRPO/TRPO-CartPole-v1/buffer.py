import numpy as np
import torch
from typing import Tuple


class TransitionBuffer:
    """存储环境交互数据的缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        
    def add(self, state, action, next_state, reward, done):
        """添加一条经验"""
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.next_states.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def size(self):
        """返回缓冲区大小"""
        return len(self.states)
    
    def to_tensor(self, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """将缓冲区数据转换为PyTorch张量"""
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(device)
        dones = torch.BoolTensor(np.array(self.dones)).to(device)
        return states, actions, next_states, rewards, dones

