import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List


class Actor(nn.Module):
    """策略网络（Actor）- 适配离散动作"""
    
    def __init__(self, state_dim: int, action_dim: int, net_arch: List[int] = [64, 64]):
        super(Actor, self).__init__()
        layers = []
        input_dim = state_dim
        
        # 构建隐藏层
        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 输出层（动作logits）
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回动作logits"""
        return self.net(state)
    
    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """获取动作分布"""
        logits = self.forward(state)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """价值网络（Critic）"""
    
    def __init__(self, state_dim: int, net_arch: List[int] = [64, 64]):
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim
        
        # 构建隐藏层
        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 输出层（状态价值）
        layers.append(nn.Linear(input_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回状态价值"""
        return self.net(state).squeeze(-1)

