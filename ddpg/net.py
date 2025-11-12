import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        # 网络结构
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        # 正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0.0)

        # 输出层：小权重初始化，确保初始动作接近0
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) # 放缩到[-1,1]
        return x * self.action_bound


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # 网络结构
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # 正交初始化
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0.0)

        # 输出层：小权重初始化
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)