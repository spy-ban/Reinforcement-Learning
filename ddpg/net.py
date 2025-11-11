import torch
import torch.nn as nn

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound  # 动作最大值（如Pendulum-v1的2）

        # 网络结构：输入层 → BN → 隐藏层1 → BN → 隐藏层2 → BN → 输出层
        self.fc1 = nn.Linear(state_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)  # 对隐藏层1输出做BN（输入维度=64）

        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)  # 对隐藏层2输出做BN

        self.fc3 = nn.Linear(64, action_dim)

        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01)  # 小标准差，避免极端输出
        nn.init.constant_(self.fc3.bias, 0.0)  # 偏置初始化为0

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        # 输入：x.shape = (batch_size, state_dim)
        x = self.fc1(x)
        x = self.bn1(x)  # BN层需在激活函数前（论文推荐：线性层→BN→激活）
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = torch.tanh(self.fc3(x))  # 输出[-1,1]
        return x * self.action_bound  # 缩放至动作空间范围


class Critic(nn.Module):
    """评论员网络（含批归一化，遵循DDPG论文设计）"""

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # 网络结构：状态+动作拼接 → 线性层 → BN → 隐藏层1 → BN → 隐藏层2 → 输出层
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)  # 对拼接后的输入做BN

        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)  # 对隐藏层1输出做BN

        self.fc3 = nn.Linear(64, 1)  # 输出单个Q值

        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01)  # 小标准差，避免极端输出
        nn.init.constant_(self.fc3.bias, 0.0)  # 偏置初始化为0

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, s, a):
        # 输入：s.shape=(batch_size, state_dim), a.shape=(batch_size, action_dim)
        x = torch.cat([s, a], dim=1)  # 拼接后：(batch_size, state_dim+action_dim)

        x = self.fc1(x)
        x = self.bn1(x)  # 线性层→BN→激活（论文标准流程）
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        return self.fc3(x)