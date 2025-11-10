from torch.nn import Linear, Module, LayerNorm, LeakyReLU, Dropout
from torch.nn.init import kaiming_normal_

class Qnet(Module):
    # 输入：[批量大小, 状态维度]
    # 输出：[批量大小, 动作维度]
    def __init__(self, state_dim, hidden_dim, action_card):
        super(Qnet, self).__init__()
        self.fc1 = Linear(state_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, action_card)

        self.leaky_relu = LeakyReLU(negative_slope=0.01)
        self.dropout = Dropout(0.5)
        self.layer_norm1 = LayerNorm(hidden_dim)  # 使用 LayerNorm
        self.layer_norm2 = LayerNorm(hidden_dim)

        kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.layer_norm1(self.fc1(x))  # 批归一化
        x = self.leaky_relu(x)  # LeakyReLU 激活函数
        x = self.dropout(x)  # Dropout 防止过拟合
        x = self.layer_norm2(self.fc2(x))  # 第二层批归一化
        x = self.leaky_relu(x)  # LeakyReLU 激活函数
        x = self.fc3(x)  # 输出层
        return x