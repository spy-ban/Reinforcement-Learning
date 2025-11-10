import torch

class HyperParams:
    lr = 3e-3  # 学习率
    num_episodes = 100  # 训练回合数
    hidden_dim = 128  # 神经网络隐藏层的维度
    gamma = 0.97  # 折扣因子，控制未来奖励的折扣程度
    eps = 0.01  # epsilon-greedy 策略中的 epsilon（探索的概率）
    target_update_frequency = 10  # 每多少回合更新目标网络的权重
    buffer_capacity = 1000  # 经验回放缓存的容量
    batch_size = 64  # 每次训练时抽取的经验批量大小
    minimal_size = 400  # 经验回放缓存的最小大小，直到可以开始训练
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")