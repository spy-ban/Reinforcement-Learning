import torch
# 训练相关
EPISODES = 100  # 总训练轮数
MAX_STEPS = 200  # 每轮最大步数
BATCH_SIZE = 64  # 批量采样大小
GAMMA = 0.99  # 折扣因子（未来奖励权重）

# 网络更新相关
TAU = 0.005  # 目标网络软更新系数（tau越小更新越平滑）
LR_ACTOR = 1e-4  # 演员网络学习率
LR_CRITIC = 1e-3  # 评论员网络学习率

# 经验回放
BUFFER_SIZE = 10000  # 经验缓冲区最大容量

# 噪声探索
NOISE_SCALE = 0.1  # 高斯噪声强度（平衡探索与利用）
SEED = 42

# 设备配置（自动识别GPU/CPU）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"