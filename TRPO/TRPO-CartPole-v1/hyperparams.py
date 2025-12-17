import torch
from typing import List


class TRPOConfig:
    """TRPO算法超参数配置"""
    # 通用RL参数
    gamma: float = 0.99
    rollout_steps: int = 1000
    batch_size: int = 64
    
    # GAE参数
    lambda_: float = 0.95
    norm_adv: bool = True
    use_td_lambda: bool = True
    
    # 共轭梯度参数
    residual_tol: float = 1e-6
    cg_steps: int = 10
    damping: float = 0.1
    
    # 线搜索参数
    delta: float = 0.01  # KL散度约束上限
    beta: float = 0.5  # 步长衰减系数
    max_backtrack: int = 10
    accept_ratio: float = 0.1
    
    # 网络结构参数
    net_arch: List[int] = [64, 64]
    activation_fn: str = "ReLU"
    
    # 价值网络参数
    lr: float = 3e-4
    optimizer: str = "Adam"
    n_update: int = 10
    
    def __init__(self):
        """初始化时设置设备"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

