import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
from torch.optim import Adam
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import grad
from typing import Tuple, Optional, Callable
import os

from hyperparams import TRPOConfig
from networks import Actor, Critic
from buffer import TransitionBuffer


class TRPOAgent:
    """TRPO算法Agent（适配离散动作空间）"""
    
    def __init__(self, state_dim: int, action_dim: int, config: TRPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化网络
        self.actor = Actor(state_dim, action_dim, config.net_arch).to(self.device)
        self.critic = Critic(state_dim, config.net_arch).to(self.device)
        
        # 价值网络优化器
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.lr)
        
        # 统计数据
        self.stats = {}
        
        # 优势函数（在update时计算）
        self.adv = None
    
    def select_action(self, state: np.ndarray, deterministic: bool = False, 
                     return_log_prob: bool = False) -> Tuple[int, Optional[float]]:
        """选择动作（适配离散动作空间）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_dist = self.actor.get_distribution(state_tensor)
            
            if deterministic:
                # 测试阶段：选择概率最大的动作
                action = action_dist.probs.argmax(dim=-1).item()
                log_prob = None
            else:
                # 训练阶段：从分布中采样
                action = action_dist.sample().item()
                if return_log_prob:
                    log_prob = action_dist.log_prob(torch.tensor(action)).item()
                else:
                    log_prob = None
        
        return action, log_prob
    
    def _compute_gae(self, states: torch.Tensor, rewards: torch.Tensor, 
                    next_states: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计（GAE）"""
        with torch.no_grad():
            # 预测状态价值
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            # 计算TD误差
            td_errors = rewards + self.config.gamma * next_values * (~dones) - values
            
            # 计算优势函数（GAE）
            advantages = torch.zeros_like(td_errors)
            last_gae = 0
            
            # 从后向前累积
            for t in reversed(range(len(td_errors))):
                if dones[t]:
                    last_gae = 0
                last_gae = td_errors[t] + self.config.gamma * self.config.lambda_ * last_gae
                advantages[t] = last_gae
            
            # 计算折扣回报（作为价值网络的目标）
            returns = advantages + values
            
            # 归一化优势函数（可选）
            if self.config.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _get_surrogate_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor) -> torch.Tensor:
        """计算替代损失"""
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * self.adv)
    
    def _select_action_dist(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """获取动作分布和log_prob（用于策略更新）"""
        action_dist = self.actor.get_distribution(states)
        log_probs = action_dist.log_prob(actions)
        return action_dist, log_probs
    
    def _update_actor(self, states: torch.Tensor, actions: torch.Tensor):
        """更新策略网络（TRPO核心算法）"""
        # 步骤1：备份原始策略参数
        original_actor_param = parameters_to_vector(self.actor.parameters()).data.clone()
        
        # 步骤2：计算策略梯度（PG）
        action_dist, log_probs = self._select_action_dist(states, actions)
        old_action_dist = Categorical(logits=self.actor(states).data.clone())  # 备份旧分布
        old_log_probs = log_probs.data.clone()
        
        # 计算替代损失
        loss = self._get_surrogate_loss(log_probs, old_log_probs)
        
        # 计算策略梯度
        pg = grad(loss, self.actor.parameters(), retain_graph=True)
        pg = parameters_to_vector(pg).detach()
        
        # 步骤3：计算KL散度的Hessian-vector乘积
        # 计算KL散度（离散动作：类别分布的KL）
        kl = torch.mean(kl_divergence(old_action_dist, action_dist))
        kl_g = grad(kl, self.actor.parameters(), create_graph=True)
        kl_g = parameters_to_vector(kl_g)
        
        # 共轭梯度法求解 H^{-1} * PG
        update_dir = self._conjugate_gradient(kl_g, pg)
        
        # 计算 Fvp = H * update_dir
        Fvp = self._Fvp_func(kl_g, update_dir)
        
        # 步骤4：计算最大步长
        full_step_size = torch.sqrt(2 * self.config.delta / (torch.dot(update_dir, Fvp) + 1e-8))
        
        # 步骤5：回溯线搜索确定最优步长
        self.stats.update({"loss/actor": 0.0})
        
        def check_constrain(alpha: float) -> bool:
            """检查约束条件"""
            step = alpha * full_step_size * update_dir
            with torch.no_grad():
                # 更新参数
                vector_to_parameters(original_actor_param + step, self.actor.parameters())
                
                try:
                    # 计算新策略的分布和损失
                    new_action_dist, new_log_probs = self._select_action_dist(states, actions)
                except:
                    # 如果出错，恢复参数
                    vector_to_parameters(original_actor_param, self.actor.parameters())
                    return False
                
                # 计算新损失和KL散度
                new_loss = self._get_surrogate_loss(new_log_probs, old_log_probs)
                new_kl = torch.mean(kl_divergence(old_action_dist, new_action_dist))
                actual_improve = new_loss - loss
            
            # 检查约束：损失提升且KL散度在约束范围内
            if actual_improve.item() > 0.0 and new_kl.item() <= self.config.delta:
                self.stats.update({"loss/actor": new_loss.item()})
                return True
            else:
                # 不满足约束，恢复参数
                vector_to_parameters(original_actor_param, self.actor.parameters())
                return False
        
        # 执行线搜索
        alpha = self._line_search(check_constrain)
        
        # 应用找到的步长（如果alpha=0，则参数已经恢复为原始值）
        if alpha > 0:
            vector_to_parameters(
                original_actor_param + alpha * full_step_size * update_dir,
                self.actor.parameters()
            )
    
    def _conjugate_gradient(self, kl_g: torch.Tensor, pg: torch.Tensor) -> torch.Tensor:
        """共轭梯度法求解 H^{-1} * g"""
        x = torch.zeros_like(pg)
        r = pg.clone()
        p = pg.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(self.config.cg_steps):
            _Fvp = self._Fvp_func(kl_g, p)
            alpha = rdotr / (torch.dot(p, _Fvp) + 1e-8)
            x += alpha * p
            r -= alpha * _Fvp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < self.config.residual_tol:
                break
        
        return x
    
    def _Fvp_func(self, kl_g: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Fisher vector product: H * p，其中H是KL散度的Hessian矩阵"""
        gvp = torch.dot(kl_g, p)
        Hvp = grad(gvp, self.actor.parameters(), retain_graph=True)
        Hvp = parameters_to_vector(Hvp).detach()
        # 添加阻尼项稳定计算
        Hvp += self.config.damping * p
        return Hvp
    
    def _line_search(self, check_constrain: Callable) -> float:
        """回溯线搜索"""
        alpha = 1.0 / self.config.beta
        for _ in range(self.config.max_backtrack):
            alpha *= self.config.beta
            if check_constrain(alpha):
                return alpha
        return 0.0
    
    def _update_value_net(self, states: torch.Tensor, returns: torch.Tensor):
        """更新价值网络"""
        total_loss = 0
        total_batches = 0
        
        for _ in range(self.config.n_update):
            # 随机打乱索引
            indices = torch.randperm(len(states))
            
            # 分批更新
            for i in range(0, len(states), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_states = states[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                values = self.critic(batch_states)
                
                # 计算MSE损失
                loss = F.mse_loss(values, batch_returns)
                
                # 反向传播
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
        
        if total_batches > 0:
            self.stats.update({"loss/critic": total_loss / total_batches})
    
    def update(self, buffer: TransitionBuffer):
        """执行一次策略和价值网络更新"""
        # 转换数据为张量
        states, actions, next_states, rewards, dones = buffer.to_tensor(str(self.device))
        
        # 计算GAE
        advantages, returns = self._compute_gae(states, rewards, next_states, dones)
        self.adv = advantages  # 保存优势函数供策略更新使用
        
        # 更新策略网络
        self._update_actor(states, actions)
        
        # 更新价值网络
        self._update_value_net(states, returns)
    
    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'config': self.config
        }, filepath)
        print(f"模型已保存至: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"模型已从 {filepath} 加载")

