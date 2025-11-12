import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from net import Actor, Critic
import HyperParams
from ReplayBuffer import ReplayBuffer
import random

def set_seed(seed=42):
    """设置所有随机种子以确保结果可重复"""
    # Python随机种子
    random.seed(seed)
    # Numpy随机种子
    np.random.seed(seed)
    # PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # PyTorch确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DDPGAgent:

    def __init__(self, state_dim, action_dim, action_bound):
        # 1. 初始化 Actor
        self.actor = Actor(state_dim, action_dim, action_bound).to(HyperParams.DEVICE)
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(HyperParams.DEVICE)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=HyperParams.LR_ACTOR)

        # 2. 初始化 Critic
        self.critic = Critic(state_dim, action_dim).to(HyperParams.DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(HyperParams.DEVICE)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=HyperParams.LR_CRITIC)

        # 3. 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(HyperParams.BUFFER_SIZE)

        # 4. 目标网络参数初始化
        self.soft_update(tau=1.0)  # 初始化目标网络等于主网络
        self.actor_losses = []
        self.critic_losses = []

    def soft_update(self, tau=HyperParams.TAU):
        """软更新目标网络：target = tau*main + (1-tau)*target"""
        # Actor 网络软更新
        for main_param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)
        # Critic 网络软更新
        for main_param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)

    def select_action(self, s, is_training=True):
        """选择动作：训练时加噪声探索，测试时纯策略输出"""
        s_tensor = torch.FloatTensor(s).unsqueeze(0).to(HyperParams.DEVICE)

        self.actor.eval()  # 评估模式
        with torch.no_grad():
            action = self.actor(s_tensor).cpu().numpy()[0]  # 输出动作
        self.actor.train()  # 恢复训练模式

        # 训练时添加高斯噪声（探索），并裁剪到动作边界
        if is_training:
            noise = np.random.normal(0, HyperParams.NOISE_SCALE, size=action.shape)
            action = np.clip(action + noise, -action_bound, action_bound)

        return action

    def update(self):
        """从经验缓冲区采样并训练网络"""
        # 采样批量经验（样本不足时跳过）
        batch = self.replay_buffer.sample_batch(HyperParams.BATCH_SIZE)
        if batch is None:
            return

        # 转换批量数据为tensor（适配网络输入）
        s, a, r, s_next, done = (torch.FloatTensor(np.array(elem)).to(HyperParams.DEVICE) for elem in zip(*batch))
        r, done = r.unsqueeze(1), done.unsqueeze(1)

        # -------------------------- 训练critic网络 --------------------------
        # 计算目标Q值：r + gamma*(1-done)*Q_target(s_next, a_next)
        a_next = self.actor_target(s_next)  # 目标演员输出-next动作
        q_target = r + HyperParams.GAMMA * (1 - done) * self.critic_target(s_next, a_next)
        # 计算预测Q值
        q_pred = self.critic(s, a)
        # 最小化MSE损失
        critic_loss = nn.MSELoss()(q_pred, q_target.detach())  # detach冻结目标网络
        self.critic_losses.append(critic_loss.item())

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # -------------------------- 训练actor网络 --------------------------
        # 最大化Q值（策略梯度：通过负号转为梯度上升）
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_losses.append(actor_loss.item())

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # -------------------------- 软更新目标网络 --------------------------
        self.soft_update()


# -------------------------- 训练与测试主逻辑 --------------------------
if __name__ == "__main__":
    set_seed(HyperParams.SEED)
    # 1. 创建连续控制环境（Pendulum-v1）
    env = gym.make("Pendulum-v1", render_mode=None)
    env.reset(seed=HyperParams.SEED)

    state_dim = env.observation_space.shape[0]  # 状态维度：3
    action_dim = env.action_space.shape[0]  # 动作维度：1
    action_bound = env.action_space.high[0]  # 动作边界：2（[-2,2]）

    # 2. 初始化DDPG智能体
    agent = DDPGAgent(state_dim, action_dim, action_bound)

    # 3. 训练循环
    total_rewards = []  # 记录每轮奖励
    for episode in range(HyperParams.EPISODES):
        s, _ = env.reset()  # 重置环境，获取初始状态
        episode_reward = 0  # 累计当前轮奖励

        for step in range(HyperParams.MAX_STEPS):
            # 选择动作并执行
            a = agent.select_action(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated  # 终止条件（超时/任务完成）

            # 存储经验到缓冲区
            agent.replay_buffer.add(s, a, r, s_next, done)

            # 训练网络
            agent.update()

            # 更新状态和累计奖励
            s = s_next
            episode_reward += r

            if done:
                break

        # 记录并打印训练进度
        total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards[-10:])  # 最近10轮平均奖励（判断收敛）

        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode + 1:4d} | 单轮奖励: {episode_reward:6.1f} | 近10轮平均: {avg_reward:6.1f}")

    torch.save(agent.actor.state_dict(), "ddpg_actor_pendulum100.pth")
    import matplotlib.pyplot as plt

    # 绘制奖励曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(total_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # 绘制损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(agent.actor_losses)
    plt.title('Actor Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 3)
    plt.plot(agent.critic_losses)
    plt.title('Critic Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_curves100.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 测试训练好的模型（可视化）
    print("\n开始测试训练好的模型...")
    env_test = gym.make("Pendulum-v1", render_mode="human")  # 开启可视化
    agent.actor.load_state_dict(torch.load("ddpg_actor_pendulum100.pth"))  # 加载模型
    agent.actor.eval()  # 评估模式（不加噪声）

    for test_ep in range(5):  # 测试5轮
        s, _ = env_test.reset()
        test_reward = 0
        for _ in range(HyperParams.MAX_STEPS):
            a = agent.select_action(s, is_training=False)  # 纯策略输出（无噪声）
            s, r, terminated, truncated, _ = env_test.step(a)
            test_reward += r
            if terminated or truncated:
                break
        print(f"测试轮{test_ep + 1} 奖励：{test_reward:.1f}")

    env.close()
    env_test.close()