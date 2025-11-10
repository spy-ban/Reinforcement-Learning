from numpy.random import random, randint
import numpy as np
import gymnasium as gym
from torch import tensor, float32, empty, int64
from torch.nn.functional import mse_loss
from torch.optim import Adam
from Qnet import Qnet
from ReplayBuffer import ReplayBuffer
from HyperParams import HyperParams

class DQN:
    def __init__(self, state_dim, hidden_dim, action_card, lr, gamma, eps, target_update_frequency, device="cuda:0"):
        # 初始化 DQN 智能体的参数
        self.action_card = action_card  # 动作空间的大小
        self.state_dim = state_dim  # 状态空间的维度
        self.gamma = gamma  # 折扣因子，用于计算未来奖励
        self.eps = eps  # epsilon-greedy 策略中的 epsilon，用于平衡探索和利用
        self.target_update_frequency = target_update_frequency  # 更新目标网络的频率
        self.count = 0  # 记录目标网络更新的次数
        self.device = device  # 设备选择，通常是 GPU 或 CPU

        # 创建 Q 网络和目标网络，分别用于当前的 Q 值估计和目标 Q 值的计算
        self.q_net = Qnet(self.state_dim, hidden_dim, self.action_card).to(device)
        self.target_net = Qnet(self.state_dim, hidden_dim, self.action_card).to(device)

        # 使用 Adam 优化器来优化 Q 网络
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)

    def take_action(self, state):
        # 以 eps 的概率选择随机动作（探索）
        if random() < self.eps:
            action = randint(self.action_card)
        else: #以 1 - eps 的概率选择当前 Q 网络输出的最佳动作
            state = tensor(state, dtype=float32).to(self.device).unsqueeze(0) # 获取当前状态
            action = self.q_net(state).argmax().item() # 获取当前状态下的Q*对应最佳动作
        return action

    def update(self, transitions):
        # 从经验回放缓存中抽取一批经验进行训练
        batch_size = len(transitions)
        # empty创建张量
        states, actions, rewards, next_states, terminateds, truncateds = empty((batch_size, self.state_dim),
            device=self.device), empty(batch_size, dtype=int64, device=self.device), empty(batch_size, dtype=int64,
            device=self.device), empty((batch_size, self.state_dim), device=self.device), empty(batch_size, dtype=int64,
            device=self.device), empty(batch_size, dtype=int64, device=self.device)
        for i, transition in zip(range(batch_size), transitions):
            states[i], actions[i], rewards[i], next_states[i], terminateds[i], truncateds[i] = tensor(
                transition[0]), tensor(transition[1]), tensor(transition[2]), tensor(transition[3]), tensor(
                transition[4]), tensor(transition[5])

        # 损失函数 计算 DQN 损失
        # 1. 用Q_net选择动作 最大的Q值对应的动作
        next_actions = self.q_net(next_states).max(1)[1]
        # 2. 用target_net评估该动作的价值
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        dqn_loss = mse_loss(
            self.q_net(states).gather(1, actions.view(-1, 1)).flatten(), # Q(s,a) q_net(s) 中选择动作 a 对应的 Q 值
            rewards + self.gamma * next_q_values * (1 - terminateds) * (1 - truncateds) # 下一步目标网络的 Q*
        )

        # 反向传播
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update_frequency == 0: # 一段时间后更新目标网络
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    replay_buffer = ReplayBuffer(HyperParams.buffer_capacity)

    state_dim = env.observation_space.shape[0]
    action_card = env.action_space.n

    agent = DQN(state_dim, HyperParams.hidden_dim, action_card, HyperParams.lr, HyperParams.gamma, HyperParams.eps,
                HyperParams.target_update_frequency, HyperParams.device)

    returns = []

    for i in range(HyperParams.num_episodes):
        episode_return = 0
        state = env.reset()[0]
        while True:
            action = agent.take_action(state) # 获取动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            # 状态转移
            replay_buffer.add(state, action, reward, next_state, terminated, truncated)
            episode_return += reward
            if terminated or truncated:
                break
            state = next_state

            if replay_buffer.size() > HyperParams.minimal_size:
                # 如果回放缓存中积累了足够的经验 开始更新网络
                transitions = replay_buffer.sample_batch(HyperParams.batch_size)
                agent.update(transitions)
        returns.append(episode_return)
        if (i + 1) % 10 == 0:
            print("episode:{}-{}, episode_return:{}.".format(i-8, i+1, np.mean(returns[-10:])))

    env.close()

    # 用训练好的DQN 可视化跑倒立杆
    env = gym.make("CartPole-v1", render_mode="human")  # 创建环境并设置渲染模式为 "human"
    for i in range(10):  # 运行10个回合进行评估
        state = env.reset()[0]  # 获取初始状态
        episode_reward = 0  # 每个回合的总奖励
        while True:
            env.render()  # 渲染环境以显示图形界面
            action = agent.take_action(state)  # 获取智能体的动作
            next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作并获取反馈
            episode_reward += reward  # 累加当前回合的奖励

            # 如果回合结束或达到最大步数，则跳出循环
            if terminated or truncated:
                print(f"Episode {i + 1} finished! Total Reward: {episode_reward}")
                break
            state = next_state  # 更新状态
