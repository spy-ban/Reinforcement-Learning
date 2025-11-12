import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

class SimpleTrainingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []  # 每个episode的总奖励
        self.actor_losses = []  # Actor网络损失
        self.critic_losses = []  # Critic网络损失
        self.current_episode_reward = 0  # 当前episode累计奖励

    def _on_step(self) -> bool:
        # 累加当前episode的奖励
        self.current_episode_reward += self.locals['rewards'][0]

        # 记录网络损失（如果有）
        log_data = self.logger.name_to_value
        if "train/actor_loss" in log_data:
            self.actor_losses.append(log_data["train/actor_loss"])
        if "train/critic_loss" in log_data:
            self.critic_losses.append(log_data["train/critic_loss"])

        # episode结束 记录总的奖励
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

        return True


# 训练阶段
env = gym.make("Pendulum-v1", render_mode=None)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

callback = SimpleTrainingCallback()

model = DDPG("MlpPolicy",env,action_noise=action_noise,verbose=1,learning_starts=1000,train_freq=(1, "episode"))

print("开始训练...")
model.learn(total_timesteps=10000, log_interval=10, callback=callback)
model.save("ddpg_pendulum")
env.close()

# 输出记录结果
print(f"\n训练完成!")
print(f"总episode数: {len(callback.episode_rewards)}")
print(f"最后10个episode平均奖励: {np.mean(callback.episode_rewards[-10:]):.1f}")

# 绘制训练曲线
plt.figure(figsize=(12, 8))

# 子图1：每个episode的奖励
plt.subplot(2, 1, 1)
plt.plot(callback.episode_rewards, linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Rewards")
plt.grid(True, alpha=0.3)

# 子图2：网络损失
plt.subplot(2, 1, 2)
if callback.actor_losses:
    plt.plot(callback.actor_losses, label="Actor Loss", alpha=0.7)
if callback.critic_losses:
    plt.plot(callback.critic_losses, label="Critic Loss", alpha=0.7)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Network Losses")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
plt.show()


# 测试阶段
print("\n开始测试...")
env_test = gym.make("Pendulum-v1", render_mode="human")
model = DDPG.load("ddpg_pendulum")

for test_ep in range(5):
    obs, _ = env_test.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"测试 Episode {test_ep + 1}: 总奖励 = {total_reward:.1f}")

env_test.close()