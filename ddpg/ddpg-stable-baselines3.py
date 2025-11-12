import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# 1. 创建训练环境：关闭渲染，提升训练效率
env_train = gym.make("Pendulum-v1", render_mode=None)

# 创建DDPG需要的动作噪声
n_actions = env_train.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 初始化DDPG模型，传入训练环境
model = DDPG("MlpPolicy", env_train, action_noise=action_noise, verbose=1)

print("开始训练（无可视化）...")
model.learn(total_timesteps=10000, log_interval=10)
model.save("ddpg_pendulum")
# 关闭训练环境，释放资源
env_train.close()

# 2. 测试阶段：创建支持human渲染的环境
print("开始测试（启用可视化）...")
env_test = gym.make("Pendulum-v1", render_mode="human")
# 加载训练好的模型
model = DDPG.load("ddpg_pendulum")

for test_ep in range(5):
    obs, _ = env_test.reset()
    test_reward = 0
    done = False
    while not done:
        # 确定性预测（关闭噪声，使动作更稳定）
        action, _states = model.predict(obs, deterministic=True)
        # 与测试环境交互
        obs, reward, terminated, truncated, info = env_test.step(action)
        test_reward += reward
        done = terminated or truncated
    print(f"测试轮{test_ep + 1} 总奖励：{test_reward:.1f}")

# 关闭测试环境
env_test.close()