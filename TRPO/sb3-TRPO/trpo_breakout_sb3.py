import gymnasium as gym
import ale_py
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


# 注册 ALE 的环境（必须步骤）
gym.register_envs(ale_py)


# 使用 Atari 经典预处理来优化 Breakout 训练效果：
# - 图像预处理：灰度 + 缩放到 84x84（由 make_atari_env 内部包装）
# - 帧堆叠：连续 4 帧作为输入（VecFrameStack）
# - 奖励裁剪：clip_reward=True（内部包装）
# - 多环境并行：n_envs > 1，加快采样，稳定梯度
n_envs = 4
env = make_atari_env(
    "ALE/Breakout-v5",
    n_envs=n_envs,
    seed=0,
    monitor_dir="./breakout_logs",  # 使用 Monitor 记录训练过程
    env_kwargs={"render_mode": "rgb_array"},  # 训练时不直接渲染窗口
)
env = VecFrameStack(env, n_stack=4)


# 使用 TRPO 算法
# 注意：Atari 为图像输入，需要使用 CNN 策略
model = TRPO(
    policy="CnnPolicy",  # 使用 CNN 策略处理图像输入
    env=env,
    learning_rate=1e-4,  # 稍小的学习率，训练更稳定
    gamma=0.99,
    verbose=1,
    target_kl=0.01,  # KL 散度约束目标值
    gae_lambda=0.95,  # 广义优势估计系数
    policy_kwargs=dict(normalize_images=False),
    n_steps=1024,
    batch_size=128,
    device="cuda",
    tensorboard_log="./trpo_breakout_tensorboard",
)


print("开始训练 TRPO 模型（Breakout）...")
model.learn(total_timesteps=1000_000)

# 保存模型
model.save("trpo_breakout_sb3")


# 创建测试环境（与训练相同的预处理，但只用 1 个环境）
env_test = make_atari_env(
    "ALE/Breakout-v5",
    n_envs=1,
    seed=0,
    monitor_dir="./breakout_eval_logs",
    env_kwargs={"render_mode": "rgb_array"},
)
env_test = VecFrameStack(env_test, n_stack=4)

# 加载模型进行测试
loaded_model = TRPO.load("trpo_breakout_sb3", env=env_test)


# 测试模型（在 VecEnv 上评估）
for test_ep in range(20):
    obs = env_test.reset()
    test_reward = 0.0
    done = False
    steps = 0

    # 对于 VecEnv，done 是批量的布尔数组
    while (not done) and steps < 5000:  # 限制最大步数，避免无限循环
        # 确定性预测（关闭噪声，使动作更稳定）
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env_test.step(action)

        # 单环境，索引 0 即可
        test_reward += float(rewards[0])
        done = bool(dones[0])
        steps += 1

    print(f"测试轮 {test_ep + 1} - 总奖励：{test_reward:.1f}, 步数：{steps}")

env.close()
env_test.close()

# tensorboard --logdir trpo_breakout_tensorboard --port 6006
# http://localhost:6006/ 网页看tensorboard