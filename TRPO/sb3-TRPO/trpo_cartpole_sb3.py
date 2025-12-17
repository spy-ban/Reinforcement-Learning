import gymnasium as gym
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = TRPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
    n_steps=2048,
)

model.learn(total_timesteps=10000)  # 训练步数
model.save("trpo_cartpole_sb3")
model.env.close()

# 评估（无渲染）
eval_env = gym.make("CartPole-v1")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Eval reward: {mean_reward:.1f} ± {std_reward:.1f}")
eval_env.close()

# 可视化测试
test_env = gym.make("CartPole-v1", render_mode="human")
loaded_model = TRPO.load("trpo_cartpole_sb3", env=test_env)

for ep in range(5):
    obs, _ = test_env.reset()
    ep_rew, done = 0.0, False
    while not done:
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        ep_rew += reward
        done = terminated or truncated
    print(f"测试轮 {ep+1} 总奖励: {ep_rew:.1f}")
test_env.close()