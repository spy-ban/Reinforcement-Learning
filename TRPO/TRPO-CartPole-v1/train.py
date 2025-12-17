import gymnasium as gym
import numpy as np

from hyperparams import TRPOConfig
from agent import TRPOAgent
from buffer import TransitionBuffer


def evaluate_agent(agent: TRPOAgent, n_episodes: int = 5, render_mode: str = "rgb_array") -> float:
    """评估agent的性能，返回平均奖励"""
    env = gym.make("CartPole-v1", render_mode=render_mode)
    episode_rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
    
    env.close()
    return np.mean(episode_rewards)


def train_trpo(max_timesteps: int = 50000, 
                model_path: str = "trpo_cartpole_custom.pth",
                target_score: float = 475.0,
                eval_frequency: int = 1000,
                eval_episodes: int = 5,
                early_stop: bool = True):
    """训练TRPO算法（支持早停）
    
    Args:
        max_timesteps: 最大训练步数
        model_path: 模型保存路径
        target_score: 目标平均分数（达到此分数可 early stop）
        eval_frequency: 评估频率（每多少步评估一次）
        eval_episodes: 每次评估的episode数
        early_stop: 是否启用早期停止
    """
    # 创建环境
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 初始化配置和Agent
    config = TRPOConfig()
    agent = TRPOAgent(state_dim, action_dim, config)
    
    # 初始化数据缓冲
    buffer = TransitionBuffer()
    
    # 训练循环
    obs, _ = env.reset()
    total_steps = 0
    episode_num = 0
    last_eval_step = 0
    best_score = 0.0
    
    print("开始训练...")
    print(f"目标分数: {target_score}, 评估频率: 每 {eval_frequency} 步, 早期停止: {'启用' if early_stop else '禁用'}")
    
    while total_steps < max_timesteps:
        # 收集数据
        action, _ = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 存入缓冲
        buffer.add(obs, action, next_obs, reward, done)
        total_steps += 1
        
        obs = next_obs
        
        if done:
            obs, _ = env.reset()
            episode_num += 1
            if episode_num % 10 == 0:
                print(f"Episode {episode_num}, Total Steps: {total_steps}")
        
        # 当缓冲满时更新网络
        if buffer.size() >= config.rollout_steps:
            agent.update(buffer)
            buffer.clear()
            
            # 打印训练统计
            if 'loss/actor' in agent.stats and 'loss/critic' in agent.stats:
                print(f"Steps: {total_steps}, Actor Loss: {agent.stats['loss/actor']:.4f}, "
                      f"Critic Loss: {agent.stats['loss/critic']:.4f}")
        
        # 定期评估并检查早期停止条件
        if early_stop and total_steps - last_eval_step >= eval_frequency:
            last_eval_step = total_steps
            avg_score = evaluate_agent(agent, n_episodes=eval_episodes, render_mode="rgb_array")
            best_score = max(best_score, avg_score)
            print(f"[评估] Steps: {total_steps}, 平均分数: {avg_score:.1f}, 最佳分数: {best_score:.1f}")
            
            if avg_score >= target_score:
                print(f"\n✓ 达到目标分数 {target_score:.1f}！平均分数: {avg_score:.1f}")
                print(f"训练提前停止在 {total_steps} 步（最大步数: {max_timesteps}）")
                break
    
    # 最终评估
    print("\n进行最终评估...")
    final_score = evaluate_agent(agent, n_episodes=eval_episodes, render_mode="rgb_array")
    print(f"最终平均分数: {final_score:.1f}")
    
    # 保存模型
    agent.save(model_path)
    env.close()
    
    return agent, model_path


def test_trpo(model_path: str, n_episodes: int = 5):
    """测试训练好的TRPO模型"""
    # 创建测试环境
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 加载模型
    config = TRPOConfig()
    agent = TRPOAgent(state_dim, action_dim, config)
    agent.load(model_path)
    
    print(f"\n开始测试（{n_episodes}轮）...")
    
    for test_ep in range(n_episodes):
        obs, _ = env.reset()
        test_reward = 0
        done = False
        
        while not done:
            # 确定性预测（关闭随机性）
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            test_reward += reward
            done = terminated or truncated
        
        print(f"测试轮 {test_ep + 1} 总奖励：{test_reward:.1f}")
    
    env.close()