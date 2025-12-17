from train import train_trpo, test_trpo


if __name__ == "__main__":
    # 训练（支持早期停止）
    agent, model_path = train_trpo(
        max_timesteps=50000,           # 最大训练步数（作为上限）
        target_score=475.0,            # 目标平均分数（达到即可停止）
        eval_frequency=1000,           # 每1000步评估一次
        eval_episodes=5,               # 每次评估5个episode
        early_stop=True                # 启用早期停止
    )
    
    # 最终测试（可视化）
    test_trpo(model_path, n_episodes=5)

