## TRPO 强化学习实验合集

本仓库包含两类 TRPO 算法实验：

- **基于 `sb3-contrib` 的 TRPO 实践**（CartPole & Breakout）
- **手写 TRPO 算法实现**（CartPole）

通过对比使用成熟库和从零实现 TRPO，可以更好地理解该算法的原理与工程化落地方式。

---

## 目录结构

- **`sb3/`**
  - **`trpo_cartpole_sb3.py`**：使用 `sb3-contrib` 中的 `TRPO` 算法在 `CartPole-v1` 上进行训练与评估  
  - **`trpo_breakout_sb3.py`**：使用 TRPO 训练 Atari `Breakout`，包含 Atari 环境预处理与 TensorBoard 日志记录  
  - **`breakout_logs/`、`breakout_eval_logs/`**：训练与测试过程的监控日志（Monitor CSV）  
  - **`trpo_breakout_tensorboard/`**：TensorBoard 事件文件，用于可视化训练过程  
  - **`*.zip`**：使用 SB3 训练好的模型权重（如 `trpo_breakout_sb3.zip`、`trpo_cartpole_sb3.zip`）

- **`TRPO-CartPole-v1/`**
  - **`agent.py`**：TRPO 算法核心逻辑（策略更新、Fisher 向量积、共轭梯度、线搜索等）  
  - **`networks.py`**：自定义 Actor-Critic 网络结构（全连接 MLP）  
  - **`buffer.py`**：采样数据缓冲区，实现交互数据的存储与张量化  
  - **`hyperparams.py`**：`TRPOConfig`，集中管理超参数与设备选择  
  - **`train.py`**：训练与测试流程（包含 GAE、early stopping、周期评估等逻辑）  
  - **`main.py`**：入口脚本，一键训练并测试 TRPO 在 `CartPole-v1` 上的表现  
  - **`trpo_cartpole_custom.pth`**：手写 TRPO 训练得到的模型权重  

---

## 环境依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：

- **gymnasium**：强化学习环境接口（CartPole、Atari 等）
- **ale-py**：Atari 环境支持（如 `ALE/Breakout-v5`）
- **stable-baselines3**：基础 RL 算法库
- **sb3-contrib**：包含 TRPO 等扩展算法的贡献模块
- **torch**：用于构建和训练深度神经网络
- **numpy**：数值计算
- **tensorboard**：训练过程可视化

完整依赖列表见根目录下的 `requirements.txt`。

---

## 运行示例

### 1. 使用 `sb3-contrib` 的 TRPO

#### CartPole

```bash
cd sb3
python trpo_cartpole_sb3.py
```

脚本会自动完成：

- 在 `CartPole-v1` 上训练 TRPO
- 无渲染评估若干轮次
- 使用人类可视化模式进行测试

#### Breakout (Atari)

```bash
cd sb3
python trpo_breakout_sb3.py
```

训练结束后，可以使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir trpo_breakout_tensorboard --port 6006
```

在浏览器中访问：

```text
http://localhost:6006/
```

---

### 2. 手写 TRPO（CartPole）

```bash
cd TRPO-CartPole-v1
python main.py
```

训练脚本特性：

- 在 `CartPole-v1` 上使用自实现 TRPO 进行训练  
- 使用 GAE（Generalized Advantage Estimation）计算优势  
- 支持 **早停**：当评估平均回报达到设定目标时提前结束训练  
- 按步数周期性评估当前策略性能  

训练完成后：

- 模型会保存到 `trpo_cartpole_custom.pth`
- 会在 `render_mode="human"` 下进行若干轮可视化测试

---

## 项目目的

- **`sb3/` 部分**：演示如何使用 `stable-baselines3` 与 `sb3-contrib` 快速搭建 TRPO 实验，专注于任务配置与实验流程。  
- **`TRPO-CartPole-v1/` 部分**：从零实现 TRPO 的关键细节（KL 约束、共轭梯度、线搜索、GAE 等），帮助深入理解算法本身的数学原理与工程实现。  