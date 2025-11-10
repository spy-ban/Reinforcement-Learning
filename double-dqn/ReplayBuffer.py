# 经验回放（Experience Replay）存储智能体与环境交互中产生的经验，并从中随机抽取一批经验进行训练
from collections import deque
from random import sample
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # 使用双端队列（deque）来存储经验，最大长度为capacity

    def add(self, state, action, reward, next_state, terminated, truncated):
        self.buffer.append([state, action, reward, next_state, terminated, truncated])
        # 将一个经验元组添加到缓冲区中

    def sample_batch(self, batch_size):
        transitions = sample(self.buffer, batch_size) # 从缓冲区中随机抽取一批经验
        return transitions

    def size(self):
        return len(self.buffer)