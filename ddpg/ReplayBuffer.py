from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)  # 固定容量，超容自动丢弃旧数据

    def add(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample_batch(self, batch_size):
        """随机采样批量经验""" 
        if len(self.buffer) < batch_size:
            return None  # 样本不足时返回None
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)