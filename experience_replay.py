from collections import deque
import random

import numpy as np
import torch


def unpack_batch(transitions):
    """把一批 transition 拆成 DQN 训练需要的 batch tensor。

    输入:
        [
            (state, action, next_state, reward, done),
            ...
        ]

    输出:
        states.shape      == [batch_size, state_dim]
        actions.shape     == [batch_size]
        next_states.shape == [batch_size, state_dim]
        rewards.shape     == [batch_size]
        dones.shape       == [batch_size]
    """
    states, actions, next_states, rewards, dones = zip(*transitions)

    return (
        torch.from_numpy(np.asarray(states, dtype=np.float32)),
        torch.tensor(actions, dtype=torch.int64),
        torch.from_numpy(np.asarray(next_states, dtype=np.float32)),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(dones, dtype=torch.bool),
    )


class ReplayMemory:
    """DQN 使用的经验回放池。

    宏观作用:
        DQN 不直接用“刚刚发生的一步”立刻训练，而是把很多历史经验存起来，
        训练时随机抽取一个 mini-batch。这样可以降低样本之间的连续相关性，
        让神经网络看到更丰富、更分散的训练样本。

    每条经验 transition 的内容沿用教程顺序:
        (state, action, next_state, reward, done)

    含义:
        state      : 当前状态 s
        action     : 在当前状态采取的动作 a
        next_state : 执行动作后进入的新状态 s'
        reward     : 执行动作后得到的即时奖励 r
        done       : 这一局是否结束，True 表示 s' 是终止状态

    注意:
        ReplayMemory 只负责“存”和“随机取”。
        它不计算 Q 值，也不做 value update。
    """

    def __init__(self, maxlen: int, seed: int | None = None):
        # deque(maxlen=...) 是一个固定容量队列。
        # 当容量满了以后，再 append 新经验时，最旧的经验会自动被挤掉。
        self.memory = deque([], maxlen=maxlen)

        # seed 只用于让 sample 的随机结果可复现，方便调试和教学检查。
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        """存入一条经验。

        transition 通常是:
            (state, action, next_state, reward, done)
        """
        self.memory.append(transition)

    def sample(self, sample_size: int):
        """随机抽取 sample_size 条经验。

        DQN 后续会用这些样本组成 mini-batch，计算 TD target:
            target_q = reward + gamma * max_a' Q_target(next_state, a')

        如果 done=True，说明 next_state 已经是终止状态，
        那么后面的未来价值项会被去掉。
        """
        return random.sample(self.memory, sample_size)

    def __len__(self):
        # 让我们可以直接写 len(memory)，判断当前存了多少条经验。
        return len(self.memory)


class PrioritizedReplayMemory:
    """按 TD error 优先采样的经验回放池。

    普通 ReplayMemory 是完全均匀随机抽样。Prioritized Replay 的直觉是:
        哪些 transition 当前“预测错得更厉害”，就更值得被模型多看几次。

    这里用 proportional prioritization:
        P(i) = priority_i ** alpha / sum(priority ** alpha)

    beta 用来计算 importance sampling weights，抵消非均匀采样带来的偏差。
    """

    def __init__(
        self,
        maxlen: int,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-5,
        seed: int | None = None,
    ):
        self.maxlen = maxlen
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.memory = []
        self.priorities = np.zeros(maxlen, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

        self.rng = np.random.default_rng(seed)

    def append(self, transition):
        if len(self.memory) < self.maxlen:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.maxlen

    def sample(self, sample_size: int, beta: float = 0.4):
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from an empty replay memory.")

        active_priorities = self.priorities[: len(self.memory)]
        scaled_priorities = active_priorities ** self.alpha
        probabilities = scaled_priorities / scaled_priorities.sum()

        indices = self.rng.choice(
            len(self.memory),
            size=sample_size,
            replace=len(self.memory) < sample_size,
            p=probabilities,
        )
        transitions = [self.memory[index] for index in indices]

        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return transitions, indices, weights

    def update_priorities(self, indices, td_errors):
        td_errors = np.asarray(td_errors, dtype=np.float32)
        new_priorities = np.abs(td_errors) + self.priority_epsilon

        for index, priority in zip(indices, new_priorities):
            self.priorities[index] = priority

        self.max_priority = max(self.max_priority, float(new_priorities.max()))

    def __len__(self):
        return len(self.memory)


if __name__ == "__main__":
    # 下面是 ReplayMemory 的最小行为检查。
    # 这里先不用真实环境，只构造几条假的 transition。
    memory = ReplayMemory(maxlen=3, seed=0)

    # 假设 state_dim=12，先用 list 模拟状态。
    state = [0.0] * 12
    next_state = [0.1] * 12

    memory.append((state, 0, next_state, 0.1, False))
    memory.append((state, 1, next_state, 0.1, False))
    memory.append((state, 0, next_state, -1.0, True))
    memory.append((state, 1, next_state, 1.0, False))

    # maxlen=3，所以第 4 次 append 后，最早的那条经验会被自动删除。
    print("memory length:", len(memory))
    assert len(memory) == 3

    batch = memory.sample(2)
    print("sample size:", len(batch))
    assert len(batch) == 2

    first_transition = batch[0]
    print("transition fields:", len(first_transition))
    assert len(first_transition) == 5

    states, actions, next_states, rewards, dones = unpack_batch(batch)

    print("states shape:", states.shape)
    print("actions shape:", actions.shape)
    print("next_states shape:", next_states.shape)
    print("rewards shape:", rewards.shape)
    print("dones shape:", dones.shape)

    assert states.shape == (2, 12)
    assert actions.shape == (2,)
    assert next_states.shape == (2, 12)
    assert rewards.shape == (2,)
    assert dones.shape == (2,)

    prioritized_memory = PrioritizedReplayMemory(maxlen=3, seed=0)
    prioritized_memory.append((state, 0, next_state, 0.1, False))
    prioritized_memory.append((state, 1, next_state, 0.1, False))
    prioritized_memory.append((state, 0, next_state, -1.0, True))

    per_batch, per_indices, per_weights = prioritized_memory.sample(2, beta=0.4)
    print("prioritized sample size:", len(per_batch))
    print("prioritized indices shape:", per_indices.shape)
    print("prioritized weights shape:", per_weights.shape)

    prioritized_memory.update_priorities(per_indices, [0.5, 1.5])
    assert len(per_batch) == 2
    assert per_weights.shape == (2,)
