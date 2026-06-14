import random

import torch
from torch import nn
import torch.nn.functional as F


def select_action(q_values: torch.Tensor, epsilon: float) -> int:
    """根据 epsilon-greedy 规则选择动作。

    epsilon 的含义:
        random.random() < epsilon 时，随机选动作，用来探索。
        否则选 Q 值最大的动作，用来利用当前网络的判断。
    """
    action_dim = q_values.shape[-1]

    if random.random() < epsilon:
        return random.randrange(action_dim)

    return q_values.argmax().item()


class DQN(nn.Module):
    """用于估计动作价值 Q(s, a) 的神经网络。

    在低维状态版 Flappy Bird 中，环境已经把画面处理成 12 个数值特征，
    所以这里不需要 CNN，只需要一个简单的全连接网络（MLP）。

    网络输入:
        x.shape == [batch_size, state_dim]
        对 FlappyBird-v0(use_lidar=False) 来说，state_dim = 12。

    网络输出:
        q_values.shape == [batch_size, action_dim]
        对 Flappy Bird 来说，action_dim = 2，分别对应:
            action 0: do nothing
            action 1: flap
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        enable_dueling_dqn: bool = False,
    ):
        super().__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        # DQN 的核心思想是用神经网络近似 Q 函数:
        #     Q(s, a; theta)
        #
        # 这里的 theta 就是神经网络里的所有可训练参数。
        # 当前模块只负责“给定 state，输出每个 action 的 Q 值”，
        # 真正的 value update 会在后面的 loss.backward() 和 optimizer.step() 里发生。
        if self.enable_dueling_dqn:
            # 第一层: 把 state_dim 个状态特征映射到 hidden_dim 个隐藏特征。
            self.feature = nn.Linear(state_dim, hidden_dim)

            # Dueling DQN:
            #   value(s) 负责判断“这个状态整体好不好”
            #   advantage(s, a) 负责判断“每个动作相对其它动作好多少”
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
        else:
            # 普通 DQN 保留原来的 self.net 参数名，兼容旧 checkpoint。
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward 只做前向计算: state -> q_values。
        #
        # 注意: 这里不会更新参数，也不会学习。
        # 参数更新发生在训练步骤中:
        #     loss.backward()
        #     optimizer.step()
        if self.enable_dueling_dqn:
            features = F.relu(self.feature(x))
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            return value + advantages - advantages.mean(dim=1, keepdim=True)

        return self.net(x)


if __name__ == "__main__":
    # 下面是这个模块的最小 shape 检查。
    # 先不用真实环境，只构造一批假的 12 维状态，确认网络输入输出维度正确。
    random.seed(0)
    torch.manual_seed(0)

    batch_size = 4
    state_dim = 12
    action_dim = 2

    model = DQN(state_dim=state_dim, action_dim=action_dim)

    # states.shape == [4, 12]
    # 表示一次喂给网络 4 个状态，每个状态有 12 个低维特征。
    states = torch.randn(batch_size, state_dim)

    # q_values.shape 应该是 [4, 2]
    # 表示每个状态都输出 2 个动作的 Q 值。
    q_values = model(states)

    # 对每个 state，选 Q 值最大的那个动作。
    # dim=1 表示在“动作维度”上取最大值:
    #     q_values[i, 0] 是第 i 个 state 下 action 0 的 Q 值
    #     q_values[i, 1] 是第 i 个 state 下 action 1 的 Q 值
    best_actions = q_values.argmax(dim=1)

    print("states shape:", states.shape)
    print("q_values shape:", q_values.shape)
    print("q_values:", q_values)
    print("best actions:", best_actions)

    # 只拿第 1 个 state 的 Q 值，演示 epsilon-greedy 怎么选动作。
    first_state_q_values = q_values[0]

    exploit_action = select_action(first_state_q_values, epsilon=0.0)
    explore_actions = [
        select_action(first_state_q_values, epsilon=1.0)
        for _ in range(8)
    ]

    print("first state q_values:", first_state_q_values)
    print("epsilon=0.0 selected action:", exploit_action)
    print("epsilon=1.0 sampled actions:", explore_actions)

    # 如果输出维度不符合预期，直接报错，方便我们尽早发现接口问题。
    assert q_values.shape == (batch_size, action_dim)
    assert best_actions.shape == (batch_size,)
