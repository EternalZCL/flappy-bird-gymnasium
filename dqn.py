import torch
from torch import nn


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

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # DQN 的核心思想是用神经网络近似 Q 函数:
        #     Q(s, a; theta)
        #
        # 这里的 theta 就是神经网络里的所有可训练参数。
        # 当前模块只负责“给定 state，输出每个 action 的 Q 值”，
        # 真正的 value update 会在后面的 loss.backward() 和 optimizer.step() 里发生。
        self.net = nn.Sequential(
            # 第一层: 把 state_dim 个状态特征映射到 hidden_dim 个隐藏特征。
            # 输入 shape:  [batch_size, state_dim]
            # 输出 shape:  [batch_size, hidden_dim]
            nn.Linear(state_dim, hidden_dim),

            # ReLU 是非线性激活函数。
            # 如果没有非线性，多层 Linear 叠起来仍然只等价于一个 Linear，
            # 网络表达能力会很弱。
            nn.ReLU(),

            # 输出层: 为每个动作输出一个 Q 值。
            # 输入 shape:  [batch_size, hidden_dim]
            # 输出 shape:  [batch_size, action_dim]
            #
            # 对 Flappy Bird 来说，输出可以理解为:
            #     q_values[:, 0] = Q(state, do nothing)
            #     q_values[:, 1] = Q(state, flap)
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward 只做前向计算: state -> q_values。
        #
        # 注意: 这里不会更新参数，也不会学习。
        # 参数更新发生在训练步骤中:
        #     loss.backward()
        #     optimizer.step()
        return self.net(x)


if __name__ == "__main__":
    # 下面是这个模块的最小 shape 检查。
    # 先不用真实环境，只构造一批假的 12 维状态，确认网络输入输出维度正确。
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

    print("states shape:", states.shape)
    print("q_values shape:", q_values.shape)

    # 如果输出维度不符合预期，直接报错，方便我们尽早发现接口问题。
    assert q_values.shape == (batch_size, action_dim)
