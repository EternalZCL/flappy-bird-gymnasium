import torch
from torch import nn

from dqn import DQN
from experience_replay import unpack_batch


def compute_dqn_loss(
    policy_dqn: DQN,
    target_dqn: DQN,
    transitions,
    gamma: float,
    enable_double_dqn: bool = False,
    loss_fn_name: str = "mse",
    importance_sampling_weights: torch.Tensor | None = None,
    return_td_errors: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """计算一批 transition 对应的 DQN loss。

    这一层只关心 DQN 的核心学习公式:

        current_q = Q_policy(state, action)
        target_q  = reward + gamma * max_a Q_target(next_state, a)

    如果 done=True，说明游戏结束了，没有未来价值:

        target_q = reward
    """
    device = next(policy_dqn.parameters()).device
    states, actions, next_states, rewards, dones = unpack_batch(transitions)
    states = states.to(device)
    actions = actions.to(device)
    next_states = next_states.to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)

    # policy_dqn(states).shape == [batch_size, action_dim]
    # gather 会根据 actions 取出“当时实际采取的那个动作”的 Q 值。
    all_current_q = policy_dqn(states)
    current_q = all_current_q.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

    with torch.no_grad():
        if enable_double_dqn:
            best_next_actions = policy_dqn(next_states).argmax(dim=1)
            max_next_q = target_dqn(next_states).gather(
                dim=1,
                index=best_next_actions.unsqueeze(dim=1),
            ).squeeze(dim=1)
        else:
            all_next_q = target_dqn(next_states)
            max_next_q = all_next_q.max(dim=1).values

        not_done = (~dones).float()
        target_q = rewards + not_done * gamma * max_next_q

    td_errors = target_q - current_q

    if loss_fn_name == "smooth_l1":
        loss_fn = nn.SmoothL1Loss(reduction="none")
        per_sample_loss = loss_fn(current_q, target_q)
    elif loss_fn_name == "mse":
        per_sample_loss = td_errors.pow(2)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")

    if importance_sampling_weights is not None:
        weights = importance_sampling_weights.to(device)
        per_sample_loss = per_sample_loss * weights

    loss = per_sample_loss.mean()
    if return_td_errors:
        return loss, td_errors.detach().abs()
    return loss


def train_one_batch(
    policy_dqn: DQN,
    target_dqn: DQN,
    optimizer: torch.optim.Optimizer,
    transitions,
    gamma: float,
    enable_double_dqn: bool = False,
    loss_fn_name: str = "mse",
    gradient_clip: float | None = None,
    importance_sampling_weights: torch.Tensor | None = None,
    return_td_errors: bool = False,
) -> float | tuple[float, list[float]]:
    """用一批 transition 更新一次 policy network。

    这一步才是真正的“学习”:
        1. 算 loss
        2. 反向传播，得到每个参数应该怎么改
        3. optimizer 根据梯度更新 policy_dqn 的参数

    注意:
        target_dqn 只负责提供稳定的 target_q，这里不会更新它。
    """
    loss_result = compute_dqn_loss(
        policy_dqn=policy_dqn,
        target_dqn=target_dqn,
        transitions=transitions,
        gamma=gamma,
        enable_double_dqn=enable_double_dqn,
        loss_fn_name=loss_fn_name,
        importance_sampling_weights=importance_sampling_weights,
        return_td_errors=return_td_errors,
    )
    if return_td_errors:
        loss, td_errors = loss_result
    else:
        loss = loss_result
        td_errors = None

    optimizer.zero_grad()
    loss.backward()
    if gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), gradient_clip)
    optimizer.step()

    if return_td_errors:
        return loss.item(), td_errors.cpu().tolist()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)

    state_dim = 12
    action_dim = 2
    batch_size = 4
    gamma = 0.99

    policy_dqn = DQN(state_dim=state_dim, action_dim=action_dim)
    target_dqn = DQN(state_dim=state_dim, action_dim=action_dim)

    target_dqn.load_state_dict(policy_dqn.state_dict())
    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.001)

    transitions = []
    for i in range(batch_size):
        state = torch.randn(state_dim).tolist()
        next_state = torch.randn(state_dim).tolist()
        action = i % action_dim
        reward = 1.0 if i < batch_size - 1 else -1.0
        done = i == batch_size - 1
        transitions.append((state, action, next_state, reward, done))

    with torch.no_grad():
        if policy_dqn.enable_dueling_dqn:
            before_weight = policy_dqn.feature.weight[0, 0].item()
        else:
            before_weight = policy_dqn.net[0].weight[0, 0].item()

    print("batch size:", batch_size)
    loss_before_update = compute_dqn_loss(
        policy_dqn=policy_dqn,
        target_dqn=target_dqn,
        transitions=transitions,
        gamma=gamma,
    ).item()
    print("loss before update:", loss_before_update)

    train_loss = train_one_batch(
        policy_dqn=policy_dqn,
        target_dqn=target_dqn,
        optimizer=optimizer,
        transitions=transitions,
        gamma=gamma,
    )

    loss_after_update = compute_dqn_loss(
        policy_dqn=policy_dqn,
        target_dqn=target_dqn,
        transitions=transitions,
        gamma=gamma,
    ).item()

    with torch.no_grad():
        if policy_dqn.enable_dueling_dqn:
            after_weight = policy_dqn.feature.weight[0, 0].item()
        else:
            after_weight = policy_dqn.net[0].weight[0, 0].item()

    print("train loss:", train_loss)
    print("loss after update:", loss_after_update)
    print("one policy weight before:", before_weight)
    print("one policy weight after:", after_weight)

    assert loss_before_update >= 0
    assert loss_after_update >= 0
    assert before_weight != after_weight
