import random

import gymnasium as gym
import torch

import flappy_bird_gymnasium  # noqa: F401 - registers FlappyBird-v0
from dqn import DQN, select_action
from experience_replay import ReplayMemory


def collect_one_episode(
    env,
    policy_dqn: DQN,
    memory: ReplayMemory,
    epsilon: float,
    max_steps: int = 1000,
):
    """用当前 policy 和环境交互一局，并把 transition 存进 ReplayMemory。

    这里还不训练，只采样数据。你可以把它理解成:

        state
          -> policy/epsilon-greedy 选 action
          -> env.step(action)
          -> 得到 next_state, reward, done
          -> 存成一条 transition
    """
    state, _ = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            q_values = policy_dqn(state_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        action = select_action(q_values=q_values, epsilon=epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.append((state, action, next_state, reward, done))

        total_reward += reward
        state = next_state

        if done:
            break

    return {
        "steps": step + 1,
        "total_reward": total_reward,
        "score": info["score"],
        "memory_size": len(memory),
    }


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    env = gym.make("FlappyBird-v0", use_lidar=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_dqn = DQN(state_dim=state_dim, action_dim=action_dim)
    memory = ReplayMemory(maxlen=1000, seed=0)

    result = collect_one_episode(
        env=env,
        policy_dqn=policy_dqn,
        memory=memory,
        epsilon=1.0,
    )

    first_transition = memory.memory[0]
    state, action, next_state, reward, done = first_transition

    print("episode result:", result)
    print("first state length:", len(state))
    print("first action:", action)
    print("first next_state length:", len(next_state))
    print("first reward:", reward)
    print("first done:", done)

    assert state_dim == 12
    assert len(memory) == result["steps"]
    assert len(first_transition) == 5

    env.close()
