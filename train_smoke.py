import random

import gymnasium as gym
import torch

import flappy_bird_gymnasium  # noqa: F401 - registers FlappyBird-v0
from collect_experience import collect_one_episode
from dqn import DQN
from experience_replay import ReplayMemory
from training_step import compute_dqn_loss, train_one_batch


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    env = gym.make("FlappyBird-v0", use_lidar=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_dqn = DQN(state_dim=state_dim, action_dim=action_dim)
    target_dqn = DQN(state_dim=state_dim, action_dim=action_dim)
    target_dqn.load_state_dict(policy_dqn.state_dict())

    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.001)
    memory = ReplayMemory(maxlen=1000, seed=0)

    batch_size = 32
    gamma = 0.99
    epsilon = 1.0

    episode_count = 0
    while len(memory) < batch_size:
        result = collect_one_episode(
            env=env,
            policy_dqn=policy_dqn,
            memory=memory,
            epsilon=epsilon,
        )
        episode_count += 1
        print(f"episode {episode_count}: {result}")

    transitions = memory.sample(batch_size)

    loss_before = compute_dqn_loss(
        policy_dqn=policy_dqn,
        target_dqn=target_dqn,
        transitions=transitions,
        gamma=gamma,
    ).item()

    train_loss = train_one_batch(
        policy_dqn=policy_dqn,
        target_dqn=target_dqn,
        optimizer=optimizer,
        transitions=transitions,
        gamma=gamma,
    )

    loss_after = compute_dqn_loss(
        policy_dqn=policy_dqn,
        target_dqn=target_dqn,
        transitions=transitions,
        gamma=gamma,
    ).item()

    print("sampled batch size:", batch_size)
    print("loss before update:", loss_before)
    print("train loss:", train_loss)
    print("loss after update:", loss_after)

    assert len(memory) >= batch_size
    assert loss_before >= 0
    assert loss_after >= 0

    env.close()
