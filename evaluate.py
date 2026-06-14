import argparse
from collections import Counter

import gymnasium as gym
import torch

import flappy_bird_gymnasium  # noqa: F401 - registers FlappyBird-v0
from dqn import DQN, select_action


def load_policy(model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu")
    hyperparameters = checkpoint.get("hyperparameters", {})

    env = gym.make(
        hyperparameters.get("env_id", "FlappyBird-v0"),
        **hyperparameters.get("env_make_params", {"use_lidar": False}),
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    policy_dqn = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hyperparameters.get("fc1_nodes", 128),
        enable_dueling_dqn=hyperparameters.get("enable_dueling_dqn", False),
    )
    policy_dqn.load_state_dict(checkpoint["policy_dqn"])
    policy_dqn.eval()
    return policy_dqn, hyperparameters


def evaluate(model_path: str, episodes: int, seed: int, max_steps: int):
    policy_dqn, hyperparameters = load_policy(model_path)
    env = gym.make(
        hyperparameters.get("env_id", "FlappyBird-v0"),
        **hyperparameters.get("env_make_params", {"use_lidar": False}),
    )

    scores = []
    rewards = []
    steps_list = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        info = {"score": 0}

        for step in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = policy_dqn(state_tensor.unsqueeze(dim=0)).squeeze(dim=0)

            action = select_action(q_values=q_values, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        scores.append(info["score"])
        rewards.append(total_reward)
        steps_list.append(step + 1)

    env.close()

    score_counts = Counter(scores)
    return {
        "episodes": episodes,
        "mean_score": sum(scores) / episodes,
        "max_score": max(scores),
        "mean_reward": sum(rewards) / episodes,
        "max_reward": max(rewards),
        "mean_steps": sum(steps_list) / episodes,
        "score_counts": dict(sorted(score_counts.items())),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    result = evaluate(
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print("evaluation:", result)
