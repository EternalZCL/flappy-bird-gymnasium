import argparse
import os
import random

import gymnasium as gym
from PIL import Image
import torch

import flappy_bird_gymnasium  # noqa: F401 - registers FlappyBird-v0
from dqn import DQN, select_action


def save_gif(frames, output_path: str, fps: int):
    """把 rgb_array frame 列表保存成 GIF。"""
    if not frames:
        raise ValueError("No frames to save.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = int(1000 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def record_episode(
    output_path: str,
    model_path: str | None,
    epsilon: float,
    max_steps: int,
    fps: int,
    seed: int,
):
    """运行一局 Flappy Bird，并保存人类可看的 GIF。

    model_path 为空时，会用未训练的 DQN + epsilon-greedy。
    现在项目还没进入完整训练阶段，所以默认 epsilon=1.0，先录随机策略。
    后面训练出模型后，可以传:

        --model runs/flappybird1.pt --epsilon 0.0
    """
    random.seed(seed)
    torch.manual_seed(seed)

    checkpoint = None
    hyperparameters = None
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "hyperparameters" in checkpoint:
            hyperparameters = checkpoint["hyperparameters"]

    env_id = "FlappyBird-v0"
    env_make_params = {"use_lidar": False}
    hidden_dim = 128
    enable_dueling_dqn = False

    if hyperparameters is not None:
        env_id = hyperparameters["env_id"]
        env_make_params = hyperparameters.get("env_make_params", env_make_params)
        hidden_dim = hyperparameters.get("fc1_nodes", hidden_dim)
        enable_dueling_dqn = hyperparameters.get("enable_dueling_dqn", False)

    env = gym.make(env_id, render_mode="rgb_array", **env_make_params)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_dqn = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        enable_dueling_dqn=enable_dueling_dqn,
    )

    if model_path is not None:
        if isinstance(checkpoint, dict) and "policy_dqn" in checkpoint:
            checkpoint = checkpoint["policy_dqn"]
        policy_dqn.load_state_dict(checkpoint)
        policy_dqn.eval()

    frames = []
    state, _ = env.reset(seed=seed)
    frames.append(env.render())

    total_reward = 0.0
    info = {"score": 0}

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            q_values = policy_dqn(state_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        action = select_action(q_values=q_values, epsilon=epsilon)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(env.render())

        if terminated or truncated:
            break

    env.close()
    save_gif(frames=frames, output_path=output_path, fps=fps)

    return {
        "output_path": output_path,
        "frames": len(frames),
        "steps": step + 1,
        "total_reward": total_reward,
        "score": info["score"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record one Flappy Bird episode.")
    parser.add_argument("--output", default="runs/videos/random_episode.gif")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = record_episode(
        output_path=args.output,
        model_path=args.model,
        epsilon=args.epsilon,
        max_steps=args.max_steps,
        fps=args.fps,
        seed=args.seed,
    )

    print("record result:", result)
