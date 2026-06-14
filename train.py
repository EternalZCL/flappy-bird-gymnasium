import argparse
import csv
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import yaml

import flappy_bird_gymnasium  # noqa: F401 - registers FlappyBird-v0
from dqn import DQN, select_action
from experience_replay import PrioritizedReplayMemory, ReplayMemory
from record_video import save_gif
from training_step import train_one_batch


RUNS_DIR = "runs"
DEVICE = "cpu"


def load_hyperparameters(name: str):
    with open("hyperparameters.yml", "r", encoding="utf-8") as file:
        all_sets = yaml.safe_load(file)

    if name not in all_sets:
        available = ", ".join(all_sets.keys())
        raise KeyError(f"Unknown hyperparameter set '{name}'. Available: {available}")

    return all_sets[name]


def make_run_paths(run_name: str):
    run_dir = os.path.join(RUNS_DIR, run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    videos_dir = os.path.join(run_dir, "videos")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "videos_dir": videos_dir,
        "log_file": os.path.join(run_dir, "train.log"),
        "latest_checkpoint": os.path.join(checkpoints_dir, "latest.pt"),
        "best_checkpoint": os.path.join(checkpoints_dir, "best.pt"),
    }


def init_log(log_file: str, resume: bool):
    if resume and os.path.exists(log_file):
        return

    with open(log_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "steps",
                "score",
                "episode_reward",
                "mean_reward_100",
                "epsilon",
                "memory_size",
                "loss",
                "best_score",
                "saved_best",
            ]
        )


def append_log(log_file: str, row):
    with open(log_file, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def is_new_best(score: int, episode_reward: float, best_score: int, best_reward: float):
    return score > best_score or (score == best_score and episode_reward > best_reward)


def make_n_step_transition(n_step_buffer, gamma: float):
    """把连续的 1-step transition 合成一个 n-step transition。"""
    reward_sum = 0.0
    next_state = n_step_buffer[-1][2]
    done = n_step_buffer[-1][4]

    for index, transition in enumerate(n_step_buffer):
        _, _, transition_next_state, reward, transition_done = transition
        reward_sum += (gamma**index) * reward
        next_state = transition_next_state
        if transition_done:
            done = True
            break

    state, action = n_step_buffer[0][0], n_step_buffer[0][1]
    return state, action, next_state, reward_sum, done


def get_per_beta(hyperparameters, train_step_count: int):
    beta_start = hyperparameters.get("prioritized_replay_beta_start", 0.4)
    beta_frames = hyperparameters.get("prioritized_replay_beta_frames", 50000)
    if beta_frames <= 0:
        return 1.0
    progress = min(1.0, train_step_count / beta_frames)
    return beta_start + progress * (1.0 - beta_start)


def save_checkpoint(
    path: str,
    policy_dqn: DQN,
    target_dqn: DQN,
    optimizer: torch.optim.Optimizer,
    episode: int,
    epsilon: float,
    best_score: int,
    best_reward: float,
    hyperparameters,
):
    torch.save(
        {
            "policy_dqn": policy_dqn.state_dict(),
            "target_dqn": target_dqn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": epsilon,
            "episode": episode,
            "best_score": best_score,
            "best_reward": best_reward,
            "hyperparameters": hyperparameters,
        },
        path,
    )


def load_checkpoint(path: str, policy_dqn: DQN, target_dqn: DQN, optimizer):
    checkpoint = torch.load(path, map_location=DEVICE)
    policy_dqn.load_state_dict(checkpoint["policy_dqn"])
    target_dqn.load_state_dict(checkpoint["target_dqn"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def record_policy_gif(
    policy_dqn: DQN,
    hyperparameters,
    output_path: str,
    seed: int,
    max_steps: int,
    fps: int,
):
    env = gym.make(
        hyperparameters["env_id"],
        render_mode="rgb_array",
        **hyperparameters.get("env_make_params", {}),
    )

    was_training = policy_dqn.training
    policy_dqn.eval()

    frames = []
    state, _ = env.reset(seed=seed)
    frames.append(env.render())

    total_reward = 0.0
    info = {"score": 0}

    try:
        for step in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                q_values = policy_dqn(state_tensor.unsqueeze(dim=0)).squeeze(dim=0)

            action = select_action(q_values=q_values, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frames.append(env.render())

            if terminated or truncated:
                break
    finally:
        env.close()
        if was_training:
            policy_dqn.train()

    save_gif(frames, output_path, fps=fps)
    return {
        "output_path": output_path,
        "frames": len(frames),
        "steps": step + 1,
        "total_reward": total_reward,
        "score": info["score"],
    }


def run_human_eval(policy_dqn: DQN, hyperparameters, seed: int, max_steps: int):
    env = gym.make(
        hyperparameters["env_id"],
        render_mode="human",
        **hyperparameters.get("env_make_params", {}),
    )

    state, _ = env.reset(seed=seed)
    was_training = policy_dqn.training
    policy_dqn.eval()

    try:
        for _ in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                q_values = policy_dqn(state_tensor.unsqueeze(dim=0)).squeeze(dim=0)

            action = select_action(q_values=q_values, epsilon=0.0)
            state, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break
    finally:
        env.close()
        if was_training:
            policy_dqn.train()


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    hyperparameters = load_hyperparameters(args.hyperparameters)
    paths = make_run_paths(args.hyperparameters)
    init_log(paths["log_file"], resume=args.resume)

    env = gym.make(
        hyperparameters["env_id"],
        **hyperparameters.get("env_make_params", {}),
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_dqn = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hyperparameters["fc1_nodes"],
        enable_dueling_dqn=hyperparameters.get("enable_dueling_dqn", False),
    ).to(DEVICE)
    target_dqn = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hyperparameters["fc1_nodes"],
        enable_dueling_dqn=hyperparameters.get("enable_dueling_dqn", False),
    ).to(DEVICE)
    target_dqn.load_state_dict(policy_dqn.state_dict())

    optimizer = torch.optim.Adam(
        policy_dqn.parameters(),
        lr=hyperparameters["learning_rate_a"],
    )
    use_prioritized_replay = hyperparameters.get("enable_prioritized_replay", False)
    if use_prioritized_replay:
        memory = PrioritizedReplayMemory(
            maxlen=hyperparameters["replay_memory_size"],
            alpha=hyperparameters.get("prioritized_replay_alpha", 0.6),
            priority_epsilon=hyperparameters.get("prioritized_replay_epsilon", 1e-5),
            seed=args.seed,
        )
    else:
        memory = ReplayMemory(hyperparameters["replay_memory_size"], seed=args.seed)

    n_step_return = hyperparameters.get("n_step_return", 1)
    if n_step_return < 1:
        raise ValueError("n_step_return must be >= 1")
    training_gamma = hyperparameters["discount_factor_g"] ** n_step_return

    epsilon = hyperparameters["epsilon_init"]
    best_score = -1
    best_reward = -float("inf")
    start_episode = 1

    if args.resume:
        if not os.path.exists(paths["latest_checkpoint"]):
            raise FileNotFoundError(
                f"Cannot resume because checkpoint does not exist: "
                f"{paths['latest_checkpoint']}"
            )
        checkpoint = load_checkpoint(
            paths["latest_checkpoint"],
            policy_dqn=policy_dqn,
            target_dqn=target_dqn,
            optimizer=optimizer,
        )
        epsilon = checkpoint["epsilon"]
        best_score = checkpoint["best_score"]
        best_reward = checkpoint["best_reward"]
        start_episode = checkpoint["episode"] + 1

    rewards_100 = deque(maxlen=100)
    train_step_count = 0

    if args.record_initial and not args.resume:
        output_path = os.path.join(paths["videos_dir"], "episode_000000.gif")
        result = record_policy_gif(
            policy_dqn=policy_dqn,
            hyperparameters=hyperparameters,
            output_path=output_path,
            seed=args.seed,
            max_steps=args.eval_max_steps,
            fps=args.gif_fps,
        )
        print(f"initial gif: {result}")

    for episode in range(start_episode, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        last_loss = None
        info = {"score": 0}
        n_step_buffer = deque(maxlen=n_step_return)

        step = 0
        done = False
        while not done and episode_reward < hyperparameters["stop_on_reward"]:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                q_values = policy_dqn(state_tensor.unsqueeze(dim=0)).squeeze(dim=0)

            action = select_action(q_values=q_values, epsilon=epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            transition = (state, action, next_state, reward, done)
            if n_step_return == 1:
                memory.append(transition)
            else:
                n_step_buffer.append(transition)
                if len(n_step_buffer) >= n_step_return:
                    memory.append(
                        make_n_step_transition(
                            n_step_buffer=n_step_buffer,
                            gamma=hyperparameters["discount_factor_g"],
                        )
                    )
                    n_step_buffer.popleft()
            episode_reward += reward

            if len(memory) >= hyperparameters["mini_batch_size"]:
                if use_prioritized_replay:
                    beta = get_per_beta(hyperparameters, train_step_count)
                    transitions, indices, weights = memory.sample(
                        hyperparameters["mini_batch_size"],
                        beta=beta,
                    )
                    last_loss, td_errors = train_one_batch(
                        policy_dqn=policy_dqn,
                        target_dqn=target_dqn,
                        optimizer=optimizer,
                        transitions=transitions,
                        gamma=training_gamma,
                        enable_double_dqn=hyperparameters.get(
                            "enable_double_dqn", False
                        ),
                        loss_fn_name=hyperparameters.get("loss_fn", "mse"),
                        gradient_clip=hyperparameters.get("gradient_clip"),
                        importance_sampling_weights=weights,
                        return_td_errors=True,
                    )
                    memory.update_priorities(indices, td_errors)
                else:
                    transitions = memory.sample(hyperparameters["mini_batch_size"])
                    last_loss = train_one_batch(
                        policy_dqn=policy_dqn,
                        target_dqn=target_dqn,
                        optimizer=optimizer,
                        transitions=transitions,
                        gamma=training_gamma,
                        enable_double_dqn=hyperparameters.get(
                            "enable_double_dqn", False
                        ),
                        loss_fn_name=hyperparameters.get("loss_fn", "mse"),
                        gradient_clip=hyperparameters.get("gradient_clip"),
                    )
                train_step_count += 1

                if train_step_count % hyperparameters["network_sync_rate"] == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            state = next_state

            if done and n_step_return > 1:
                while n_step_buffer:
                    memory.append(
                        make_n_step_transition(
                            n_step_buffer=n_step_buffer,
                            gamma=hyperparameters["discount_factor_g"],
                        )
                    )
                    n_step_buffer.popleft()

        epsilon = max(
            epsilon * hyperparameters["epsilon_decay"],
            hyperparameters["epsilon_min"],
        )

        rewards_100.append(episode_reward)
        mean_reward_100 = sum(rewards_100) / len(rewards_100)
        score = info["score"]

        saved_best = is_new_best(score, episode_reward, best_score, best_reward)
        if saved_best:
            best_score = score
            best_reward = episode_reward
            save_checkpoint(
                paths["best_checkpoint"],
                policy_dqn=policy_dqn,
                target_dqn=target_dqn,
                optimizer=optimizer,
                episode=episode,
                epsilon=epsilon,
                best_score=best_score,
                best_reward=best_reward,
                hyperparameters=hyperparameters,
            )

        save_checkpoint(
            paths["latest_checkpoint"],
            policy_dqn=policy_dqn,
            target_dqn=target_dqn,
            optimizer=optimizer,
            episode=episode,
            epsilon=epsilon,
            best_score=best_score,
            best_reward=best_reward,
            hyperparameters=hyperparameters,
        )

        append_log(
            paths["log_file"],
            [
                episode,
                step,
                score,
                f"{episode_reward:.6f}",
                f"{mean_reward_100:.6f}",
                f"{epsilon:.6f}",
                len(memory),
                "" if last_loss is None else f"{last_loss:.8f}",
                best_score,
                int(saved_best),
            ],
        )

        loss_text = "None" if last_loss is None else f"{last_loss:.6f}"
        print(
            f"episode={episode} steps={step} score={score} "
            f"reward={episode_reward:.2f} mean100={mean_reward_100:.2f} "
            f"epsilon={epsilon:.4f} memory={len(memory)} loss={loss_text} "
            f"best_score={best_score} saved_best={saved_best}"
        )

        if args.gif_every > 0 and episode % args.gif_every == 0:
            output_path = os.path.join(
                paths["videos_dir"],
                f"episode_{episode:06d}.gif",
            )
            result = record_policy_gif(
                policy_dqn=policy_dqn,
                hyperparameters=hyperparameters,
                output_path=output_path,
                seed=args.seed + episode,
                max_steps=args.eval_max_steps,
                fps=args.gif_fps,
            )
            print(f"eval gif: {result}")

            if args.render_eval:
                run_human_eval(
                    policy_dqn=policy_dqn,
                    hyperparameters=hyperparameters,
                    seed=args.seed + episode,
                    max_steps=args.eval_max_steps,
                )

    env.close()
    print(f"training complete. log: {paths['log_file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Flappy Bird DQN agent.")
    parser.add_argument("hyperparameters", help="YAML config key, e.g. flappybird1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--gif-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--record-initial", action="store_true")
    parser.add_argument("--eval-max-steps", type=int, default=1000)
    parser.add_argument("--gif-fps", type=int, default=30)
    train(parser.parse_args())
