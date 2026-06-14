# Flappy Bird DQN Training Guide

这是一个基于 `flappy-bird-gymnasium` 的强化学习训练项目。项目主线是用低维状态特征训练 Flappy Bird agent，并逐步从普通 DQN 升级到更稳定的 DQN 变体。

当前最推荐的配置是 `flappybird3`：

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- 3-step return
- CPU 训练
- 训练日志、checkpoint、GIF 展示视频

## 1. 环境准备

推荐使用 Python 3.10。

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

如果已经有 `.venv`，可以直接安装依赖：

```bash
.venv/bin/pip install -r requirements.txt
```

项目默认使用 CPU 版 PyTorch，低维状态 DQN 足够跑通。

## 2. 快速检查

先运行几个小测试，确认核心模块正常：

```bash
.venv/bin/python dqn.py
.venv/bin/python experience_replay.py
.venv/bin/python training_step.py
.venv/bin/python train_smoke.py
```

训练入口 smoke test：

```bash
.venv/bin/python train.py flappybird3 --episodes 3 --gif-every 2
```

这会生成：

```text
runs/flappybird3/train.log
runs/flappybird3/checkpoints/latest.pt
runs/flappybird3/videos/episode_000002.gif
```

## 3. 正式训练

推荐从 `flappybird3` 开始训练：

```bash
.venv/bin/python train.py flappybird3 --episodes 10000 --gif-every 1000 --record-initial
```

如果想清空之前的同名实验：

```bash
rm -rf runs/flappybird3
.venv/bin/python train.py flappybird3 --episodes 10000 --gif-every 1000 --record-initial
```

如果要从已有 checkpoint 继续：

```bash
.venv/bin/python train.py flappybird3 --episodes 10000 --gif-every 1000 --resume
```

训练输出默认在：

```text
runs/flappybird3/train.log
runs/flappybird3/checkpoints/latest.pt
runs/flappybird3/checkpoints/best.pt
runs/flappybird3/videos/
```

注意：`runs/` 已经被 `.gitignore` 忽略，checkpoint、日志和 GIF 默认只保存在本地，不会被提交到 Git。

## 4. 训练日志

`train.log` 每一行是一局 episode。

字段含义：

- `episode`: 第几局训练
- `steps`: 这一局活了多少环境步
- `score`: 通过了多少根管道
- `episode_reward`: 这一局累计 reward
- `mean_reward_100`: 最近 100 局平均 reward
- `epsilon`: 当前 epsilon-greedy 探索率
- `memory_size`: Replay Memory 中经验数量
- `loss`: 这一局最后一次 mini-batch 更新的 TD loss
- `best_score`: 到目前为止训练中出现过的最高分
- `saved_best`: 本局是否保存了新的 `best.pt`

重点看：

```text
score
episode_reward
mean_reward_100
best_score
saved_best
```

## 5. 评估模型

评估当前最新模型：

```bash
.venv/bin/python evaluate.py --model runs/flappybird3/checkpoints/latest.pt --episodes 50
```

评估训练中保存的 best 模型：

```bash
.venv/bin/python evaluate.py --model runs/flappybird3/checkpoints/best.pt --episodes 50
```

经验上，`latest.pt` 有时会比 `best.pt` 更稳定，因为 `best.pt` 当前按训练中的单局最高分保存，而不是按多局评估平均分保存。

## 6. 录制展示 GIF

录制当前最新策略：

```bash
.venv/bin/python record_video.py \
  --model runs/flappybird3/checkpoints/latest.pt \
  --output runs/flappybird3/videos/latest_showcase.gif \
  --epsilon 0.0 \
  --max-steps 2000 \
  --fps 30
```

如果想录更长：

```bash
.venv/bin/python record_video.py \
  --model runs/flappybird3/checkpoints/latest.pt \
  --output runs/flappybird3/videos/latest_showcase_5000.gif \
  --epsilon 0.0 \
  --max-steps 5000 \
  --fps 30
```

## 7. 配置版本

所有训练配置在 `hyperparameters.yml`。

`flappybird1`:

- 普通 DQN
- 均匀 Replay Memory
- 教学基线

`flappybird2`:

- Double DQN
- Dueling DQN
- Smooth L1 loss
- Gradient clipping

`flappybird3`:

- `flappybird2` 的全部能力
- Prioritized Experience Replay
- 3-step return

推荐使用：

```bash
.venv/bin/python train.py flappybird3 --episodes 10000 --gif-every 1000 --record-initial
```

## 8. 主要代码结构

```text
dqn.py                 DQN / Dueling DQN 网络和 epsilon-greedy 动作选择
experience_replay.py   ReplayMemory 和 PrioritizedReplayMemory
training_step.py       DQN loss、Double DQN target、一次 mini-batch 更新
train.py               完整训练入口、日志、checkpoint、周期 GIF
evaluate.py            多局纯策略评估
record_video.py        从 checkpoint 录制 GIF
collect_experience.py  单局采样示例
train_smoke.py         小规模训练 smoke test
hyperparameters.yml    训练配置
```

## 9. 当前效果参考

一次 `flappybird3` 训练到 10000 episode 后，`latest.pt` 曾在展示录制中达到：

```text
max_steps = 2000
actual_steps = 2000
score = 52
```

这表示 agent 在 2000 step 展示窗口内没有提前死亡，效果已经比较稳定。

## 10. 后续优化方向

当前项目已经具备完整训练闭环。后续可以继续改进：

- 把 `best.pt` 保存标准改成周期评估平均分，而不是训练单局最高分
- 保存 replay memory，使 `--resume` 能完全恢复训练状态
- 加入 TensorBoard 或 matplotlib 曲线
- 尝试 NoisyNet 或 distributional DQN，继续接近 Rainbow DQN
