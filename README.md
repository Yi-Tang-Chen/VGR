# Value-Gated Remasking (VGR) Phase A

Phase A only trains a value critic with outcome reward and keeps the actor frozen. At inference, VGR uses the critic to trigger local remasking and resampling, enabling self-correction without RL finetuning.

## Goals
- Freeze actor parameters (no RL updates).
- Train a critic $V_\phi(x_t, t, c)$ from outcome-only rewards.
- Use value-gated remasking to improve GSM8K/MATH accuracy and reduce error types.

## Data and Reward
- Training data: GSM8K train (7,473) + MATH train (7,500), total ~15K prompts.
- Evaluation data: GSM8K test (1,319) + MATH test (5,000).
- Reward: $R(x_0, c) = \mathbb{1}(\text{final answer correct})$.

## Phase A Workflow

### Stage A: Offline Feature Collection (one-time rollout)
Run the frozen actor once to collect a compact dataset for critic training.

What gets saved:
- `features`: pooled last hidden state of the generated region (dimension 4096).
- `timesteps`: diffusion step ratio $t \in [0,1]$.
- `rewards`: outcome reward $R \in \{0,1\}$.
- Optional extra features: mask/fill ratios and block position.

Command:
```bash
bash scripts/collect_critic_features.sh
```

Output:
- `datasets_cache/critic_train.pt` containing `features`, `timesteps`, `rewards`, `task_ids`, and `meta`.

### Stage B: Offline Critic Training (no actor calls)
Train a small MLP value head on the cached dataset. This stage is fast and does not run diffusion.

Command:
```bash
bash scripts/train_critic.sh
```

Output:
- Critic head checkpoints under `runs/critic/...` (e.g., `critic_head_epoch1.pt`).

## VGR Inference
VGR uses the critic to gate remasking during diffusion sampling.

Gate rule (v1):
- Trigger remask when value is non-improving: $V_{t} \le V_{t-1}$.

Remask action:
- Select top-$k$ least confident tokens from newly updated positions.
- Remask them and resample to allow correction.

Command:
```bash
bash scripts/eval_math_vgr.sh
```

## Default Hyperparameters (v1)

Sampling:
- steps: 256
- gen_length: 256
- block_length: 8
- temperature: 1.0

VGR:
- gate_start_step: 32
- retry_m: 8
- max_backtracks_total: 32

Critic:
- pooled hidden size: 4096
- steps sampled per trajectory: $K=2$ (Stage A default)
- loss: BCE

## Files and Entry Points
- `training/collect_critic_features.py`: offline feature collector.
- `training/train_critic.py`: offline critic trainer.
- `networks/value_critic.py`: critic head model.
- `networks/vgr_sampler.py`: VGR sampling logic.
- `scripts/collect_critic_features.sh`: Stage A runner.
- `scripts/train_critic.sh`: Stage B runner.
- `scripts/eval_math_vgr.sh`: evaluation with VGR.
