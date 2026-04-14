# Reinforcement Learning Project (Group 2)

## Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/AugustinRequeut/RL_group_2.git
```

**2. Create a virtual environment**

```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Start training**

The training configuration can be changed in `src/config.py`.

The choice of training algorithm can be changed via CLI in `main.py`.

To start training, run for example:

```bash
./.venv/bin/python main.py --model custom --quick
```

## Environment

The project is based on the Gymnasium environment `highway-v0`, with a custom configuration defined in `src/config.py`.

## DQN (Custom vs SB3) - Detailed Usage

Comparison between:
- a custom DQN (`src/dqn.py`)
- a Stable-Baselines3 DQN (`stable_baselines3.DQN`)

Shared benchmark setup:
- env: `highway-v0`
- shared config: `src/config.py` (`SHARED_CORE_CONFIG`)
- main script: `main.py` (unified CLI)

## 1) Main command

Everything goes through:

```bash
./.venv/bin/python main.py --model {custom|sb3|reinforce} [options]
```

Useful options:
- `--seed`
- `--timesteps` (custom/sb3)
- `--episodes` (reinforce)
- `--eval-runs`
- `--num-envs`
- `--custom-network {flat_mlp,shared_pool,pairwise_ego}`
- `--pooling {mean,max}`
- `--checkpoint-every-episodes` (default: `100`)
- `--save-json-every-episodes` (default: `100`)
- `--quick` (quick default: `5000` timesteps and `10` eval-runs for custom/sb3)
- `--no-eval`

### 1.1 Custom architectures: `shared_pool` and `pairwise_ego`

Input observation: matrix `(10, 5)` with `ego` at index 0 and `9` non-ego vehicles.

`shared_pool`:
- each non-ego vehicle is passed through a shared MLP `phi: 5 -> 128 -> 128`
- pooling (`mean` or `max`) is applied over non-ego embeddings
- this pooled vector is concatenated with `ego` features (total dim `5 + 128 = 133`)
- final head `133 -> 128 -> 128 -> n_actions`

`pairwise_ego`:
- each non-ego vehicle is also passed through `phi: 5 -> 128 -> 128`
- for each vehicle `i`, a pair `concat(ego, phi_i)` is built (dim `133`)
- this pair is passed through a shared MLP `psi: 133 -> 128 -> 128`
- pair embeddings are pooled (`mean` or `max`)
- final head `133 -> 128 -> 128 -> n_actions`

## 2) Training: useful commands

### 2.1 Quick run (smoke test)

```bash
./.venv/bin/python main.py --model custom --quick --seed 0 --output-dir results/custom_dqn/quick
./.venv/bin/python main.py --model sb3 --quick --seed 0 --output-dir results/sb3_dqn/quick
```

### 2.2 Full run: 50k timesteps, 50 eval-runs (1 seed)

```bash
./.venv/bin/python main.py --model custom --seed 0 --timesteps 50000 --eval-runs 50 --custom-network flat_mlp --output-dir results/custom_dqn/flat_mlp
./.venv/bin/python main.py --model sb3 --seed 0 --timesteps 50000 --eval-runs 50 --output-dir results/sb3_dqn/flat_mlp
```

### 2.3 Run 3 seeds in parallel (flat MLP)

Custom:
```bash
seq 0 2 | xargs -I{} -P 3 ./.venv/bin/python main.py \
  --model custom --seed {} \
  --timesteps 50000 --eval-runs 50 \
  --custom-network flat_mlp \
  --output-dir results/custom_dqn/flat_mlp
```

SB3:
```bash
seq 0 2 | xargs -I{} -P 3 ./.venv/bin/python main.py \
  --model sb3 --seed {} \
  --timesteps 50000 --eval-runs 50 \
  --output-dir results/sb3_dqn/flat_mlp
```

### 2.4 Run custom structural variants (1 seed, in parallel)

```bash
SEED=0
./.venv/bin/python main.py --model custom --seed $SEED --timesteps 50000 --eval-runs 50 --custom-network shared_pool  --pooling mean --output-dir results/custom_dqn/shared_pool_mean &
./.venv/bin/python main.py --model custom --seed $SEED --timesteps 50000 --eval-runs 50 --custom-network shared_pool  --pooling max  --output-dir results/custom_dqn/shared_pool_max &
./.venv/bin/python main.py --model custom --seed $SEED --timesteps 50000 --eval-runs 50 --custom-network pairwise_ego --pooling mean --output-dir results/custom_dqn/pairwise_ego_mean &
./.venv/bin/python main.py --model custom --seed $SEED --timesteps 50000 --eval-runs 50 --custom-network pairwise_ego --pooling max  --output-dir results/custom_dqn/pairwise_ego_max &
wait
```

## 3) Saved artifacts

For each run (`.../seed_X/`):
- `metrics.json`
- `train_episode_rewards.json`
- `train_losses.json`
- `training_curves.png` (loss + rewards + epsilon)
- `eval_rewards.json` (if evaluation is enabled)
- final checkpoint:
  - custom: `custom_dqn_qnet.pt`
  - sb3: `sb3_dqn_model.zip`
- intermediate checkpoints (`checkpoints/`) every `N` episodes if enabled.

## 4) Evaluate checkpoints (intermediate + final)

Script: `evaluate_custom_checkpoints.py`  
Supports `--algo custom` and `--algo sb3`.

Custom example:
```bash
RUN="results/custom_dqn/flat_mlp/seed_0"
MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python evaluate_custom_checkpoints.py \
  --algo custom \
  --run-dir "$RUN" \
  --episodes-per-checkpoint 50 \
  --seed-start 40000 \
  --parallel-workers 8 \
  --output "$RUN/checkpoint_eval_diagnostics.json"
```

SB3 example:
```bash
RUN="results/sb3_dqn/flat_mlp/seed_0"
MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python evaluate_custom_checkpoints.py \
  --algo sb3 \
  --run-dir "$RUN" \
  --episodes-per-checkpoint 50 \
  --seed-start 40000 \
  --parallel-workers 8 \
  --output "$RUN/checkpoint_eval_diagnostics.json"
```

Outputs:
- diagnostics JSON per checkpoint
- plot `checkpoint_eval_evolution_ci95.png` (crash rate, mean speed, mean reward + 95% CI)

### 4.1 Evaluate intermediate checkpoints for 3 seeds

```bash
EP=50
SEED_START=40000

# Custom flat, seeds 0..2
for RUN in \
  results/custom_dqn/flat_mlp/seed_0 \
  results/custom_dqn/flat_mlp/seed_1 \
  results/custom_dqn/flat_mlp/seed_2
do
  MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python evaluate_custom_checkpoints.py \
    --algo custom \
    --run-dir "$RUN" \
    --episodes-per-checkpoint "$EP" \
    --seed-start "$SEED_START" \
    --parallel-workers 8 \
    --output "$RUN/checkpoint_eval_diagnostics.json"
done

# SB3 flat, seeds 0..2
for RUN in \
  results/sb3_dqn/flat_mlp/seed_0 \
  results/sb3_dqn/flat_mlp/seed_1 \
  results/sb3_dqn/flat_mlp/seed_2
do
  MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python evaluate_custom_checkpoints.py \
    --algo sb3 \
    --run-dir "$RUN" \
    --episodes-per-checkpoint "$EP" \
    --seed-start "$SEED_START" \
    --parallel-workers 8 \
    --output "$RUN/checkpoint_eval_diagnostics.json"
done

# Custom pairwise max, seeds 0..2
for RUN in \
  results/custom_dqn/pairwise_ego_max/seed_0 \
  results/custom_dqn/pairwise_ego_max/seed_1 \
  results/custom_dqn/pairwise_ego_max/seed_2
do
  MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python evaluate_custom_checkpoints.py \
    --algo custom \
    --run-dir "$RUN" \
    --episodes-per-checkpoint "$EP" \
    --seed-start "$SEED_START" \
    --parallel-workers 8 \
    --output "$RUN/checkpoint_eval_diagnostics.json"
done
```

## 5) Generate videos from already-trained checkpoints

Script: `record_trained_videos.py`  
Videos are not recorded during training.

Custom:
```bash
./.venv/bin/python record_trained_videos.py \
  --algo custom \
  --checkpoint results/custom_dqn/flat_mlp/seed_0/custom_dqn_qnet.pt \
  --n-videos 3 \
  --seed 30000 \
  --output-dir results/recorded_rollouts
```

SB3:
```bash
./.venv/bin/python record_trained_videos.py \
  --algo sb3 \
  --checkpoint results/sb3_dqn/flat_mlp/seed_0/sb3_dqn_model.zip \
  --n-videos 3 \
  --seed 30000 \
  --output-dir results/recorded_rollouts
```

## 6) Replot curves from JSON

Script: `plot_training_curves_from_json.py`

```bash
MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python plot_training_curves_from_json.py \
  --input-path results \
  --output-name training_curves_from_json.png  \
  --ma-window 100
```

## 7) Final custom vs sb3 comparison (3 seeds)

```bash
./.venv/bin/python compare_dqn_results.py \
  --custom-dir results/custom_dqn/flat_mlp \
  --sb3-dir results/sb3_dqn/flat_mlp \
  --output results/dqn_comparison_flat_mlp.json
```

This script computes:
- mean and standard deviation of mean rewards per seed
- paired difference custom - sb3 over common seeds

## Authors

| Name               | Email                                                                       |
| ------------------ | --------------------------------------------------------------------------- |
| Martinelli Mickael | [mickael.martinelli@student-cs.fr](mailto:mickael.martinelli@student-cs.fr) |
| Musina Karina      | [karina.musina@student-cs.fr](mailto:karina.musina@student-cs.fr)           |
| Oiknine Nathan     | [nathan.oiknine@student-cs.fr](mailto:nathan.oiknine@student-cs.fr)         |
| Requeut Augustin   | [augustin.requeut@student-cs.fr](mailto:augustin.requeut@student-cs.fr)     |
