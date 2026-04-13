# RL Group 2 - DQN (Custom vs SB3)

Comparaison entre:
- un DQN custom (`src/dqn.py`)
- un DQN Stable-Baselines3 (`stable_baselines3.DQN`)

Benchmark commun:
- env: `highway-v0`
- config commune: `src/config.py` (`SHARED_CORE_CONFIG`)
- script principal: `main.py` (CLI unifiee)

## 1) Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Commande principale

Tout passe par:

```bash
./.venv/bin/python main.py --model {custom|sb3|reinforce} [options]
```

Options utiles:
- `--seed`
- `--timesteps` (custom/sb3)
- `--episodes` (reinforce)
- `--eval-runs`
- `--num-envs`
- `--custom-network {flat_mlp,shared_pool,pairwise_ego}`
- `--pooling {mean,max}`
- `--checkpoint-every-episodes` (defaut: `100`)
- `--save-json-every-episodes` (defaut: `100`)
- `--quick` (defaut rapide: `5000` timesteps et `10` eval-runs pour custom/sb3)
- `--no-eval`

## 3) Entrainement: commandes utiles

### 3.1 Run rapide (smoke)

```bash
./.venv/bin/python main.py --model custom --quick --seed 0 --output-dir results/custom_dqn/quick
./.venv/bin/python main.py --model sb3 --quick --seed 0 --output-dir results/sb3_dqn/quick
```

### 3.2 Run complet 50k timesteps, 50 eval-runs (1 seed)

```bash
./.venv/bin/python main.py --model custom --seed 0 --timesteps 50000 --eval-runs 50 --custom-network flat_mlp --output-dir results/custom_dqn/flat_mlp
./.venv/bin/python main.py --model sb3 --seed 0 --timesteps 50000 --eval-runs 50 --output-dir results/sb3_dqn/flat_mlp
```

### 3.3 Lancer 3 seeds en parallele (flat MLP)

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

### 3.4 Lancer les variantes custom structurelles (1 seed, en parallele)

```bash
SEED=0
./.venv/bin/python main.py --model custom --seed $SEED --timesteps 50000 --eval-runs 50 --custom-network shared_pool  --pooling mean --output-dir results/custom_dqn/shared_pool_mean &
./.venv/bin/python main.py --model custom --seed $SEED --timesteps 50000 --eval-runs 50 --custom-network shared_pool  --pooling max  --output-dir results/custom_dqn/shared_pool_max &
./.venv/bin/python main.py --model custom --seed $SEED --timesteps 50000 --eval-runs 50 --custom-network pairwise_ego --pooling mean --output-dir results/custom_dqn/pairwise_ego_mean &
./.venv/bin/python main.py --model custom --seed $SEED --timesteps 50000 --eval-runs 50 --custom-network pairwise_ego --pooling max  --output-dir results/custom_dqn/pairwise_ego_max &
wait
```

## 4) Artifacts sauvegardes

Par run (`.../seed_X/`):
- `metrics.json`
- `train_episode_rewards.json`
- `train_losses.json`
- `training_curves.png` (loss + rewards + epsilon)
- `eval_rewards.json` (si eval activee)
- checkpoint final:
  - custom: `custom_dqn_qnet.pt`
  - sb3: `sb3_dqn_model.zip`
- checkpoints intermediaires (`checkpoints/`) tous les `N` episodes si active.

## 5) Evaluer les checkpoints (intermediaires + final)

Script: `evaluate_custom_checkpoints.py`  
Supporte `--algo custom` et `--algo sb3`.

Exemple custom:
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

Exemple sb3:
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

Sorties:
- JSON de diagnostics par checkpoint
- plot `checkpoint_eval_evolution_ci95.png` (crash rate, mean speed, mean reward + CI95)

### 5.1 Eval checkpoints intermediaires pour 3 seeds 
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

## 6) Generer des videos a partir de checkpoints deja entraines

Script: `record_trained_videos.py`  
Les videos ne sont pas enregistrees pendant le train.

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

Note:
- eviter `--headless` si vous voulez une video non-noire sur machine locale avec affichage.

## 7) Replot des courbes a partir des JSON

Script: `plot_training_curves_from_json.py`

```bash
MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python plot_training_curves_from_json.py \
  --input-path results \
  --output-name training_curves_from_json.png  \
  --ma-window 100
```

## 8) Comparaison finale custom vs sb3 (3 seeds)

```bash
./.venv/bin/python compare_dqn_results.py \
  --custom-dir results/custom_dqn/flat_mlp \
  --sb3-dir results/sb3_dqn/flat_mlp \
  --output results/dqn_comparison_flat_mlp.json
```

Le script calcule:
- moyenne et ecart-type des rewards moyens par seed
- difference pairee custom - sb3 sur seeds communes
