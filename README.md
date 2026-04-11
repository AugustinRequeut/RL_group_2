# Reinforcement Learning Project (Group 2)

## Authors

| Name               | Email                                                                       |
| ------------------ | --------------------------------------------------------------------------- |
| Martinelli Mickaël | [mickael.martinelli@student-cs.fr](mailto:mickael.martinelli@student-cs.fr)       |
| Musina Karina      | [karina.musina@student-cs.fr](mailto:karina.musina@student-cs.fr)               |
| Oiknine Nathan     | [nathan.oiknine@student-cs.fr](mailto:nathan.oiknine@student-cs.fr) |
| Requeut Augustin   | [augustin.requeut@student-cs.fr](mailto:augustin.requeut@student-cs.fr)     |

## Compare Custom DQN vs SB3 DQN

Le repo contient des scripts suivants:

- `main.py`: entrée unique avec `--model custom|sb3|reinforce`.
- `compare_dqn_results.py`: agrège les métriques et compare les deux.

### 1) Lancer les expériences


```bash
for SEED in 0 1 2; do
  ./.venv/bin/python main.py --model custom --seed $SEED --timesteps 50000 --eval-runs 50
  ./.venv/bin/python main.py --model sb3 --seed $SEED --timesteps 50000 --eval-runs 50
done
```

### 1bis) Lancer 3 seeds en parallèle 

```bash
# Custom DQN: seeds 0,1,2 en parallèle
seq 0 2 | xargs -I{} -P 3 ./.venv/bin/python main.py \
  --model custom --seed {} --timesteps 50000 --eval-runs 50

# SB3 DQN: seeds 0,1,2 en parallèle
seq 0 2 | xargs -I{} -P 3 ./.venv/bin/python main.py \
  --model sb3 --seed {} --timesteps 50000 --eval-runs 50
```

### 2) Agréger la comparaison

```bash
./.venv/bin/python compare_dqn_results.py \
  --custom-dir results/custom_dqn \
  --sb3-dir results/sb3_dqn \
  --output results/dqn_comparison_summary.json
```

Le fichier `results/dqn_comparison_summary.json` contient:

- moyenne et écart-type (sur les seeds) pour chaque méthode,
- différence appariée `custom - sb3` sur les seeds communs.

### Mode rapide (itération)

Pour tester plus vite, utilisez le preset `--quick`:

```bash
for SEED in 0 1 2; do
  ./.venv/bin/python main.py --model custom --seed $SEED --quick
  ./.venv/bin/python main.py --model sb3 --seed $SEED --quick
done
```

Ce mode utilise, par défaut:

- `timesteps=5000`
- `eval-runs=10`

Pour afficher la reward pendant le train:

- `--log-train-every 50` (par défaut)

Le pipeline d'entraînement n'enregistre pas de vidéo.
Pour les vidéos, utilisez uniquement des checkpoints déjà entraînés (section ci-dessous).

Chaque seed exporte aussi les artefacts d'entraînement:

- `training_curves.png` (graphe reward + loss si disponible)
- `train_episode_rewards.json` (dictionnaire `episode_i -> reward`)
- `train_rewards.npy` (liste brute des rewards)
- `train_losses.json` (dictionnaire `update_i -> loss`)
- `train_losses.npy` (liste brute des losses)

Note: sur SB3, les losses peuvent être vides sur des runs très courts (pas assez de steps avant le début d'apprentissage, ici `learning_starts=batch_size=128`).
Note: les JSON d'entraînement (`train_episode_rewards.json`, `train_losses.json`) sont mis à jour
pendant le training tous les 100 épisodes par défaut.

Checkpoints:

- checkpoint final toujours sauvegardé:
  - Custom: `results/custom_dqn/seed_<SEED>/custom_dqn_qnet.pt`
  - SB3: `results/sb3_dqn/seed_<SEED>/sb3_dqn_model.zip`
- checkpoints intermédiaires: activés par défaut tous les 100 épisodes
  - sortie dans `results/<model>/seed_<SEED>/checkpoints/`
  - noms:
    - Custom: `custom_dqn_qnet_ep_000100.pt`, `..._ep_000200.pt`, etc.
    - SB3: `sb3_dqn_model_ep_000100.zip`, `..._ep_000200.zip`, etc.
- pour changer la fréquence: `--checkpoint-every-episodes <N>`
- pour désactiver: `--checkpoint-every-episodes 0`
- snapshots JSON pendant training:
  - fréquence: `--save-json-every-episodes <N>` (défaut: `100`)
  - désactiver: `--save-json-every-episodes 0`

Exemple:

```bash
./.venv/bin/python main.py --model custom --seed 0 --timesteps 50000 --eval-runs 50 \
  --checkpoint-every-episodes 50

./.venv/bin/python main.py --model sb3 --seed 0 --timesteps 50000 --eval-runs 50 \
  --checkpoint-every-episodes 50
```

### Générer des vidéos après entraînement (sans retrain)

On peut enregistrer plusieurs rollouts depuis un checkpoint déjà entraîné:

```bash
# Custom DQN (exemple: checkpoint seed 0)
./.venv/bin/python record_trained_videos.py \
  --algo custom \
  --checkpoint results/custom_dqn/seed_0/custom_dqn_qnet.pt \
  --n-videos 3 \
  --output-dir results/post_train_videos

# SB3 DQN (exemple: checkpoint seed 0)
./.venv/bin/python record_trained_videos.py \
  --algo sb3 \
  --checkpoint results/sb3_dqn/seed_0/sb3_dqn_model.zip \
  --n-videos 3 \
  --output-dir results/post_train_videos
```

Les vidéos sont sauvegardées dans:

- `results/post_train_videos/custom/`
- `results/post_train_videos/sb3/`

En environnement sans affichage, ajouter `--headless` (plus robuste, mais peut produire une vidéo noire selon la machine).
