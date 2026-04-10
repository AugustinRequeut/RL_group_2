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

### 1) Lancer les expériences (mêmes seeds)


```bash
for SEED in 0 1 2; do
  ./.venv/bin/python main.py --model custom --seed $SEED --timesteps 100000 --eval-runs 50
  ./.venv/bin/python main.py --model sb3 --seed $SEED --timesteps 100000 --eval-runs 50
done
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
  ./.venv/bin/python main.py --model custom --seed $SEED --quick --record-video
  ./.venv/bin/python main.py --model sb3 --seed $SEED --quick --record-video
done
```

Ce mode utilise, par défaut:

- `timesteps=20000`
- `eval-runs=10`

Pour afficher la reward pendant le train:

- `--log-train-every 20` (par défaut)

Chaque run sauvegarde aussi une vidéo dans:

- `results/custom_dqn/seed_<SEED>/video/`
- `results/sb3_dqn/seed_<SEED>/video/`

Chaque seed exporte aussi les artefacts d'entraînement:

- `train_reward_per_episode.png` (graphe reward/épisode)
- `train_episode_rewards.json` (dictionnaire `episode_i -> reward`)

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
