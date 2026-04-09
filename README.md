# Reinforcement Learning Project (Group 2)

## Authors

| Name               | Email                                                                       |
| ------------------ | --------------------------------------------------------------------------- |
| Martinelli Mickaël | [mickael.martinelli@student-cs.fr](mailto:mickael.martinelli@student-cs.fr)       |
| Musina Karina      | [karina.musina@student-cs.fr](mailto:karina.musina@student-cs.fr)               |
| Oiknine Nathan     | [nathan.oiknine@student-cs.fr](mailto:nathan.oiknine@student-cs.fr) |
| Requeut Augustin   | [augustin.requeut@student-cs.fr](mailto:augustin.requeut@student-cs.fr)     |

## Compare Custom DQN vs SB3 DQN

Le repo contient maintenant deux scripts comparables:

- `custom_dqn_baseline.py`: entraîne le DQN maison.
- `sb3_dqn_baseline.py`: entraîne le DQN de Stable-Baselines3.
- `compare_dqn_results.py`: agrège les métriques et compare les deux.

### 1) Lancer les expériences (mêmes seeds)

```bash
for SEED in 0 1 2 3 4; do
  ./.venv/bin/python custom_dqn_baseline.py --seed $SEED --timesteps 100000 --eval-runs 50
  ./.venv/bin/python sb3_dqn_baseline.py --seed $SEED --timesteps 100000 --eval-runs 50
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
  ./.venv/bin/python custom_dqn_baseline.py --seed $SEED --quick --record-video
  ./.venv/bin/python sb3_dqn_baseline.py --seed $SEED --quick --record-video
done
```

Ce mode utilise, par défaut:

- `timesteps=20000`
- `eval-runs=10`

Chaque run sauvegarde aussi une vidéo dans:

- `results/custom_dqn/seed_<SEED>/video/`
- `results/sb3_dqn/seed_<SEED>/video/`
