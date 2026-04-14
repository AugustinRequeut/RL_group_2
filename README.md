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

**4. Start a training**

The training configuration can be changed in `src/config.py`.

The choice of the training algorithm can be changed in `src/main.py`, as well as the number of episodes for the training ,and the number of runs for the evaluation.

To start the training, execute the following command

```bash
python -m main
```

## Environment

The project is based on the gymnasium environment `highway-v0`, with a custom configuration given in the file `src/config.py`.

## Authors

| Name               | Email                                                                       |
| ------------------ | --------------------------------------------------------------------------- |
| Martinelli Mickaël | [mickael.martinelli@student-cs.fr](mailto:mickael.martinelli@student-cs.fr)       |
| Musina Karina      | [karina.musina@student-cs.fr](mailto:karina.musina@student-cs.fr)               |
| Oiknine Nathan     | [nathan.oiknine@student-cs.fr](mailto:nathan.oiknine@student-cs.fr) |
| Requeut Augustin   | [augustin.requeut@student-cs.fr](mailto:augustin.requeut@student-cs.fr)     |