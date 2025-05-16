# private-tuning-ml

This repository provides to reproduce the experiments of the paper "Bandit-based hyperparameter tuning":



## Installation

Python version used is 3.12.9

To install the required Python dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Experiments

This project includes two main scripts for tuning hyperparameters under differential privacy:

- `sequential_halving_sg.py`
- `random_stopping__sg.py`

Both scripts include a variable named `exp` that controls which experiment is run:

| `exp` Value | Description                                         |
|-------------|-----------------------------------------------------|
| `1`         | Plots **simple regret** as a function of **r**      |
| `2`         | Plots **simple regret** as a function of **ε (epsilon)** |

### Run Commands

To run the experiments, execute:

```bash
PYTHONPATH=. python sequential_halving_sg.py
PYTHONPATH=. python random_stopping__sg.py
```