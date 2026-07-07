[![License](https://img.shields.io/github/license/giansimone/sac-mujoco-humanoid)](https://github.com/giansimone/sac-mujoco-humanoid/blob/main/LICENSE)

# Soft Actor-Critic (SAC) for MuJoCo Humanoid Environment

A PyTorch implementation of the Soft Actor-Critic (SAC) algorithm to train an agent to play with the Humanoid environment from MuJoCo.

## Installation

You can clone the repository and install the required dependencies using Poetry or pip. This project requires **Python 3.13**.

### Using Poetry (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/giansimone/sac-mujoco-humanoid.git
cd sac-mujoco-humanoid

# 2. Initialize environment and install dependencies
poetry env use python3.13
poetry install

# 3. Activate the shell
eval $(poetry env activate)
```

### Using Pip

```bash
# 1. Clone the repository
git clone https://github.com/giansimone/sac-mujoco-humanoid.git
cd sac-mujoco-humanoid

# 2. Create and activate virtual environment
python3.13 -m venv venv
source venv/bin/activate

# 3. Install package in editable mode
pip install -e .
```

## Project Structure

```bash
sac-mujoco-humanoid/
├── sac_mujoco_humanoid/
│   ├── __init__.py
│   ├── agent.py       # SAC implementation (Actor/Critic)
│   ├── buffer.py      # Replay Buffer
│   ├── config.yaml    # Training hyperparameters
│   ├── environment.py # Gym environment wrappers
│   ├── enjoy.py       # Evaluation script
│   ├── export.py      # Hugging Face export script
│   ├── model.py       # PyTorch Network definitions
│   ├── train.py       # Main training loop
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml
```

## Usage

Ensure you are in the ```sac_mujoco_humanoid``` source directory where ```config.yaml``` is located before running these commands.

```bash
cd sac-mujoco-humanoid
```

### Training

Train a SAC agent with the default configuration.

**Note:** The Replay Buffer pre-allocates memory. Ensure your system has at least 8GB of RAM available.

```bash
python -m train
```

### Configuration

Edit ```config.yaml``` to customise training parameters.

```yaml
#Environment
env_name: Humanoid-v5

#Network Architecture
hidden_dim: 256

#Training
total_steps: 2_000_000
buffer_size: 1_000_000
batch_size: 256
start_steps: 10_000
updates_per_step: 1

#SAC Agent
lr: 0.0003
gamma: 0.99
tau: 0.005
alpha: 0.2
auto_tune_alpha: True

#Logging
log_dir: runs/

#System
seed: 42
```

### Enjoying a Trained Agent

Watch a trained agent by running the enjoy script. Point the artifact argument to your saved model file.

```bash
python -m enjoy --artifact runs/sac_Humanoid-v5_YYYY-MM-DD_HHhMMmSSs/final_model.pt --num-episodes 5
```
### Exporting to Hugging Face Hub

Share your trained model, config, and a replay video to the Hugging Face Hub.

```bash
python -m export \
    --username YOUR_HF_USERNAME \
    --repo-name sac-mujoco-humanoid \
    --artifact-path runs/sac_Humanoid-v5_YYYY-MM-DD_HHhMMmSSs/final_model.pt \
    --movie-fps 30 \
    --n-eval 10
```

This will automatically:

- Upload the model weights and config.

- Generate a model card with evaluation metrics (Mean Reward +/- Std).

- Record and upload a video of the agent.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.