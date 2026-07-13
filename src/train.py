"""
Training script for SAC agent in MuJoCo Humanoid environment.
"""
from pathlib import Path
from datetime import datetime

import torch

from agent import SAC 
from environment import make_env
from utils import set_seed, load_config, save_config


def train(config_filename: Path = Path("config.yaml"), device_id: str = "cpu"):
    """Train the SAC agent in the MuJoCo Humanoid environment."""
    config = load_config(config_filename)
    device = torch.device(device_id)

    set_seed(config["seed"])

    run_name = "sac_" + config["env_name"] + "_" + datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    log_dir = Path(config["log_dir"])
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config.copy(), run_dir / "config.yaml")

    history_file = run_dir / "training_history.csv"
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            f.write("Step,Reward\n")
    except Exception as e:
        print(f"Aviso: Falha ao criar arquivo de histórico CSV: {e}")

    env_name = config["env_name"]

    env, state_dim, action_dim = make_env(
        env_name,
        render_mode=None,
    )

    print(f"| State space: {state_dim}, Action space: {action_dim}")

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config["hidden_dim"],
        capacity=config["buffer_size"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        alpha=config["alpha"],
        auto_tune_alpha=config["auto_tune_alpha"],
        action_space=env.action_space,
        device=device,
    )

    print(f"Total timesteps: {config['total_steps']}")

    state, _ = env.reset(seed=config["seed"])
    episode_reward = 0.

    for step in range(config["total_steps"]):
        if step < config["start_steps"]:
            action = env.action_space.sample()
        else:
            action_tensor = agent.select_action(state)
            action = action_tensor.cpu().numpy().flatten()

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        agent.memory.push(
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(float(reward), dtype=torch.float32).view(1),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(terminated, dtype=torch.int64).view(1),
        )

        state = next_state
        episode_reward += float(reward)

        if step >= config["start_steps"]:
            agent.learn()

        if done:
            print(f"Step: {step}, Reward: {episode_reward:.2f}, Alpha: {agent.alpha:.4f}")
            try:
                with open(history_file, "a", encoding="utf-8") as f:
                    f.write(f"{step},{episode_reward:.2f}\n")
            except Exception:
                # Falha silenciosa/Aviso leve para garantir que I/O errors não matem o loop de treino
                print(f"Aviso: Não foi possível escrever no histórico CSV no step {step}.")
                
            state, _ = env.reset()
            episode_reward = 0

    agent.save_model(run_dir / "final_model.pt")

    env.close()


if __name__ == "__main__":
    train()
