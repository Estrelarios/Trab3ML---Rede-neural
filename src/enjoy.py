"""
Module to enjoy a trained SAC agent playing MuJoCo Humanoid environment.
"""
import argparse
from pathlib import Path

from agent import SAC
from environment import make_env
from utils import load_config, record_movie


def enjoy(artifact_path: Path, n_episodes: int) -> None:
    """Enjoy a trained SAC agent on the MuJoCo Humanoid environment."""
    config_path = artifact_path.parent / "config.yaml"
    config = load_config(config_path)

    env_name = config["env_name"]
    hidden_dim = config["hidden_dim"]

    env, state_dim, action_dim = make_env(
        env_name,
        render_mode="human",
    )

    agent = SAC(
        state_dim, action_dim, hidden_dim, action_space=env.action_space
    )
    agent.load_model(artifact_path)

    for episode in range(1, n_episodes + 1):
        
        state, _ = env.reset()
        done = False
        episode_reward = 0.
        step = 0
        while not done:
            step += 1
            
            action_tensor = agent.select_action(state, deterministic=True)
            action = action_tensor.cpu().numpy().flatten()

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
                
            state = next_state
            episode_reward += float(reward)
            
        # video_path = f"videos/episode_{episode}.mp4"
        # record_movie(env, agent, video_path, 60)
        print(f"MuJoCo Humanoid Episode {episode} | Reward: {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        "-a",
        type=str,
        required=True,
        help="The path to the trained model artifact to play MuJoCo Humanoid.",
    )
    parser.add_argument(
        "--num-episodes",
        "-n",
        type=int,
        default=10,
        help="The number of MuJoCo Humanoid episodes to enjoy.",
    )
    args = parser.parse_args()

    artifact = Path(args.artifact)
    num_episodes = args.num_episodes

    enjoy(artifact_path=artifact, n_episodes=num_episodes)
