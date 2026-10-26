"""
Module to create and manage the MuJoCo Humanoid environment.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation


def save_obs_stats(env: gym.Env, filepath: Path) -> None:
    """Save the running mean/var of a NormalizeObservation-wrapped env."""
    rms = env.obs_rms
    np.savez(filepath, mean=rms.mean, var=rms.var, count=rms.count)


def load_obs_stats(env: gym.Env, filepath: Path) -> None:
    """Load saved normalization statistics and freeze them (for evaluation)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"normalize_obs=True, mas as estatísticas não foram encontradas: {filepath}"
        )
    data = np.load(filepath)
    env.obs_rms.mean = data["mean"]
    env.obs_rms.var = data["var"]
    env.obs_rms.count = float(data["count"])
    env.update_running_mean = False


def make_env(
    env_id: str,
    render_mode: str | None = None,
    normalize_obs: bool = False,
    obs_stats_path: Path | None = None,
) -> Tuple[gym.Env, int, int]:
    """Create and wrap environment.

    Se normalize_obs=True, aplica NormalizeObservation. Quando obs_stats_path
    é fornecido (avaliação/inferência), carrega as estatísticas salvas no
    treino e congela a atualização; no treino, deixe como None.
    """
    env = gym.make(env_id, render_mode=render_mode, camera_name="track")

    if normalize_obs:
        env = NormalizeObservation(env)
        if obs_stats_path is not None:
            load_obs_stats(env, obs_stats_path)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    return env, state_dim, action_dim


def make_vec_env(
    env_id: str,
    num_envs: int,
    render_mode: str | None = None,
) -> Tuple[gym.vector.VectorEnv, int, int]:
    "Create and wrap environment."
    envs = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode="async",
        render_mode=render_mode,
    )

    envs = gym.wrappers.vector.ClipAction(envs)

    env_state_dim = envs.single_observation_space.shape[0]
    env_action_dim = envs.single_action_space.shape[0]

    return envs, env_state_dim, env_action_dim
