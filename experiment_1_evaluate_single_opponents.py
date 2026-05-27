# Experiment 1 evaluation: evaluate single-opponent-trained models.
from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Tuple

import numpy as np

from env import PrisonersDilemmaEnv
from strategies import OPPONENT_REGISTRY

State = Tuple[int, ...]

TRAINED_OPPONENTS = [
    "allc",
    "alld",
    "tft",
    "hard_tft",
    "cycler_ccd",
    "cycler_ddc",
    "generous_tft",
    "grudger",
    "soft_grudger",
    "tftt",
    "prober",
    "joss",
    "random",
]


class QLearningAgent:
    """
    Minimal Q-learning agent used only for loading a saved model
    and selecting greedy actions during evaluation.
    """

    def __init__(self, action_size: int, seed: int = 42) -> None:
        self.action_size = action_size
        self.rng = np.random.default_rng(seed)
        self.q_table: DefaultDict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_size, dtype=np.float64)
        )

    def load(self, filepath: str | Path) -> None:
        filepath = Path(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        loaded_q_table = data["q_table"]

        self.q_table = defaultdict(
            lambda: np.zeros(self.action_size, dtype=np.float64)
        )
        for state, values in loaded_q_table.items():
            self.q_table[state] = np.array(values, dtype=np.float64)

    def get_action(self, state: State) -> int:
        """
        Greedy evaluation policy: always choose the best learned action.
        """
        return int(np.argmax(self.q_table[state]))


def obs_to_state(obs: np.ndarray) -> State:
    """
    Convert observation array into a hashable tuple.
    Example:
        np.array([2, 17]) -> (2, 17)
    """
    return tuple(int(value) for value in obs)


def load_model_metadata(model_path: str | Path) -> dict:
    model_path = Path(model_path)
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return dict(data.get("metadata", {}))


def evaluate_model_against_opponent(
    model_path: str | Path,
    opponent: str,
    episodes: int = 100,
    n_rounds: int = 200,
    p_noise: float = 0.0,
    observation_mode: str = "last_n",
    history_length: int = 3,
    seed: int = 42,
) -> dict:
    """
    Evaluate one saved model against one opponent.
    Returns summary statistics for that matchup.
    """
    env = PrisonersDilemmaEnv(
        n_rounds=n_rounds,
        p_noise=p_noise,
        opponent=opponent,
        observation_mode=observation_mode,
        history_length=history_length,
        render_mode=None,
    )

    agent = QLearningAgent(action_size=env.action_space.n, seed=seed)
    agent.load(model_path)

    episode_rewards = []
    cooperation_rates = []
    defection_rates = []

    for episode in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + episode)
        state = obs_to_state(obs)

        done = False
        total_reward = 0.0
        cooperation_count = 0
        defection_count = 0

        while not done:
            action = agent.get_action(state)

            if action == 0:
                cooperation_count += 1
            else:
                defection_count += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = obs_to_state(next_obs)

            total_reward += reward
            state = next_state
            done = terminated or truncated

        episode_rewards.append(total_reward)
        cooperation_rates.append(cooperation_count / n_rounds)
        defection_rates.append(defection_count / n_rounds)

    env.close()

    metadata = load_model_metadata(model_path)

    return {
        "model": Path(model_path).stem,
        "training_population": metadata.get("population_name", ""),
        "training_p_noise": metadata.get(
            "training_p_noise",
            metadata.get("p_noise", ""),
        ),
        "test_opponent": opponent,
        "episodes": episodes,
        "n_rounds": n_rounds,
        "p_noise": p_noise,
        "observation_mode": observation_mode,
        "history_length": history_length,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_reward_per_round": float(np.mean(episode_rewards) / n_rounds),
        "mean_cooperation_rate": float(np.mean(cooperation_rates)),
        "mean_defection_rate": float(np.mean(defection_rates)),
    }


def run_model_evaluations(
    models_dir: str = "models/experiment_1",
    save_csv_path: str = "results/experiment_1/all_model_evaluations.csv",
    test_opponents: list[str] | None = None,
    model_pattern: str = "*.pkl",
    exclude_model_substrings: list[str] | None = None,
    episodes: int = 100,
    n_rounds: int = 200,
    p_noise: float = 0.0,
    observation_mode: str = "last_n",
    history_length: int = 3,
    seed: int = 42,
) -> None:
    """
    Evaluate every saved model in models_dir against every opponent in test_opponents.
    Save one combined CSV with all results.
    """
    if test_opponents is None:
        test_opponents = TRAINED_OPPONENTS

    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_path}")

    model_files = sorted(models_path.glob(model_pattern))
    if exclude_model_substrings is not None:
        model_files = [
            model_path
            for model_path in model_files
            if not any(
                excluded in model_path.name
                for excluded in exclude_model_substrings
            )
        ]
    if not model_files:
        raise FileNotFoundError(
            f"No .pkl model files matching '{model_pattern}' found in {models_path}"
        )

    total_runs = len(model_files) * len(test_opponents)
    current_run = 0
    all_results = []

    print("Starting model evaluations...")
    print(f"Models directory: {models_path}")
    print(f"Found {len(model_files)} model(s).")
    print(f"Test opponents: {test_opponents}")
    print(f"Total evaluations: {total_runs}")
    print()

    for model_path in model_files:
        for opponent in test_opponents:
            current_run += 1

            print("=" * 80)
            print(
                f"Evaluation {current_run}/{total_runs} | "
                f"Model={model_path.stem} | Opponent={opponent}"
            )
            print("=" * 80)

            result = evaluate_model_against_opponent(
                model_path=model_path,
                opponent=opponent,
                episodes=episodes,
                n_rounds=n_rounds,
                p_noise=p_noise,
                observation_mode=observation_mode,
                history_length=history_length,
                seed=seed,
            )

            all_results.append(result)
            print(result)
            print()

    save_path = Path(save_csv_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "training_population",
                "training_p_noise",
                "test_opponent",
                "episodes",
                "n_rounds",
                "p_noise",
                "observation_mode",
                "history_length",
                "mean_reward",
                "std_reward",
                "mean_reward_per_round",
                "mean_cooperation_rate",
                "mean_defection_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("All model evaluations completed.")
    print(f"Saved combined results to: {save_path}")


if __name__ == "__main__":
    run_model_evaluations(
        models_dir="models/experiment_1",
        save_csv_path="results/experiment_1/all_model_evaluations.csv",
        test_opponents=TRAINED_OPPONENTS,
        episodes=100,
        n_rounds=200,
        p_noise=0.0,
        observation_mode="last_n",
        history_length=3,
        seed=42,
    )
