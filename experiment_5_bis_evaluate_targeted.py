from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import numpy as np

from env import PrisonersDilemmaEnv
from experiment_5_bis_train_targeted import (
    HISTORY_LENGTH,
    IMPROVED_BASELINE_LABEL,
    INITIAL_Q_VALUE_FOR_COOPERATE,
    INITIAL_Q_VALUE_FOR_DEFECT,
    OBSERVATION_MODE,
    P_NOISE,
    ROUNDS_PER_EPISODE,
    TARGET_STRATEGIES,
)
from train_agent import obs_to_state


MODELS_DIR = Path("models") / "experiment_5_bis"
RESULTS_PATH = Path("results") / "experiment_5_bis" / "targeted_model_evaluations.csv"

MODEL_PATTERNS = [
    "q_agent_exp5bis_standard_vs_*.pkl",
    f"q_agent_exp5bis_{IMPROVED_BASELINE_LABEL}_vs_*.pkl",
]
TEST_OPPONENTS = TARGET_STRATEGIES
EPISODES = 100
EVALUATION_EPSILON = 0.0
SEED = 42

MAX_ATTAINABLE_PAYOFFS = {
    "hard_tft": 302,
    "grudger": 302,
}

FIELDNAMES = [
    "model",
    "training_opponent",
    "hyperparameter_config",
    "learning_rate_start",
    "learning_rate_end",
    "learning_rate_schedule",
    "final_learning_rate",
    "discount_factor",
    "initial_epsilon",
    "epsilon_decay",
    "epsilon_min",
    "final_epsilon",
    "observation_mode",
    "history_length",
    "history_flags_enabled",
    "model_variant",
    "seed",
    "test_opponent",
    "episodes",
    "n_rounds",
    "p_noise",
    "evaluation_epsilon",
    "mean_reward",
    "std_reward",
    "mean_reward_per_round",
    "cooperation_rate",
    "defection_rate",
    "first_defection_rate",
    "average_round_of_first_defection",
    "payoff_percent_of_max",
    "q_c_minus_d_initial_state",
    "q_c_minus_d_after_first_agent_defection",
]


class GreedyLoadedAgent:
    def __init__(self, action_size: int) -> None:
        self.action_size = action_size
        self.default_q_values = np.zeros(self.action_size, dtype=np.float64)
        self.q_table: DefaultDict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: self.default_q_values.copy()
        )

    def load(self, model_path: str | Path) -> dict:
        with Path(model_path).open("rb") as f:
            data = pickle.load(f)

        metadata = dict(data.get("metadata", {}))
        stored_initial_q_values = data.get("initial_q_values")
        if stored_initial_q_values is not None:
            self.default_q_values = np.array(stored_initial_q_values, dtype=np.float64)
        else:
            self.default_q_values = np.array(
                [
                    float(metadata.get("initial_q_value_for_cooperate", 0.0)),
                    float(metadata.get("initial_q_value_for_defect", 0.0)),
                ],
                dtype=np.float64,
            )

        self.q_table = defaultdict(lambda: self.default_q_values.copy())
        for state, values in data["q_table"].items():
            self.q_table[tuple(state)] = np.array(values, dtype=np.float64)

        return metadata

    def get_action(self, state: tuple[int, ...]) -> int:
        return int(np.argmax(self.q_table[state]))


def q_c_minus_d(
    q_table: DefaultDict[tuple[int, ...], np.ndarray],
    state: tuple[int, ...],
) -> float | str:
    if state not in q_table:
        return ""
    q_values = q_table[state]
    return float(q_values[0] - q_values[1])


def model_variant(metadata: dict) -> str:
    explicit_variant = str(metadata.get("model_variant", ""))
    if explicit_variant:
        return explicit_variant

    config = str(metadata.get("hyperparameter_config", ""))
    initial_q_for_cooperate = float(metadata.get("initial_q_value_for_cooperate", 0.0))
    initial_q_for_defect = float(metadata.get("initial_q_value_for_defect", 0.0))
    if (
        config == "baseline"
        and initial_q_for_cooperate == INITIAL_Q_VALUE_FOR_COOPERATE
        and initial_q_for_defect == INITIAL_Q_VALUE_FOR_DEFECT
    ):
        return "improved_baseline"
    return config


def diagnostic_state_from_history(
    history: list[tuple[int, int]],
    history_length: int,
) -> tuple[int, ...]:
    encoded_history = [
        PrisonersDilemmaEnv._encode_outcome(agent_action, opp_action)
        for agent_action, opp_action in history
    ]
    unseen_token = 4
    visible_history = encoded_history[-history_length:]
    padded_history = [unseen_token] * (history_length - len(visible_history)) + visible_history
    return tuple([*padded_history, len(history)])


def discover_model_files() -> list[Path]:
    model_files: list[Path] = []
    seen_paths: set[Path] = set()
    for pattern in MODEL_PATTERNS:
        for model_path in sorted(MODELS_DIR.glob(pattern)):
            resolved = model_path.resolve()
            if resolved not in seen_paths:
                model_files.append(model_path)
                seen_paths.add(resolved)
    return model_files


def evaluate_model(
    model_path: Path,
    test_opponent: str,
    metadata: dict,
) -> dict:
    history_length = int(metadata.get("history_length", HISTORY_LENGTH))
    observation_mode = str(metadata.get("observation_mode", OBSERVATION_MODE))

    env = PrisonersDilemmaEnv(
        n_rounds=ROUNDS_PER_EPISODE,
        p_noise=P_NOISE,
        opponent=test_opponent,
        observation_mode=observation_mode,
        history_length=history_length,
        render_mode=None,
    )

    agent = GreedyLoadedAgent(action_size=env.action_space.n)
    agent.load(model_path)

    episode_rewards: list[float] = []
    cooperation_rates: list[float] = []
    defection_rates: list[float] = []
    first_defection_rounds: list[int] = []

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset(seed=SEED + episode)
        state = obs_to_state(obs)

        total_reward = 0.0
        cooperation_count = 0
        defection_count = 0
        first_defection_round = None
        done = False

        while not done:
            action = agent.get_action(state)
            if action == 0:
                cooperation_count += 1
            else:
                defection_count += 1
                if first_defection_round is None:
                    first_defection_round = env._round_idx + 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = obs_to_state(next_obs)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        cooperation_rates.append(cooperation_count / ROUNDS_PER_EPISODE)
        defection_rates.append(defection_count / ROUNDS_PER_EPISODE)
        if first_defection_round is not None:
            first_defection_rounds.append(first_defection_round)

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    max_payoff = MAX_ATTAINABLE_PAYOFFS.get(test_opponent, np.nan)
    payoff_percent_of_max = (
        float(100 * mean_reward / max_payoff)
        if not np.isnan(max_payoff)
        else ""
    )

    return {
        "model": model_path.stem,
        "training_opponent": metadata.get("training_opponent", ""),
        "hyperparameter_config": metadata.get("hyperparameter_config", ""),
        "learning_rate_start": metadata.get("learning_rate_start", ""),
        "learning_rate_end": metadata.get("learning_rate_end", ""),
        "learning_rate_schedule": metadata.get("learning_rate_schedule", ""),
        "final_learning_rate": metadata.get("final_learning_rate", ""),
        "discount_factor": metadata.get("discount_factor", ""),
        "initial_epsilon": metadata.get("initial_epsilon", ""),
        "epsilon_decay": metadata.get("epsilon_decay", ""),
        "epsilon_min": metadata.get("epsilon_min", ""),
        "final_epsilon": metadata.get("final_epsilon", ""),
        "observation_mode": observation_mode,
        "history_length": history_length,
        "history_flags_enabled": metadata.get("history_flags_enabled", False),
        "model_variant": model_variant(metadata),
        "seed": metadata.get("seed", ""),
        "test_opponent": test_opponent,
        "episodes": EPISODES,
        "n_rounds": ROUNDS_PER_EPISODE,
        "p_noise": P_NOISE,
        "evaluation_epsilon": EVALUATION_EPSILON,
        "mean_reward": mean_reward,
        "std_reward": float(np.std(episode_rewards)),
        "mean_reward_per_round": mean_reward / ROUNDS_PER_EPISODE,
        "cooperation_rate": float(np.mean(cooperation_rates)),
        "defection_rate": float(np.mean(defection_rates)),
        "first_defection_rate": len(first_defection_rounds) / EPISODES,
        "average_round_of_first_defection": (
            float(np.mean(first_defection_rounds)) if first_defection_rounds else ""
        ),
        "payoff_percent_of_max": payoff_percent_of_max,
        "q_c_minus_d_initial_state": q_c_minus_d(
            agent.q_table,
            diagnostic_state_from_history([], history_length),
        ),
        "q_c_minus_d_after_first_agent_defection": q_c_minus_d(
            agent.q_table,
            diagnostic_state_from_history([(1, 0)], history_length),
        ),
    }


def main() -> None:
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    model_files = discover_model_files()
    if not model_files:
        raise FileNotFoundError(
            f"No Experiment 5 bis models matching {MODEL_PATTERNS} "
            f"found in {MODELS_DIR}"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_runs = len(model_files) * len(TEST_OPPONENTS)
    current_run = 0
    all_results = []

    print("Starting Experiment 5 bis feature-augmented evaluations...")
    print(f"Found {len(model_files)} model(s).")
    print(f"Test opponents: {TEST_OPPONENTS}")
    print(f"Total evaluations: {total_runs}")
    print()

    for model_path in model_files:
        with model_path.open("rb") as f:
            metadata = dict(pickle.load(f).get("metadata", {}))

        for test_opponent in TEST_OPPONENTS:
            current_run += 1
            print("=" * 80)
            print(
                f"Evaluation {current_run}/{total_runs} | "
                f"Model={model_path.stem} | Opponent={test_opponent}"
            )
            print("=" * 80)

            result = evaluate_model(
                model_path=model_path,
                test_opponent=test_opponent,
                metadata=metadata,
            )
            all_results.append(result)
            print(result)
            print()

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print("Experiment 5 bis feature-augmented evaluations completed.")
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
