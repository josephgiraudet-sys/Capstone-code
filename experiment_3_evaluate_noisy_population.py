from __future__ import annotations

import csv
from pathlib import Path

from population_eval_config import TEST_OPPONENTS
from experiment_1_evaluate_single_opponents import evaluate_model_against_opponent


# Evaluate only the noisy population-trained models created by
# experiment_3_train_noisy_population.py.
MODEL_PATTERN = "q_agent_population_*_pnoise*_seed*.pkl"
SAVE_CSV_PATH = "results/experiment_3/population_noise_model_evaluations.csv"

EPISODES = 100
N_ROUNDS = 200
EVAL_P_NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2]
OBSERVATION_MODE = "last_n"
HISTORY_LENGTH = 3
SEED = 42


FIELDNAMES = [
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
]


def main() -> None:
    models_path = Path("models") / "experiment_3"
    model_files = sorted(models_path.glob(MODEL_PATTERN))
    if not model_files:
        raise FileNotFoundError(
            f"No population models matching '{MODEL_PATTERN}' found in {models_path}"
        )

    total_runs = len(model_files) * len(TEST_OPPONENTS) * len(EVAL_P_NOISE_LEVELS)
    current_run = 0
    all_results = []

    print("Starting Experiment 3 noisy population model evaluations...")
    print(f"Found {len(model_files)} model(s).")
    print(f"Evaluation noise levels: {EVAL_P_NOISE_LEVELS}")
    print(f"Total evaluations: {total_runs}")
    print()

    for model_path in model_files:
        for eval_p_noise in EVAL_P_NOISE_LEVELS:
            for opponent in TEST_OPPONENTS:
                current_run += 1
                print("=" * 80)
                print(
                    f"Evaluation {current_run}/{total_runs} | "
                    f"Model={model_path.stem} | "
                    f"Eval noise={eval_p_noise} | "
                    f"Opponent={opponent}"
                )
                print("=" * 80)

                result = evaluate_model_against_opponent(
                    model_path=model_path,
                    opponent=opponent,
                    episodes=EPISODES,
                    n_rounds=N_ROUNDS,
                    p_noise=eval_p_noise,
                    observation_mode=OBSERVATION_MODE,
                    history_length=HISTORY_LENGTH,
                    seed=SEED,
                )
                all_results.append(result)
                print(result)
                print()

    save_path = Path(SAVE_CSV_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print("Experiment 3 noisy population evaluations completed.")
    print(f"Saved combined results to: {save_path}")


if __name__ == "__main__":
    main()
