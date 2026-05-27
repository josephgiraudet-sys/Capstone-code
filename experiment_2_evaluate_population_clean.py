from __future__ import annotations

from population_eval_config import TEST_OPPONENTS
from experiment_1_evaluate_single_opponents import run_model_evaluations


# Evaluate Experiment 2 population models with the expanded opponent suite and
# clean evaluation noise.
MODEL_PATTERN = "q_agent_population_*_seed*.pkl"
SAVE_CSV_PATH = "results/experiment_2/population_model_evaluations.csv"

EPISODES = 100
N_ROUNDS = 200
P_NOISE = 0.0
OBSERVATION_MODE = "last_n"
HISTORY_LENGTH = 3
SEED = 42


def main() -> None:
    run_model_evaluations(
        models_dir="models/experiment_2",
        save_csv_path=SAVE_CSV_PATH,
        test_opponents=TEST_OPPONENTS,
        model_pattern=MODEL_PATTERN,
        exclude_model_substrings=["_pnoise"],
        episodes=EPISODES,
        n_rounds=N_ROUNDS,
        p_noise=P_NOISE,
        observation_mode=OBSERVATION_MODE,
        history_length=HISTORY_LENGTH,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
