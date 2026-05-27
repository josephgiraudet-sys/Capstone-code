from __future__ import annotations

import csv
import re
from pathlib import Path

from experiment_1_evaluate_single_opponents import (
    evaluate_model_against_opponent,
    load_model_metadata,
)
from experiment_4_train_partial_observability import (
    HISTORY_LENGTHS,
    OBSERVATION_MODE,
    PARTIAL_OBSERVABILITY_STRATEGIES,
)


MODELS_DIR = Path("models") / "experiment_4"
RESULTS_PATH = Path("results/experiment_4/partial_observability_model_evaluations.csv")

MODEL_PATTERN = "q_agent_partial_obs_vs_*_last*_seed*.pkl"
EPISODES = 100
N_ROUNDS = 200
P_NOISE = 0.0
SEED = 42

FIELDNAMES = [
    "model",
    "training_population",
    "training_p_noise",
    "training_opponent",
    "training_history_length",
    "training_memory_label",
    "seed",
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


def parse_model_name(model_name: str) -> dict:
    match = re.match(
        r"^q_agent_partial_obs_vs_(?P<opponent>.+)_last(?P<history_length>\d+)_seed(?P<seed>\d+)$",
        model_name,
    )
    if match is None:
        raise ValueError(f"Unexpected Experiment 4 model name: {model_name}")

    return {
        "training_opponent": match.group("opponent"),
        "training_history_length": int(match.group("history_length")),
        "seed": int(match.group("seed")),
    }


def main() -> None:
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    model_files = sorted(MODELS_DIR.glob(MODEL_PATTERN))
    if not model_files:
        raise FileNotFoundError(
            f"No Experiment 4 models matching '{MODEL_PATTERN}' found in {MODELS_DIR}"
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_runs = len(model_files) * len(PARTIAL_OBSERVABILITY_STRATEGIES)
    current_run = 0
    all_results = []

    print("Starting Experiment 4 partial-observability evaluations...")
    print(f"Found {len(model_files)} model(s).")
    print(f"Test opponents: {PARTIAL_OBSERVABILITY_STRATEGIES}")
    print(f"Total evaluations: {total_runs}")
    print()

    for model_path in model_files:
        parsed = parse_model_name(model_path.stem)
        metadata = load_model_metadata(model_path)
        training_history_length = int(
            metadata.get(
                "history_length",
                parsed["training_history_length"],
            )
        )
        observation_mode = str(metadata.get("observation_mode", OBSERVATION_MODE))

        if training_history_length not in HISTORY_LENGTHS:
            raise ValueError(
                f"Unexpected history length {training_history_length} in {model_path}"
            )

        for test_opponent in PARTIAL_OBSERVABILITY_STRATEGIES:
            current_run += 1
            print("=" * 80)
            print(
                f"Evaluation {current_run}/{total_runs} | "
                f"Model={model_path.stem} | "
                f"History length={training_history_length} | "
                f"Opponent={test_opponent}"
            )
            print("=" * 80)

            result = evaluate_model_against_opponent(
                model_path=model_path,
                opponent=test_opponent,
                episodes=EPISODES,
                n_rounds=N_ROUNDS,
                p_noise=P_NOISE,
                observation_mode=observation_mode,
                history_length=training_history_length,
                seed=SEED,
            )
            result.update(
                {
                    "training_opponent": metadata.get(
                        "opponent",
                        parsed["training_opponent"],
                    ),
                    "training_history_length": training_history_length,
                    "training_memory_label": f"last{training_history_length}",
                    "seed": parsed["seed"],
                }
            )
            all_results.append(result)
            print(result)
            print()

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print("Experiment 4 partial-observability evaluations completed.")
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
