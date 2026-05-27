from __future__ import annotations

from pathlib import Path

from experiment_3_train_noisy_population import (
    EPISODES,
    HISTORY_LENGTH,
    N_ROUNDS,
    OBSERVATION_MODE,
    POPULATIONS,
    SEEDS,
    train_against_population,
)


P_NOISE = 0.0


def train_clean_populations() -> None:
    models_dir = Path("models") / "experiment_2"
    models_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(POPULATIONS) * len(SEEDS)
    current_run = 0

    print("Starting Experiment 2 clean population training...")
    print(f"Populations: {list(POPULATIONS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Episodes: {EPISODES}")
    print(f"Rounds per episode: {N_ROUNDS}")
    print(f"Noise: {P_NOISE}")
    print()

    for population_name, training_opponents in POPULATIONS.items():
        for seed in SEEDS:
            current_run += 1
            save_path = models_dir / f"q_agent_population_{population_name}_seed{seed}.pkl"

            print("=" * 80)
            print(f"Run {current_run}/{total_runs}")
            print(f"Population: {population_name}")
            print(f"Opponents: {training_opponents}")
            print(f"Noise: {P_NOISE}")
            print(f"Seed: {seed}")
            print(f"Saving to: {save_path}")
            print("=" * 80)

            train_against_population(
                population_name=population_name,
                training_opponents=training_opponents,
                seed=seed,
                p_noise=P_NOISE,
                save_path=save_path,
            )

    print("Experiment 2 clean population training completed.")


if __name__ == "__main__":
    train_clean_populations()
