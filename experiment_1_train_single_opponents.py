from __future__ import annotations

from pathlib import Path

from train_agent import train_q_learning

def run_all_training_experiments() -> None:
    """
    Run training for all opponents and save results.
    """
    opponents = [
        "allc",
        "alld",
        "random",
        "tft",
        "hard_tft",
        "tftt",
        "generous_tft",
        "grudger",
        "soft_grudger",
        "joss",
        "prober",
        "cycler_ccd",
        "cycler_ddc",
    ]

    seeds = [1, 2, 3]

    episodes = 20000
    n_rounds = 200
    p_noise = 0.0
    observation_mode = "last_n"
    history_length = 3

    models_dir = Path("models") / "experiment_1"
    models_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(opponents) * len(seeds)
    current_run = 0

    print("Starting training experiments...")
    print(f"Total runs: {total_runs}")
    print(f"Opponents: {opponents}")
    print(f"Seeds: {seeds}")
    print(f"Noise: {p_noise}")
    print(f"Observation mode: {observation_mode}")
    print(f"History length: {history_length}")
    print()

    for opponent in opponents:
        for seed in seeds:
            current_run += 1

            save_path = models_dir / f"q_agent_vs_{opponent}_seed{seed}.pkl"

            print("=" * 70)
            print(
                f"Run {current_run}/{total_runs} | "
                f"Opponent={opponent} | Seed={seed}"
            )
            print(f"Saving to: {save_path}")
            print("=" * 70)


            train_q_learning(
                episodes=episodes,
                n_rounds=n_rounds,
                p_noise=p_noise,
                opponent=opponent,
                observation_mode=observation_mode,
                history_length=history_length,
                seed=seed,
                save_path=str(save_path),
            )

            print()

    print("All training experiments completed.")

if __name__ == "__main__":
    run_all_training_experiments()
