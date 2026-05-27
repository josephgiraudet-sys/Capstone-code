from __future__ import annotations

from pathlib import Path

from train_agent import train_q_learning


PARTIAL_OBSERVABILITY_STRATEGIES = [
    "tftt",
    "hard_tft",
    "cycler_ddc",
    "cycler_ccd",
    "hard_majo",
    "soft_majo",
]

HISTORY_LENGTHS = [1, 3]
SEEDS = [1, 2, 3]

EPISODES = 20000
N_ROUNDS = 200
P_NOISE = 0.0
OBSERVATION_MODE = "last_n"

MODELS_DIR = Path("models") / "experiment_4"


def memory_label(history_length: int) -> str:
    return f"last{history_length}"


def train_partial_observability_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(PARTIAL_OBSERVABILITY_STRATEGIES) * len(HISTORY_LENGTHS) * len(SEEDS)
    current_run = 0

    print("Starting Experiment 4 partial-observability training...")
    print(f"Strategies: {PARTIAL_OBSERVABILITY_STRATEGIES}")
    print(f"History lengths: {HISTORY_LENGTHS}")
    print(f"Seeds: {SEEDS}")
    print(f"Episodes: {EPISODES}")
    print(f"Rounds per episode: {N_ROUNDS}")
    print()

    for history_length in HISTORY_LENGTHS:
        for opponent in PARTIAL_OBSERVABILITY_STRATEGIES:
            for seed in SEEDS:
                current_run += 1
                model_path = (
                    MODELS_DIR
                    / f"q_agent_partial_obs_vs_{opponent}_{memory_label(history_length)}_seed{seed}.pkl"
                )

                print("=" * 80)
                print(f"Training {current_run}/{total_runs}")
                print(f"Opponent: {opponent}")
                print(f"Observation mode: {OBSERVATION_MODE}")
                print(f"History length: {history_length}")
                print(f"Seed: {seed}")
                print(f"Saving to: {model_path}")
                print("=" * 80)

                train_q_learning(
                    episodes=EPISODES,
                    n_rounds=N_ROUNDS,
                    p_noise=P_NOISE,
                    opponent=opponent,
                    observation_mode=OBSERVATION_MODE,
                    history_length=history_length,
                    seed=seed,
                    save_path=str(model_path),
                )

                print()

    print("Experiment 4 partial-observability training completed.")


if __name__ == "__main__":
    train_partial_observability_models()
