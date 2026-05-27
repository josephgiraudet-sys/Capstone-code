from __future__ import annotations

from pathlib import Path

import numpy as np

from env import PrisonersDilemmaEnv
from train_agent import QLearningAgent, obs_to_state


# -----------------------------
# Population definitions
# -----------------------------

RECIPROCAL_POPULATION = [
    "tft",
    "mist",
    "tftt",
    "hard_tft",
    "generous_tft",
]

EXPLOITATIVE_POPULATION = [
    "alld",
    "prober",
    "prober_3",
    "hard_prober",
]

MIXED_POPULATION = [
    "tft",
    "hard_tft",
    "alld",
    "prober",
    "prober_3",
]

POPULATIONS: dict[str, list[str]] = {
    "reciprocal": RECIPROCAL_POPULATION,
    "exploitative": EXPLOITATIVE_POPULATION,
    "mixed": MIXED_POPULATION,
}


# -----------------------------
# Training settings
# -----------------------------

SEEDS = [1, 2, 3]
EPISODES = 20000
N_ROUNDS = 200
P_NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2]
OBSERVATION_MODE = "last_n"
HISTORY_LENGTH = 3


def format_noise_label(p_noise: float) -> str:
    return str(p_noise).replace(".", "p")


def train_against_population(
    population_name: str,
    training_opponents: list[str],
    seed: int,
    episodes: int = EPISODES,
    n_rounds: int = N_ROUNDS,
    p_noise: float = 0.0,
    observation_mode: str = OBSERVATION_MODE,
    history_length: int = HISTORY_LENGTH,
    save_path: str | Path | None = None,
) -> None:
    """
    Train one Q-learning agent against a population.

    One opponent is sampled uniformly at random at the start of each episode.
    """
    env = PrisonersDilemmaEnv(
        n_rounds=n_rounds,
        p_noise=p_noise,
        opponent=training_opponents[0],
        observation_mode=observation_mode,
        history_length=history_length,
        render_mode=None,
    )

    agent = QLearningAgent(
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        seed=seed,
    )

    population_rng = np.random.default_rng(seed)
    episode_rewards: list[float] = []
    opponent_counts = {name: 0 for name in training_opponents}

    for episode in range(1, episodes + 1):
        current_opponent = str(population_rng.choice(training_opponents))
        opponent_counts[current_opponent] += 1

        obs, _ = env.reset(
            seed=seed + episode,
            options={"opponent": current_opponent},
        )
        state = obs_to_state(obs)

        done = False
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs_to_state(next_obs)

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
            )

            state = next_state
            total_reward += reward
            done = terminated or truncated

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode % 500 == 0:
            mean_recent_reward = np.mean(episode_rewards[-500:])
            print(
                f"Episode {episode}/{episodes} | "
                f"Population={population_name} | "
                f"Noise={p_noise} | "
                f"Seed={seed} | "
                f"Last opponent={current_opponent} | "
                f"Mean reward (last 500): {mean_recent_reward:.3f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    if save_path is None:
        noise_label = format_noise_label(p_noise)
        save_path = (
            Path("models")
            / "experiment_3"
            / f"q_agent_population_{population_name}_pnoise{noise_label}_seed{seed}.pkl"
        )
    else:
        save_path = Path(save_path)
    agent.save(
        save_path,
        metadata={
            "population_name": population_name,
            "training_opponents": training_opponents,
            "sampling": "uniform_random_per_episode",
            "n_rounds": n_rounds,
            "p_noise": p_noise,
            "training_p_noise": p_noise,
            "observation_mode": observation_mode,
            "history_length": history_length,
            "seed": seed,
        },
    )

    print()
    print(f"Finished training: {population_name} | noise={p_noise} | seed={seed}")
    print(f"Saved model to: {save_path}")
    print(f"Opponent counts: {opponent_counts}")
    print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.3f}")
    print()


def train_all_populations() -> None:
    (Path("models") / "experiment_3").mkdir(parents=True, exist_ok=True)

    total_runs = len(POPULATIONS) * len(P_NOISE_LEVELS) * len(SEEDS)
    current_run = 0

    print("Starting Experiment 3 noisy population training...")
    print(f"Populations: {list(POPULATIONS.keys())}")
    print(f"Seeds: {SEEDS}")
    print(f"Episodes: {EPISODES}")
    print(f"Rounds per episode: {N_ROUNDS}")
    print(f"Noise levels: {P_NOISE_LEVELS}")
    print()

    for population_name, training_opponents in POPULATIONS.items():
        for p_noise in P_NOISE_LEVELS:
            for seed in SEEDS:
                current_run += 1
                print("=" * 80)
                print(f"Run {current_run}/{total_runs}")
                print(f"Population: {population_name}")
                print(f"Opponents: {training_opponents}")
                print(f"Noise: {p_noise}")
                print(f"Seed: {seed}")
                print("=" * 80)

                train_against_population(
                    population_name=population_name,
                    training_opponents=training_opponents,
                    seed=seed,
                    p_noise=p_noise,
                )

    print("Experiment 3 noisy population training completed.")


if __name__ == "__main__":
    train_all_populations()
