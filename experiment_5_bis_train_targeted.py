from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from env import PrisonersDilemmaEnv
from train_agent import QLearningAgent, obs_to_state


TARGET_STRATEGIES = [
    "hard_tft",
    "grudger",
]


@dataclass(frozen=True)
class HyperparameterConfig:
    name: str
    learning_rate_start: float
    learning_rate_end: float
    learning_rate_schedule: str
    discount_factor: float
    initial_epsilon: float
    epsilon_min: float
    epsilon_decay: float


def exponential_decay_rate(
    start_value: float,
    end_value: float,
    n_steps: int,
) -> float:
    if start_value <= 0 or end_value <= 0:
        raise ValueError("Exponential decay values must be positive.")
    if n_steps <= 1:
        return 1.0
    return float((end_value / start_value) ** (1 / (n_steps - 1)))


SEEDS = [1, 2, 3]
TRAINING_EPISODES = 20_000
ROUNDS_PER_EPISODE = 100
P_NOISE = 0.0
OBSERVATION_MODE = "last_n"
HISTORY_LENGTH = 3
INITIAL_Q_VALUE_FOR_COOPERATE = 1.0
INITIAL_Q_VALUE_FOR_DEFECT = 0.0
IMPROVED_BASELINE_LABEL = "qinit_cooperate"

BASELINE_EPSILON_DECAY = 0.995


HYPERPARAMETER_CONFIGS = [
    HyperparameterConfig(
        name="baseline",
        learning_rate_start=0.1,
        learning_rate_end=0.1,
        learning_rate_schedule="constant",
        discount_factor=0.95,
        initial_epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=BASELINE_EPSILON_DECAY,
    ),
]


MODELS_DIR = Path("models") / "experiment_5_bis"


def inverse_time_learning_rate(
    episode: int,
    start_value: float,
    end_value: float,
    n_steps: int,
) -> float:
    if n_steps <= 1:
        return end_value
    decay_rate = (start_value / end_value - 1) / (n_steps - 1)
    return float(start_value / (1 + decay_rate * (episode - 1)))


def learning_rate_for_episode(
    config: HyperparameterConfig,
    episode: int,
) -> float:
    if config.learning_rate_schedule == "constant":
        return config.learning_rate_start
    if config.learning_rate_schedule == "inverse_time":
        return inverse_time_learning_rate(
            episode=episode,
            start_value=config.learning_rate_start,
            end_value=config.learning_rate_end,
            n_steps=TRAINING_EPISODES,
        )
    raise ValueError(f"Unknown learning-rate schedule: {config.learning_rate_schedule}")


def train_one_model(
    opponent: str,
    config: HyperparameterConfig,
    seed: int,
    save_path: Path,
) -> None:
    env = PrisonersDilemmaEnv(
        n_rounds=ROUNDS_PER_EPISODE,
        p_noise=P_NOISE,
        opponent=opponent,
        observation_mode=OBSERVATION_MODE,
        history_length=HISTORY_LENGTH,
        render_mode=None,
    )

    agent = QLearningAgent(
        action_size=env.action_space.n,
        learning_rate=config.learning_rate_start,
        discount_factor=config.discount_factor,
        epsilon=config.initial_epsilon,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=config.epsilon_min,
        seed=seed,
        initial_q_values=[
            INITIAL_Q_VALUE_FOR_COOPERATE,
            INITIAL_Q_VALUE_FOR_DEFECT,
        ],
    )

    episode_rewards: list[float] = []

    for episode in range(1, TRAINING_EPISODES + 1):
        agent.learning_rate = learning_rate_for_episode(config, episode)

        obs, _ = env.reset(seed=seed + episode)
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
                f"Episode {episode}/{TRAINING_EPISODES} | "
                f"Opponent={opponent} | "
                f"Config={config.name} | "
                f"Seed={seed} | "
                f"Mean reward (last 500): {mean_recent_reward:.3f} | "
                f"LR: {agent.learning_rate:.6f} | "
                f"Epsilon: {agent.epsilon:.6f}"
            )

    env.close()

    agent.save(
        save_path,
        metadata={
            "experiment": "experiment_5_bis_standard_state_hyperparameters",
            "training_opponent": opponent,
            "opponent": opponent,
            "hyperparameter_config": config.name,
            "learning_rate_start": config.learning_rate_start,
            "learning_rate_end": config.learning_rate_end,
            "learning_rate_schedule": config.learning_rate_schedule,
            "final_learning_rate": agent.learning_rate,
            "discount_factor": config.discount_factor,
            "initial_epsilon": config.initial_epsilon,
            "epsilon_decay": config.epsilon_decay,
            "epsilon_min": config.epsilon_min,
            "final_epsilon": agent.epsilon,
            "initial_q_value_for_cooperate": INITIAL_Q_VALUE_FOR_COOPERATE,
            "initial_q_value_for_defect": INITIAL_Q_VALUE_FOR_DEFECT,
            "history_flags_enabled": False,
            "model_variant": "improved_baseline",
            "observation_mode": OBSERVATION_MODE,
            "history_length": HISTORY_LENGTH,
            "training_episodes": TRAINING_EPISODES,
            "n_rounds": ROUNDS_PER_EPISODE,
            "p_noise": P_NOISE,
            "evaluation_epsilon": 0.0,
            "seed": seed,
        },
    )

    print()
    print(
        f"Finished training: opponent={opponent} | config={config.name} | seed={seed}"
    )
    print(f"Saved model to: {save_path}")
    print(f"Final learning rate: {agent.learning_rate:.6f}")
    print(f"Final epsilon: {agent.epsilon:.6f}")
    print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.3f}")
    print()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(TARGET_STRATEGIES) * len(HYPERPARAMETER_CONFIGS) * len(SEEDS)
    current_run = 0

    print("Starting Experiment 5 bis standard-state hyperparameter training...")
    print(f"Target strategies: {TARGET_STRATEGIES}")
    print(f"Hyperparameter configs: {[config.name for config in HYPERPARAMETER_CONFIGS]}")
    print(f"Observation mode: {OBSERVATION_MODE}")
    print(f"History length: {HISTORY_LENGTH}")
    print(
        "Initial Q-values: "
        f"C={INITIAL_Q_VALUE_FOR_COOPERATE}, D={INITIAL_Q_VALUE_FOR_DEFECT}"
    )
    print(f"Training episodes: {TRAINING_EPISODES}")
    print(f"Rounds per episode: {ROUNDS_PER_EPISODE}")
    print(f"Total runs: {total_runs}")
    print()

    for opponent in TARGET_STRATEGIES:
        for config in HYPERPARAMETER_CONFIGS:
            for seed in SEEDS:
                current_run += 1
                save_path = (
                    MODELS_DIR
                    / (
                        f"q_agent_exp5bis_{IMPROVED_BASELINE_LABEL}_vs_"
                        f"{opponent}_{config.name}_seed{seed}.pkl"
                    )
                )

                print("=" * 80)
                print(f"Training {current_run}/{total_runs}")
                print(f"Opponent: {opponent}")
                print(f"Config: {config}")
                print(f"Seed: {seed}")
                print(f"Saving to: {save_path}")
                print("=" * 80)

                if save_path.exists():
                    print(f"Skipping existing model: {save_path}")
                    print()
                    continue

                train_one_model(
                    opponent=opponent,
                    config=config,
                    seed=seed,
                    save_path=save_path,
                )

    print("Experiment 5 bis standard-state hyperparameter training completed.")


if __name__ == "__main__":
    main()
