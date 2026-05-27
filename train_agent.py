# train_agent.py
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Sequence, Tuple

import numpy as np

from env import PrisonersDilemmaEnv

State = Tuple[int, ...]

class QLearningAgent:
    """
    Simple tabular Q-learning agent for a Gymnasium environment
    with discrete actions and tuple-based observations.

    State format in this project:
        tuple(obs)

    Action format:
        0 = Cooperate
        1 = Defect
    """

    def __init__(
        self,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: int = 42,
        initial_q_values: Sequence[float] | None = None,
    ) -> None:
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)
        if initial_q_values is None:
            self.initial_q_values = np.zeros(self.action_size, dtype=np.float64)
        else:
            if len(initial_q_values) != self.action_size:
                raise ValueError("initial_q_values must match action_size.")
            self.initial_q_values = np.array(initial_q_values, dtype=np.float64)

        # Q-table:
        # key   -> state tuple, e.g. (0, 0)
        # value -> np.array of Q-values for each action
        self.q_table: DefaultDict[State, np.ndarray] = defaultdict(
            lambda: self.initial_q_values.copy()
        )

    def get_action(self, state: State) -> int:
        """
        Epsilon-greedy action selection.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        terminated: bool,
    ) -> None:
        """
        Standard Q-learning update:
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]

        if terminated:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])

        td_error = target - current_q
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str | Path, metadata: dict | None = None) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "q_table": dict(self.q_table),
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "initial_q_values": self.initial_q_values,
            "metadata": metadata or {},
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)


def obs_to_state(obs: np.ndarray) -> State:
    """
    Convert Gymnasium observation array into a hashable state tuple.
    Example:
        np.array([2, 17]) -> (2, 17)
    """
    return tuple(int(value) for value in obs)


def print_greedy_rollout_summary(
    agent: QLearningAgent,
    opponent: str,
    n_rounds: int,
    p_noise: float,
    observation_mode: str,
    history_length: int,
    seed: int,
    max_rows: int = 20,
) -> None:
    env = PrisonersDilemmaEnv(
        n_rounds=n_rounds,
        p_noise=p_noise,
        opponent=opponent,
        observation_mode=observation_mode,
        history_length=history_length,
        render_mode=None,
    )

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    obs, info = env.reset(seed=seed)
    state = obs_to_state(obs)
    rows = []
    action_counts = {0: 0, 1: 0}
    total_reward = 0.0
    done = False

    while not done:
        known_state = state in agent.q_table
        action = agent.get_action(state)
        q_values = agent.q_table[state].copy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = obs_to_state(next_obs)

        if len(rows) < max_rows:
            rows.append((env._round_idx, state, action, reward, known_state, q_values))

        action_counts[action] += 1
        total_reward += reward
        state = next_state
        done = terminated or truncated

    agent.epsilon = original_epsilon
    env.close()

    print("\nGreedy rollout summary against training opponent:")
    print(f"Opponent: {opponent}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Cooperations: {action_counts[0]} | Defections: {action_counts[1]}")
    print(f"First {len(rows)} greedy decisions:")
    for round_idx, state, action, reward, known_state, q_values in rows:
        action_name = "C" if action == 0 else "D"
        print(
            f"  round={round_idx:3d} | state={state} | action={action_name} | "
            f"reward={reward:.1f} | known={known_state} | q={q_values}"
        )


def train_q_learning(
    episodes: int = 20000,
    n_rounds: int = 200,
    p_noise: float = 0.0,
    opponent: str = "tft",
    observation_mode: str = "last_n",
    history_length: int = 3,
    seed: int = 42,
    save_path: str = "models/q_agent_tft.pkl",
) -> None:
    """
    Train a Q-learning agent against one fixed opponent.
    """
    env = PrisonersDilemmaEnv(
        n_rounds=n_rounds,
        p_noise=p_noise,
        opponent=opponent,
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

    episode_rewards = []

    for episode in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + episode)
        state = obs_to_state(obs)

        done = False
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)

            next_obs, reward, terminated, truncated, info = env.step(action)
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
                f"Opponent={opponent} | "
                f"Mean reward (last 500): {mean_recent_reward:.3f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()
    agent.save(
        save_path,
        metadata={
            "n_rounds": n_rounds,
            "p_noise": p_noise,
            "opponent": opponent,
            "observation_mode": observation_mode,
            "history_length": history_length,
            "seed": seed,
        },
    )

    print("\nTraining finished.")
    print(f"Saved model to: {save_path}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.3f}")

    print_greedy_rollout_summary(
        agent=agent,
        opponent=opponent,
        n_rounds=n_rounds,
        p_noise=p_noise,
        observation_mode=observation_mode,
        history_length=history_length,
        seed=seed + episodes + 1,
    )


if __name__ == "__main__":
    train_q_learning(
        episodes=20000,
        n_rounds=200,
        p_noise=0.0,
        opponent="alld",
        observation_mode="last_n",
        history_length=3,
        seed=42,
        save_path="models/q_agent_vs_alld.pkl",
    )
