from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict

import numpy as np

from env import PDPayoffs, PrisonersDilemmaEnv
from strategies import OPPONENT_REGISTRY, OpponentPolicy


RESULTS_DIR = Path("results") / "final_round_robin_tournament"
RANKINGS_PATH = RESULTS_DIR / "rankings.csv"
MATCHUPS_PATH = RESULTS_DIR / "matchups.csv"

N_ROUNDS = 200
EPISODES = 20
SEED = 42
PAYOFFS = PDPayoffs()

CLASSICAL_STRATEGIES = [
    "allc",
    "alld",
    "tft",
    "mist",
    "tftt",
    "hard_tft",
    "generous_tft",
    "grudger",
    "soft_grudger",
    "soft_majo",
    "hard_majo",
    "joss",
    "prober",
    "prober_3",
    "hard_prober",
    "cycler_ccd",
    "cycler_ddc",
]

Q_LEARNERS = [
    (
        "q_exp1_alld",
        Path("models/experiment_1/q_agent_vs_alld_seed3.pkl"),
    ),
    (
        "q_exp2_reciprocal",
        Path("models/experiment_2/q_agent_population_reciprocal_seed1.pkl"),
    ),
    (
        "q_exp2_exploitative",
        Path("models/experiment_2/q_agent_population_exploitative_seed1.pkl"),
    ),
    (
        "q_exp2_mixed",
        Path("models/experiment_2/q_agent_population_mixed_seed2.pkl"),
    ),
    (
        "q_exp3_exploitative_clean",
        Path("models/experiment_3/q_agent_population_exploitative_pnoise0p0_seed1.pkl"),
    ),
    (
        "q_exp4_tftt_last1",
        Path("models/experiment_4/q_agent_partial_obs_vs_tftt_last1_seed3.pkl"),
    ),
    (
        "q_exp5bis_improved_grudger",
        Path("models/experiment_5_bis/q_agent_exp5bis_qinit_cooperate_vs_grudger_baseline_seed1.pkl"),
    ),
    (
        "q_exp5bis_grudger_gamma0999_lr05_001",
        Path(
            "models/experiment_5_bis/"
            "q_agent_exp5bis_standard_vs_grudger_"
            "gamma_0999_lr_05_to_001_eps_09_to_001_seed1.pkl"
        ),
    ),
    (
        "q_exp5bis_grudger_gamma099_lr05_001",
        Path(
            "models/experiment_5_bis/"
            "q_agent_exp5bis_standard_vs_grudger_"
            "gamma_099_lr_05_to_001_eps_09_to_0001_seed1.pkl"
        ),
    ),
]


def payoff(action_a: int, action_b: int) -> tuple[float, float]:
    if action_a == 0 and action_b == 0:
        return PAYOFFS.R, PAYOFFS.R
    if action_a == 0 and action_b == 1:
        return PAYOFFS.S, PAYOFFS.T
    if action_a == 1 and action_b == 0:
        return PAYOFFS.T, PAYOFFS.S
    return PAYOFFS.P, PAYOFFS.P


def encode_state(
    own_history: list[tuple[int, int]],
    observation_mode: str,
    history_length: int,
    n_rounds: int,
) -> tuple[int, ...]:
    encoded_history = [
        PrisonersDilemmaEnv._encode_outcome(own_action, opp_action)
        for own_action, opp_action in own_history
    ]
    round_idx = len(own_history)

    if observation_mode == "last_outcome":
        previous_outcome = encoded_history[-1] if encoded_history else 0
        return (previous_outcome, round_idx)

    unseen_token = 4
    if observation_mode == "full_history":
        padded_history = encoded_history + [unseen_token] * (
            n_rounds - len(encoded_history)
        )
    else:
        visible_history = encoded_history[-history_length:]
        padded_history = [unseen_token] * (
            history_length - len(visible_history)
        ) + visible_history

    return tuple([*padded_history, round_idx])


class Player:
    label: str
    participant_type: str

    def reset(self, seed: int) -> None:
        return

    def act(self, own_history: list[tuple[int, int]]) -> int:
        raise NotImplementedError


class StrategyPlayer(Player):
    def __init__(self, strategy_name: str) -> None:
        self.strategy_name = strategy_name
        self.label = strategy_name
        self.participant_type = "classical"
        self.policy: OpponentPolicy | None = None

    def reset(self, seed: int) -> None:
        policy_cls = OPPONENT_REGISTRY[self.strategy_name]
        try:
            self.policy = policy_cls(seed=seed)
        except TypeError:
            self.policy = policy_cls()
        self.policy.reset()

    def act(self, own_history: list[tuple[int, int]]) -> int:
        if self.policy is None:
            raise RuntimeError("StrategyPlayer must be reset before acting.")
        strategy_history = [
            (opp_action, own_action) for own_action, opp_action in own_history
        ]
        return int(self.policy.act(strategy_history))


class QModelPlayer(Player):
    def __init__(self, label: str, model_path: Path) -> None:
        self.label = label
        self.participant_type = "q_learner"
        self.model_path = model_path
        self.metadata: dict = {}
        self.default_q_values = np.zeros(2, dtype=np.float64)
        self.q_table: DefaultDict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: self.default_q_values.copy()
        )
        self._load()

    def _load(self) -> None:
        with self.model_path.open("rb") as f:
            data = pickle.load(f)

        self.metadata = dict(data.get("metadata", {}))
        stored_initial_q_values = data.get("initial_q_values")
        if stored_initial_q_values is not None:
            self.default_q_values = np.array(stored_initial_q_values, dtype=np.float64)
        else:
            self.default_q_values = np.array(
                [
                    float(self.metadata.get("initial_q_value_for_cooperate", 0.0)),
                    float(self.metadata.get("initial_q_value_for_defect", 0.0)),
                ],
                dtype=np.float64,
            )

        self.q_table = defaultdict(lambda: self.default_q_values.copy())
        for state, values in data["q_table"].items():
            self.q_table[tuple(state)] = np.array(values, dtype=np.float64)

    def act(self, own_history: list[tuple[int, int]]) -> int:
        state = encode_state(
            own_history=own_history,
            observation_mode=str(self.metadata.get("observation_mode", "last_n")),
            history_length=int(
                self.metadata.get(
                    "history_length",
                    self.metadata.get("state_memory_length", 3),
                )
            ),
            n_rounds=N_ROUNDS,
        )
        return int(np.argmax(self.q_table[state]))


@dataclass(frozen=True)
class ParticipantSpec:
    label: str
    participant_type: str
    strategy_name: str | None = None
    model_path: Path | None = None

    def make_player(self) -> Player:
        if self.strategy_name is not None:
            return StrategyPlayer(self.strategy_name)
        if self.model_path is None:
            raise ValueError(f"Missing model_path for {self.label}")
        return QModelPlayer(self.label, self.model_path)


def build_participants() -> tuple[list[ParticipantSpec], list[ParticipantSpec]]:
    classical = [
        ParticipantSpec(
            label=f"strategy::{name}",
            participant_type="classical",
            strategy_name=name,
        )
        for name in CLASSICAL_STRATEGIES
    ]

    missing_models = [path for _, path in Q_LEARNERS if not path.exists()]
    if missing_models:
        missing = "\n".join(f"  - {path}" for path in missing_models)
        raise FileNotFoundError(f"Missing Q-learner model file(s):\n{missing}")

    q_learners = [
        ParticipantSpec(
            label=label,
            participant_type="q_learner",
            model_path=model_path,
        )
        for label, model_path in Q_LEARNERS
    ]
    return classical + q_learners, classical


def play_match(
    participant: ParticipantSpec,
    opponent: ParticipantSpec,
    episode_seed: int,
) -> dict:
    player = participant.make_player()
    opponent_player = opponent.make_player()

    episode_rewards: list[float] = []
    cooperation_rates: list[float] = []

    for episode in range(EPISODES):
        player.reset(episode_seed + episode)
        opponent_player.reset(episode_seed + 10_000 + episode)

        player_history: list[tuple[int, int]] = []
        opponent_history: list[tuple[int, int]] = []
        total_reward = 0.0
        cooperation_count = 0

        for _ in range(N_ROUNDS):
            action = player.act(player_history)
            opponent_action = opponent_player.act(opponent_history)

            reward, _ = payoff(action, opponent_action)
            total_reward += reward
            cooperation_count += int(action == 0)

            player_history.append((action, opponent_action))
            opponent_history.append((opponent_action, action))

        episode_rewards.append(total_reward)
        cooperation_rates.append(cooperation_count / N_ROUNDS)

    return {
        "participant": participant.label,
        "participant_type": participant.participant_type,
        "opponent": opponent.label,
        "opponent_type": opponent.participant_type,
        "episodes": EPISODES,
        "n_rounds": N_ROUNDS,
        "mean_reward": float(np.mean(episode_rewards)),
        "cooperation_rate": float(np.mean(cooperation_rates)),
    }


def run_tournament() -> tuple[list[dict], list[dict]]:
    participants, benchmark_opponents = build_participants()
    matchup_rows: list[dict] = []

    for participant_index, participant in enumerate(participants):
        for opponent_index, opponent in enumerate(benchmark_opponents):
            row = play_match(
                participant=participant,
                opponent=opponent,
                episode_seed=SEED + participant_index * 1_000 + opponent_index * 100,
            )
            matchup_rows.append(row)

    rankings = []
    for participant in participants:
        rows = [row for row in matchup_rows if row["participant"] == participant.label]
        rankings.append(
            {
                "participant": participant.label,
                "participant_type": participant.participant_type,
                "mean_reward": float(np.mean([row["mean_reward"] for row in rows])),
                "cooperation_rate": float(
                    np.mean([row["cooperation_rate"] for row in rows])
                ),
            }
        )

    rankings.sort(key=lambda row: row["mean_reward"], reverse=True)
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank

    return rankings, matchup_rows


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rankings(rankings: list[dict]) -> None:
    print("\nFinal round-robin benchmark rankings")
    print(
        f"{'rank':>4}  {'participant':<42}  {'type':<10}  "
        f"{'mean_reward':>11}  {'coop_rate':>9}"
    )
    print("-" * 86)
    for row in rankings:
        print(
            f"{row['rank']:>4}  "
            f"{row['participant']:<42}  "
            f"{row['participant_type']:<10}  "
            f"{row['mean_reward']:>11.3f}  "
            f"{row['cooperation_rate']:>9.3f}"
        )


def main() -> None:
    rankings, matchup_rows = run_tournament()

    save_csv(
        RANKINGS_PATH,
        rankings,
        ["rank", "participant", "participant_type", "mean_reward", "cooperation_rate"],
    )
    save_csv(
        MATCHUPS_PATH,
        matchup_rows,
        [
            "participant",
            "participant_type",
            "opponent",
            "opponent_type",
            "episodes",
            "n_rounds",
            "mean_reward",
            "cooperation_rate",
        ],
    )

    print_rankings(rankings)
    print(f"\nSaved rankings to: {RANKINGS_PATH}")
    print(f"Saved matchups to: {MATCHUPS_PATH}")


if __name__ == "__main__":
    main()
