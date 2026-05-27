from __future__ import annotations

from dataclasses import dataclass  
from typing import Dict, List, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from strategies import OpponentPolicy, OPPONENT_REGISTRY

@dataclass
class PDPayoffs:
    R: float = 3.0  # Reward for mutual cooperation
    S: float = 0.0  # Sucker's payoff (cooperate while opponent defects)
    T: float = 5.0  # Temptation to defect (defect while opponent cooperates)
    P: float = 1.0  # Punishment for mutual defection

class PrisonersDilemmaEnv(gym.Env):

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        n_rounds: int = 200,
        p_noise: float = 0.0,
        payoffs: PDPayoffs = PDPayoffs(),
        opponent: str = "tft",
        observation_mode: str = "last_n",
        history_length: int = 3,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if n_rounds <= 0:
            raise ValueError("n_rounds must be > 0")
        if not (0.0 <= p_noise <= 1.0):
            raise ValueError("p_noise must be in [0, 1]")
        if opponent not in OPPONENT_REGISTRY:
            raise ValueError(f"Unknown opponent '{opponent}'. Options: {list(OPPONENT_REGISTRY.keys())}")
        if observation_mode not in {"last_n", "full_history", "last_outcome"}:
            raise ValueError("observation_mode must be 'last_n', 'full_history', or 'last_outcome'")
        if history_length <= 0:
            raise ValueError("history_length must be > 0")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode={render_mode}. Options: {self.metadata['render_modes']}")

        self.n_rounds = int(n_rounds)
        self.p_noise = float(p_noise)
        self.payoffs = payoffs
        self.opponent_name = opponent
        self.observation_mode = observation_mode
        self.history_length = int(history_length)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)  # 0: Cooperate, 1: Defect
        if self.observation_mode == "full_history":
            self.observation_space = spaces.MultiDiscrete(
                [5] * self.n_rounds + [self.n_rounds + 1]
            )
        elif self.observation_mode == "last_n":
            self.observation_space = spaces.MultiDiscrete(
                [5] * self.history_length + [self.n_rounds + 1]
            )
        else:
            self.observation_space = spaces.MultiDiscrete([4, self.n_rounds + 1])

        self._round_idx: int = 0
        self._prev_outcome: int = 0  # Initial CC outcome.
        self._history: List[Tuple[int, int]] = []
        self._opponent: OpponentPolicy = OPPONENT_REGISTRY[self.opponent_name]()

    @staticmethod
    def _encode_outcome(agent_action: int, opp_action: int) -> int:
        # (0,0)->0  (0,1)->1  (1,0)->2  (1,1)->3
        return (agent_action << 1) | opp_action
    
    def _apply_noise(self, intended_action: int) -> int:
        # Action-execution noise.
        if self.np_random.random() < self.p_noise:
            return 1 - int(intended_action)
        return int(intended_action)
    
    def _get_obs(self) -> np.ndarray:
        if self.observation_mode == "last_outcome":
            return np.array([self._prev_outcome, self._round_idx], dtype=np.int64)

        encoded_history = [
            self._encode_outcome(agent_action, opp_action)
            for agent_action, opp_action in self._history
        ]
        unseen_token = 4
        if self.observation_mode == "last_n":
            visible_history = encoded_history[-self.history_length:]
            padded_history = [unseen_token] * (self.history_length - len(visible_history)) + visible_history
        else:
            padded_history = encoded_history + [unseen_token] * (self.n_rounds - len(encoded_history))
        return np.array([*padded_history, self._round_idx], dtype=np.int64)
    
    def _get_info(
        self,
        intended_agent_action: Optional[int] = None,
        intended_opp_action: Optional[int] = None,
        executed_agent_action: Optional[int] = None,
        executed_opp_action: Optional[int] = None,
    ) -> Dict:
        return {
            "round_idx": self._round_idx,
            "opponent": self.opponent_name,
            "p_noise": self.p_noise,
            "observation_mode": self.observation_mode,
            "history_length": self.history_length,
            "intended_agent_action": intended_agent_action,
            "intended_opp_action": intended_opp_action,
            "executed_agent_action": executed_agent_action,
            "executed_opp_action": executed_opp_action,
        }
    
    def _payoff_agent(self, agent_action: int, opp_action: int) -> float:
        # Payoffs are computed from executed actions.
        if agent_action == 0 and opp_action == 0:  # CC
            return self.payoffs.R
        if agent_action == 0 and opp_action == 1:  # CD (agent cooperates, opp defects)
            return self.payoffs.S
        if agent_action == 1 and opp_action == 0:  # DC (agent defects, opp cooperates)
            return self.payoffs.T
        # DD
        return self.payoffs.P
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if options is not None and "opponent" in options:
            opp_name = options["opponent"]
            if opp_name not in OPPONENT_REGISTRY:
                raise ValueError(f"Unknown opponent '{opp_name}'. Options: {list(OPPONENT_REGISTRY.keys())}")
            self.opponent_name = opp_name
            self._opponent = OPPONENT_REGISTRY[self.opponent_name]()

        self._round_idx = 0
        self._prev_outcome = 0
        self._history = []
        self._opponent.reset()

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info
    
    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; must be 0 (C) or 1 (D).")

        intended_agent_action = int(action)
        intended_opp_action = int(self._opponent.act(self._history))

        executed_agent_action = self._apply_noise(intended_agent_action)
        executed_opp_action = self._apply_noise(intended_opp_action)

        reward = float(self._payoff_agent(executed_agent_action, executed_opp_action))

        self._history.append((executed_agent_action, executed_opp_action))
        self._prev_outcome = self._encode_outcome(executed_agent_action, executed_opp_action)
        self._round_idx += 1

        terminated = self._round_idx >= self.n_rounds
        truncated = False

        obs = self._get_obs()
        info = self._get_info(
            intended_agent_action=intended_agent_action,
            intended_opp_action=intended_opp_action,
            executed_agent_action=executed_agent_action,
            executed_opp_action=executed_opp_action,
        )

        if self.render_mode is not None:
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        if self.render_mode == "human":
            print(self._render_text())
            return None
        return None

    def _render_text(self) -> str:
        outcome_map = {0: "CC", 1: "CD", 2: "DC", 3: "DD"}
        last = outcome_map.get(self._prev_outcome, "??")
        return (
            f"IPD round {self._round_idx}/{self.n_rounds} | "
            f"opponent={self.opponent_name} | last_outcome={last} | "
            f"p_noise={self.p_noise:.3f} | observation_mode={self.observation_mode} | "
            f"history_length={self.history_length}"
        )

    def close(self):
        return
