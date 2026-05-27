from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

History = List[Tuple[int, int]]


def _opponent_actions(history: History) -> List[int]:
    return [opponent_action for opponent_action, _ in history]


def _last_opponent_action(history: History) -> int:
    return int(history[-1][0])


class OpponentPolicy:
    """
    History is stored from the policy's perspective:
    (opponent_action, self_action) for each previous round.
    """

    def reset(self) -> None:
        return

    def act(self, history: History) -> int:
        raise NotImplementedError


class AlwaysCooperate(OpponentPolicy):
    def act(self, history: History) -> int:
        return 0


class AlwaysDefect(OpponentPolicy):
    def act(self, history: History) -> int:
        return 1


class RandomStrategy(OpponentPolicy):
    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, history: History) -> int:
        return int(self.rng.random() < 0.5)


class TitForTat(OpponentPolicy):
    def act(self, history: History) -> int:
        if not history:
            return 0
        return _last_opponent_action(history)
    
class SuspiciousTitForTat(OpponentPolicy):
    def act(self, history: History) -> int:
        if not history:
            return 1
        return _last_opponent_action(history)


class TitForTwoTats(OpponentPolicy):
    def act(self, history: History) -> int:
        if len(history) < 2:
            return 0
        return int(all(action == 1 for action in _opponent_actions(history[-2:])))


class HardTFT(OpponentPolicy):
    def act(self, history: History) -> int:
        if not history:
            return 0
        return int(any(action == 1 for action in _opponent_actions(history[-3:])))
    
    
class GenerousTitForTat(OpponentPolicy):
    def __init__(self, p_cooperate_after_defection: float = 0.1, seed: Optional[int] = None) -> None:
        if not (0.0 <= p_cooperate_after_defection <= 1.0):
            raise ValueError("p_cooperate_after_defection must be in [0, 1].")
        self.p_cooperate_after_defection = float(p_cooperate_after_defection)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, history: History) -> int:
        if not history:
            return 0
        if _last_opponent_action(history) == 1:
            return 0 if self.rng.random() < self.p_cooperate_after_defection else 1
        return 0


class Grudger(OpponentPolicy):
    def __init__(self) -> None:
        self.triggered = False

    def reset(self) -> None:
        self.triggered = False

    def act(self, history: History) -> int:
        if self.triggered:
            return 1
        if history and _last_opponent_action(history) == 1:
            self.triggered = True
            return 1
        return 0


class SoftGrudger(OpponentPolicy):
    def __init__(self) -> None:
        self.response_queue: List[int] = []

    def reset(self) -> None:
        self.response_queue = []

    def act(self, history: History) -> int:
        if self.response_queue:
            return self.response_queue.pop(0)
        if history and _last_opponent_action(history) == 1:
            # Standard soft-grudger punishment cycle.
            self.response_queue = [1, 1, 1, 0, 0]
            return 1
        return 0


class HardMajo(OpponentPolicy):
    def act(self, history: History) -> int:
        if not history:
            return 1

        opponent_actions = _opponent_actions(history)
        cooperations = sum(action == 0 for action in opponent_actions)
        defections = len(opponent_actions) - cooperations
        return int(defections >= cooperations)


class SoftMajo(OpponentPolicy):
    def act(self, history: History) -> int:
        if not history:
            return 0

        opponent_actions = _opponent_actions(history)
        cooperations = sum(action == 0 for action in opponent_actions)
        defections = len(opponent_actions) - cooperations
        return int(defections > cooperations)


class Joss(OpponentPolicy):
    def __init__(self, p_cooperate_after_cooperation: float = 0.9, seed: Optional[int] = None) -> None:
        if not (0.0 <= p_cooperate_after_cooperation <= 1.0):
            raise ValueError("p_cooperate_after_cooperation must be in [0, 1].")
        self.p_cooperate_after_cooperation = float(p_cooperate_after_cooperation)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, history: History) -> int:
        if not history:
            return 0
        if _last_opponent_action(history) == 1:
            return 1
        return 0 if self.rng.random() < self.p_cooperate_after_cooperation else 1


class SoftJoss(Joss):
    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(p_cooperate_after_cooperation=0.95, seed=seed)


class Prober(OpponentPolicy):
    """D, C, C; defect forever if opponent played C, C on rounds 2 and 3."""
    def act(self, history: History) -> int:
        opening = [1, 0, 0]
        turn = len(history)
        if turn < len(opening):
            return opening[turn]

        opponent_opening = _opponent_actions(history[:3])
        if opponent_opening[1:] == [0, 0]:
            return 1
        return _last_opponent_action(history)


class Prober2(OpponentPolicy):
    """D, C, C; cooperate forever if opponent played D, C on rounds 2 and 3."""

    def act(self, history: History) -> int:
        opening = [1, 0, 0]
        turn = len(history)
        if turn < len(opening):
            return opening[turn]

        opponent_opening = _opponent_actions(history[:3])
        if opponent_opening[1:] == [1, 0]:
            return 0
        return _last_opponent_action(history)


class Prober3(OpponentPolicy):
    """D, C; defect forever if opponent cooperated on round 2."""

    def act(self, history: History) -> int:
        opening = [1, 0]
        turn = len(history)
        if turn < len(opening):
            return opening[turn]

        if _opponent_actions(history[:2])[1] == 0:
            return 1
        return _last_opponent_action(history)


class HardProber(OpponentPolicy):
    """D, D, C, C; defect forever if opponent cooperated on rounds 2 and 3."""

    def act(self, history: History) -> int:
        opening = [1, 1, 0, 0]
        turn = len(history)
        if turn < len(opening):
            return opening[turn]

        if _opponent_actions(history[:4])[1:3] == [0, 0]:
            return 1
        return _last_opponent_action(history)
    

class CyclerCCD(OpponentPolicy):
    # Cycle through C, C, D regardless of opponent actions.
    def act (self, history: History) -> int:
        cycle = [0, 0, 1]
        return cycle[len(history) % 3]
    
class CyclerDDC(OpponentPolicy):
    # Cycle through D, D, C regardless of opponent actions.
    def act (self, history: History) -> int:
        cycle = [1, 1, 0]
        return cycle[len(history) % 3]



STRATEGY_FAMILIES: Dict[str, List[str]] = {
    "simple": ["allc", "alld", "random"],
    "tit_for_tat": ["tft", "mist", "tftt", "hard_tft"],
    "grudger": ["grudger", "soft_grudger"],
    "majority": ["soft_majo", "hard_majo"],
    "joss": ["joss", "soft_joss"],
    "prober": ["prober", "prober_2", "prober_3", "hard_prober", "naive_prober"],
}

OPPONENT_REGISTRY = {
    "allc": AlwaysCooperate,
    "alld": AlwaysDefect,
    "random": RandomStrategy,
    "tft": TitForTat,
    "mist" : SuspiciousTitForTat,
    "tftt": TitForTwoTats,
    "hard_tft": HardTFT,
    "generous_tft": GenerousTitForTat,
    "grudger": Grudger,
    "soft_grudger": SoftGrudger,
    "soft_majo": SoftMajo,
    "hard_majo": HardMajo,
    "joss": Joss,
    "soft_joss": SoftJoss,
    "prober": Prober,
    "prober_3": Prober3,
    "hard_prober": HardProber,
    "cycler_ccd": CyclerCCD,
    "cycler_ddc": CyclerDDC
}
