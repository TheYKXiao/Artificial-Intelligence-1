# sa.py - skeleton for simulated annealing algorithm
from __future__ import annotations
from config import Config
from typing import Tuple, Any


def simulated_annealing(cfg: Config, *, steps: int, T0: float, alpha: float, seed: int) -> Tuple[Any, float, list]:
    """Implement simulated annealing and return (best_state, best_score, score_curve).

    This file is intentionally a minimal stub so students can implement their code.
    """
    raise NotImplementedError("simulated_annealing not implemented")
