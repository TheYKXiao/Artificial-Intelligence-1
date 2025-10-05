# sa.py - skeleton for simulated annealing algorithm
from __future__ import annotations
from config import Config
from typing import Tuple, Any, List
import random
import math
from env import (
    WarehouseState, artery_mask, all_artery_cells, all_non_artery_cells,
    build_state_from_choices
)
from sim import evaluate


def simulated_annealing(cfg: Config, *, steps: int, T0: float, alpha: float, seed: int) -> Tuple[Any, float, list]:
    # """Implement simulated annealing and return (best_state, best_score, score_curve).

    # This file is intentionally a minimal stub so students can implement their code.
    # """
    # raise NotImplementedError("simulated_annealing not implemented")

    rng = random.Random(seed)
    best_state  = None
    best_score  = float("inf")
    score_curve: List[float] = []

    current = _random_layout(cfg, rng)  # Initialize
    cur_score, *_ = evaluate(current, cfg, seed)

    best_state  = current.clone()
    best_score  = cur_score

    score_curve: List[float] = [best_score]
    T = T0

    for step in range(1, steps + 1):
        # Schedule: cools down
        T = T0 * (alpha ** step)
        if T < 1e-6:  # equal to T==0
            break

        # random_successor
        neighbor = _random_neighbor(current, cfg, rng)
        neigh_score, *_ = evaluate(neighbor, cfg, seed)

        delta = cur_score - neigh_score

        if delta > 0: # The smaller the neighbor score, the better. 
            current, cur_score = neighbor, neigh_score
        elif rng.random() < math.exp(delta / T):
            current, cur_score = neighbor, neigh_score

        # Update
        if cur_score < best_score:
            best_state, best_score = current.clone(), cur_score

        score_curve.append(best_score)

    return best_state, best_score, score_curve


def _random_layout(cfg: Config, rng: random.Random) -> WarehouseState:
    # Generate a random feasible layout 
    A = artery_mask(cfg.rows, cfg.cols)
    artery_cells     = all_artery_cells(A)
    non_artery_cells = all_non_artery_cells(A)

    station_idxs = sorted(rng.sample(range(len(artery_cells)), cfg.n_stations))
    empty_idxs   = sorted(
        rng.sample(range(len(non_artery_cells)), cfg.target_non_artery_empties)
    )
    return build_state_from_choices(cfg, station_idxs, empty_idxs)


def _random_neighbor(state: WarehouseState, cfg: Config, rng: random.Random) -> WarehouseState:
    # Produce a neighbor by swapping 
    s = state.clone()

    empties_non   = s.empties_non_artery()
    empties_art   = s.empties_artery()
    shelves       = s.shelves[:]
    stations      = s.stations[:]

    move_type = rng.choice(["shelf_swap", "station_move"])

    if move_type == "shelf_swap" and shelves and empties_non:
        sh  = rng.choice(shelves)
        emp = rng.choice(empties_non)
        s.swap_cells(sh, emp)

    elif move_type == "station_move" and stations and empties_art:
        st  = rng.choice(stations)
        emp = rng.choice(empties_art)
        s.swap_cells(st, emp)

    return s