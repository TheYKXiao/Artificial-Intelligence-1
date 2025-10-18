# ga.py - skeleton for genetic algorithm
from __future__ import annotations
from config import Config
from typing import Tuple, Any, List
import random
from env import (
    WarehouseState, artery_mask, all_artery_cells, all_non_artery_cells,
    build_state_from_choices
)
from sim import evaluate


def genetic_algorithm(cfg: Config, *, pop_size: int, generations: int, seed: int) -> Tuple[Any, float]:
    # """Implement a GA and return (best_state, best_score).

    # This file is intentionally a minimal stub so students can implement their code.
    # """
    # raise NotImplementedError("genetic_algorithm not implemented")
    rng = random.Random(seed)

    # Initial population
    population: List[WarehouseState] = [_random_layout(cfg, rng) for _ in range(pop_size)]

    # weighted by
    weights = [_weighted_by(cfg, ind, seed) for ind in population]

    best_idx   = min(range(len(population)), key=lambda i: weights[i])
    best_state = population[best_idx].clone()
    best_score = weights[best_idx]

    for _ in range(generations):
        new_pop: List[WarehouseState] = []
        for _ in range(len(population)):
            p1 = _select(population, weights, rng)
            p2 = _select(population, weights, rng)
            child = _crossover(cfg, p1, p2, rng)
            child = _mutate(cfg, child, rng, p_mut=0.2)
            new_pop.append(child)

        population = new_pop
        weights = [_weighted_by(cfg, ind, seed) for ind in population]

        gen_best = min(range(len(population)), key=lambda i: weights[i])
        if weights[gen_best] < best_score:
            best_score = weights[gen_best]
            best_state = population[gen_best].clone()

    return best_state, float(best_score)


def _domain(cfg: Config):
    A = artery_mask(cfg.rows, cfg.cols)
    artery_cells = all_artery_cells(A)
    non_artery_cells = all_non_artery_cells(A)
    return A, artery_cells, non_artery_cells

def _indices_from_state(st: WarehouseState, cfg: Config) -> Tuple[List[int], List[int]]:
    A, artery_cells, non_artery_cells = _domain(cfg)
    idx_art = {coord: i for i, coord in enumerate(artery_cells)}
    idx_non = {coord: i for i, coord in enumerate(non_artery_cells)}

    s_idxs = sorted(idx_art[xy] for xy in st.stations)
    empties_non = st.empties_non_artery()
    e_idxs = sorted(idx_non[xy] for xy in empties_non)
    return s_idxs, e_idxs

def _random_layout(cfg: Config, rng: random.Random) -> WarehouseState:
    A, artery_cells, non_artery_cells = _domain(cfg)
    s = sorted(rng.sample(range(len(artery_cells)), cfg.n_stations))
    E = min(cfg.target_non_artery_empties, len(non_artery_cells))
    e = sorted(rng.sample(range(len(non_artery_cells)), E))
    return build_state_from_choices(cfg, s, e)

def _random_neighbor(state: WarehouseState, cfg: Config, rng: random.Random) -> WarehouseState:
    s = state.clone()
    empties_non = s.empties_non_artery()
    empties_art = s.empties_artery()
    shelves     = s.shelves[:]
    stations    = s.stations[:]

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


def _weighted_by(cfg: Config, st: WarehouseState, seed: int) -> float:
    w, *_ = evaluate(st, cfg, seed)
    return float(w)

def _select(pop: List[WarehouseState], fits: List[float], rng: random.Random) -> WarehouseState:
    # WEIGHTED-RANDOM-CHOICES
    max_fit = max(fits)
    eps = 1e-12
    weights = [(max_fit - f) + eps for f in fits]

    parent = rng.choices(population=pop, weights=weights, k=1)[0]
    return parent.clone()

def _crossover(cfg: Config, A: WarehouseState, B: WarehouseState, rng: random.Random) -> WarehouseState:
    _, artery_cells, non_artery_cells = _domain(cfg)
    sA, eA = _indices_from_state(A, cfg)
    sB, eB = _indices_from_state(B, cfg)

    s_union = list(set(sA) | set(sB))
    e_union = list(set(eA) | set(eB))

    s_need = max(0, cfg.n_stations - len(s_union))
    if s_need > 0:
        pool = [i for i in range(len(artery_cells)) if i not in s_union]
        s_union += rng.sample(pool, s_need)

    E = min(cfg.target_non_artery_empties, len(non_artery_cells))
    e_need = max(0, E - len(e_union))
    if e_need > 0:
        pool = [i for i in range(len(non_artery_cells)) if i not in e_union]
        take = min(e_need, len(pool))
        e_union += rng.sample(pool, take)

    s_child = sorted(rng.sample(s_union, cfg.n_stations))
    e_child = sorted(rng.sample(e_union, E))

    return build_state_from_choices(cfg, s_child, e_child)

def _mutate(cfg: Config, st: WarehouseState, rng: random.Random, p_mut: float = 0.2) -> WarehouseState:
    if rng.random() < p_mut:
        return _random_neighbor(st, cfg, rng)
    return st.clone()