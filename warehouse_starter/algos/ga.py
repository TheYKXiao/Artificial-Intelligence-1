#The GA algorithm was done with the assistance of AI(ChatGPT), the extend is minimal ~ moderate. 
#The use of AI has been labeled.
# ga.py - skeleton for genetic algorithm
from __future__ import annotations
from config import Config
from typing import Tuple, Any, List
import random
from env import (
<<<<<<< HEAD
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
=======
    WarehouseState, artery_mask, all_artery_cells, all_non_artery_cells, 
    build_state_from_choices
)
from sim import evaluate




def genetic_algorithm(cfg: Config, *, pop_size: int, generations: int, seed: int) -> Tuple[Any, float]:
    """Implement a GA and return (best_state, best_score).
    This file is intentionally a minimal stub so students can implement their code.
    """

    rng = random.Random(seed)
    population:List[WarehouseState] = [_random_layout(cfg, rng) for _ in range(pop_size)]

    best_state = min(population, key=lambda s: evaluate(s, cfg,seed)[0])
    best_score = evaluate(best_state, cfg, seed)[0]
    no_improve = 0
    mutation = 0.05
    patience = 20

    for _ in range(generations):
        weights =  _weight_calculate(population, cfg, seed)

        population2: List[WarehouseState] = []
        
        while len(population2) < pop_size:
            parent1 = rng.choices(population, weights=weights, k=1)[0]
            parent2 = rng.choices(population, weights=weights, k=1)[0]

            child = produce(parent1, parent2,cfg, rng)

            if rng.random() < mutation:
                child = mutate(child, cfg, rng)

            population2.append(child)

        population = population2
        
        gen_best = min(population, key=lambda s: evaluate(s, cfg,seed)[0])
        gen_best_score = evaluate(gen_best, cfg, seed)[0]
        if gen_best_score < best_score:
            best_state, best_score = gen_best.clone(), gen_best_score
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_state, best_score

#helper function to define the fitness score
def _fitness_calculate(state: WarehouseState, cfg: Config, seed:int) -> float: 
    score, *_ = evaluate(state, cfg, seed)
    score = float(score)
    return 1.0 / (1.0 + score) #return the inverse of the evaluation value


#helper function to define the weight
def _weight_calculate(population, cfg, seed) -> list[float]:
    fitness = [_fitness_calculate(i,cfg,seed) for i in population] 
    return [i / sum(fitness) for i in fitness]


#helper function to define the produce processing (This helper function was assisted with AI)
def produce(parent1: WarehouseState, parent2: WarehouseState, cfg: Config, rng: random.Random) -> WarehouseState:
    A = artery_mask(cfg.rows, cfg.cols)
    artery_cells     = all_artery_cells(A)
    non_artery_cells = all_non_artery_cells(A)

    p1_st = list(parent1.stations)
    p1_em = list(parent1.empties_non_artery())
    p2_st = list(parent2.stations)
    p2_em = list(parent2.empties_non_artery())

    g1 = p1_st + p1_em
    g2 = p2_st + p2_em
    n  = len(g1)

    c = rng.randint(1, n - 1)
    child_genome = g1[:c] + g2[c:]

    raw_st = child_genome[:cfg.n_stations]
    raw_em = child_genome[cfg.n_stations:]

    def dedup_preserve_order(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out, seen

    st_list, used_st = dedup_preserve_order(raw_st)
    em_list, used_em = dedup_preserve_order(raw_em)

    avail_st = [c for c in artery_cells if c not in used_st]
    avail_em = [c for c in non_artery_cells if c not in used_em]

    while len(st_list) < cfg.n_stations and avail_st:
        st_list.append(avail_st.pop(rng.randrange(len(avail_st))))
    while len(em_list) < cfg.target_non_artery_empties and avail_em:
        em_list.append(avail_em.pop(rng.randrange(len(avail_em))))

    st_list.sort()
    em_list.sort()

    artery_index = {cell: i for i, cell in enumerate(artery_cells)}
    non_artery_index = {cell: i for i, cell in enumerate(non_artery_cells)}

    station_idxs = sorted(artery_index[c] for c in st_list)
    empty_idxs   = sorted(non_artery_index[c] for c in em_list)

    return build_state_from_choices(cfg, station_idxs, empty_idxs)

#helper function to generate a initial state
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


#helper function to define the state
def mutate(state: WarehouseState, cfg: Config, rng: random.Random) -> WarehouseState:
    s = state.clone()
    move_type = rng.choice(["shelf_swap", "station_move"])

    empties_non   = s.empties_non_artery()
    empties_art   = s.empties_artery()
    shelves       = s.shelves[:]
    stations      = s.stations[:]


>>>>>>> 85800128cbe40559eb98be5f8df806418e8beda7
    if move_type == "shelf_swap" and shelves and empties_non:
        sh  = rng.choice(shelves)
        emp = rng.choice(empties_non)
        s.swap_cells(sh, emp)
<<<<<<< HEAD
=======

>>>>>>> 85800128cbe40559eb98be5f8df806418e8beda7
    elif move_type == "station_move" and stations and empties_art:
        st  = rng.choice(stations)
        emp = rng.choice(empties_art)
        s.swap_cells(st, emp)
<<<<<<< HEAD
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
=======

    return s
>>>>>>> 85800128cbe40559eb98be5f8df806418e8beda7
