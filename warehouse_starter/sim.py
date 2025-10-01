# warehouse_starter/sim.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from collections import deque, defaultdict
import random
from config import Coord, NEI, Config
from env import WarehouseState

# -------- Orders (Zipf) --------
class OrderSampler:
    """Samples item IDs with Zipf-like popularity (few very popular items)."""
    def __init__(self, n_shelves: int, k: int, alpha: float, rng: random.Random):
        self.ids = list(range(n_shelves))
        self.k = k
        self.rng = rng
        ranks = list(range(1, n_shelves+1))
        weights = [1.0/(r**alpha) for r in ranks]
        Z = sum(weights)
        self.probs = [w/Z for w in weights]
    def sample(self) -> List[int]:
        return [self.rng.choices(self.ids, weights=self.probs, k=1)[0] for _ in range(self.k)]

# -------- Geometry + BFS --------
def passable(cell: str) -> bool:
    return cell == '.' or cell.startswith('P')

def dir_allowed(r0: int, c0: int, r1: int, c1: int, cfg: Config) -> bool:
    if not cfg.ge_one_way:
        return True
    dr, dc = r1 - r0, c1 - c0
    if cfg.ge_row_dir == "even_right":
        if r0 % 2 == 0 and dc < 0: return False
        if r0 % 2 == 1 and dc > 0: return False
    elif cfg.ge_row_dir == "even_left":
        if r0 % 2 == 0 and dc > 0: return False
        if r0 % 2 == 1 and dc < 0: return False
    if cfg.ge_col_dir == "up_only" and dr > 0: return False
    if cfg.ge_col_dir == "down_only" and dr < 0: return False
    if cfg.ge_col_dir == "alt_up_down":
        if c0 % 2 == 0 and dr > 0: return False
        if c0 % 2 == 1 and dr < 0: return False
    return True

def bfs(grid: List[List[str]], start: Coord, goal: Coord, cfg: Config) -> Optional[List[Coord]]:
    """Shortest path (4-neighborhood). We *allow* stepping onto the goal even if it's a shelf."""
    R, C = len(grid), len(grid[0])
    q = deque([start]); prev = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal: break
        for dr, dc in NEI:
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C and (rr,cc) not in prev:
                if ((rr,cc) == goal) or (passable(grid[rr][cc]) and dir_allowed(r,c,rr,cc,cfg)):
                    prev[(rr, cc)] = (r, c); q.append((rr, cc))
    if goal not in prev: return None
    path = []; cur = goal
    while cur is not None:
        path.append(cur); cur = prev[cur]
    path.reverse(); return path

def build_tour(grid: List[List[str]], station: Coord, items: List[Coord], cfg: Config,
               track_picks: bool = False):
    """Greedy nearest-neighbor tour: station → items → station."""
    cur = station; remaining = items[:]
    path: List[Coord] = [cur]; picks: List[Tuple[int, Coord]] = []
    while remaining:
        best = None; bestp = None
        for it in remaining:
            p = bfs(grid, cur, it, cfg)
            if p is None: return None
            if bestp is None or len(p) < len(bestp):
                best, bestp = it, p
        path += bestp[1:]
        if track_picks: picks.append((len(path)-1, best))
        cur = best; remaining.remove(best)
    back = bfs(grid, cur, station, cfg)
    if back is None: return None
    path += back[1:]
    return (path, picks) if track_picks else path

# -------- Evaluation / Simulation --------
def evaluate(state: WarehouseState, cfg: Config, seed: int):
    """Compute score and metrics over cfg.orders (no animation)."""
    return _simulate(state, cfg, seed, collect_paths=False)

def simulate_with_paths(state: WarehouseState, cfg: Config, seed: int, limit_orders: int):
    """Short run (subset of orders) that returns paths + pick events for animation."""
    return _simulate(state, cfg, seed, collect_paths=True, limit_orders=limit_orders)

def _simulate(state: WarehouseState, cfg: Config, seed: int, collect_paths: bool, limit_orders: int | None = None):
    rng = random.Random(seed)
    n_shelves = len(state.shelves)
    sampler = OrderSampler(n_shelves, cfg.items_per_order, cfg.zipf_alpha, rng)
    id2coord = {i: state.shelves[i] for i in range(n_shelves)}

    per_station_times = defaultdict(list)
    heat = [[0 for _ in range(state.cols)] for _ in range(state.rows)]
    paths: List[List[Coord]] = []
    picks_per_order: List[List[Tuple[int, Coord]]] = []

    total_dist = 0; penalty = 0
    total_orders = cfg.orders if not collect_paths else min(cfg.orders, limit_orders or cfg.orders)

    for k in range(total_orders):
        sid = k % max(1, len(state.stations))
        station = state.stations[sid]
        items = sampler.sample()
        coords = [id2coord[i] for i in set(items)]

        res = build_tour(state.grid, station, coords, cfg, track_picks=collect_paths)
        if res is None:
            penalty += 500
            if collect_paths: paths.append([]); picks_per_order.append([])
            continue

        if collect_paths:
            p, pe = res; picks_per_order.append(pe); paths.append(p)
        else:
            p = res  # type: ignore

        dist = len(p) - 1; total_dist += dist
        per_station_times[f"P{sid}"].append(dist)
        for (r,c) in p: heat[r][c] += 1

    # If we animated a subset, finish heat/metrics for the rest quickly (no paths/picks)
    if collect_paths and total_orders < cfg.orders:
        for k in range(total_orders, cfg.orders):
            sid = k % max(1, len(state.stations)); station = state.stations[sid]
            items = sampler.sample(); coords = [id2coord[i] for i in set(items)]
            p = build_tour(state.grid, station, coords, cfg, track_picks=False)
            if p is None: penalty += 500; continue
            dist = len(p) - 1; total_dist += dist
            per_station_times[f"P{sid}"].append(dist)
            for (r,c) in p: heat[r][c] += 1

    # ------------------------------------------------------------------
    #   STUDENT TODO: DEFINE YOUR OBJECTIVE HERE (smaller score is better)
    # ------------------------------------------------------------------
    # You now have the raw signals:
    #   - total_dist        (int): sum of all robot steps across orders
    #   - heat[r][c]        (int): how many times each cell was traversed
    #   - per_station_times (dict: station -> list of tour lengths)
    #   - penalty           (int): infeasible orders, path failures, etc.
    #   - cfg               (Config): rows/cols/orders/items_per_order/seed/...
    #
    # Your task: combine these into a single scalar score.
    # You choose the components and their weights; explain your choice in the report.
    #
    # Suggested components (choose and justify):
    #   J1 (Distance):    average distance per order. 
    #                     (Tip: normalize by number of orders so it’s comparable across runs.)
    #   J2 (Congestion):  how much paths overlap. 
    #                     (Options: count overlaps, squared overlaps, or top-k “hot” cells.)
    #   J3 (Fairness):    how evenly stations are loaded.
    #                     (Options: variance, max–min spread, or a Gini-like measure.)
    #   Penalty:          add a large cost for any infeasible order/path.
    #
    # Combine them with weights you pick (w1, w2, w3, etc.). Keep terms on similar scales
    # via normalization so one term doesn’t dominate by accident.


    # Distance, average distance per order
    J1 = total_dist / cfg.orders

    # Congestion, how much paths overlap (squared overlaps)
    overlap_sqo = sum((h - 1) ** 2 for r in heat for h in r if h > 1)
    J2 = overlap_sqo / cfg.orders

    # Fairness, how evenly stations are loaded
    station_means = [sum(t) / len(t) for t in per_station_times.values() if t]
    if station_means:
        mu = sum(station_means) / len(station_means)
        sigma = (sum((x - mu) ** 2 for x in station_means) / len(station_means)) ** 0.5
        J3 = sigma / max(1e-6, mu)
    else:
        J3 = 0.0

    score = cfg.w1 * J1 + cfg.w2 * J2 + cfg.w3 * J3 + penalty

    # score = 0.0
    # metrics = {
    #     "note": "placeholder objective (implement in sim.py TODO block)",
    #     "total_distance": float(total_dist),
    #     "penalty": float(penalty),
    #     "orders": float(cfg.orders),
    # }

    metrics = {
        "note": "placeholder objective (implement in sim.py TODO block)",
        "total_distance": float(total_dist),
        "penalty": float(penalty),
        "orders": float(cfg.orders),
        "J1": float(J1),
        "J2": float(J2),
        "J3": float(J3),
        "penalty": float(penalty),
    }



    if collect_paths:
        return score, metrics, per_station_times, heat, paths, picks_per_order
    else:
        return score, metrics, per_station_times, heat, []
