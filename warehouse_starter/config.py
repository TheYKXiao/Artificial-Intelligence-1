# warehouse_starter/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List

# Coordinates are (row, col) with origin at the top-left of the grid.
Coord = Tuple[int, int]
NEI: List[Coord] = [(1,0), (-1,0), (0,1), (0,-1)]

@dataclass
class Config:
    # Grid
    rows: int = 8
    cols: int = 8

    # Layout counts (we place stations on arteries; non-arteries become shelves except a small empty set)
    n_stations: int = 3
    target_non_artery_empties: int = 6

    # Workload
    orders: int = 120
    items_per_order: int = 2
    zipf_alpha: float = 1.2  # >= 1.0 means head-heavy (few popular items)

    # Objective weights
    w1: float = 0.0   # travel
    w2: float = 0.0  # congestion
    w3: float = 0.0  # fairness

    # Randomness
    seed: int = 42

    # Optional one-way aisle (GE knobs — off by default)
    ge_one_way: bool = False
    ge_row_dir: str = "off"   # off|even_right|even_left
    ge_col_dir: str = "off"   # off|up_only|down_only|alt_up_down
