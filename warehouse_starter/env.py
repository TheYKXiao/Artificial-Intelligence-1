# warehouse_starter/env.py
from __future__ import annotations
from typing import List, Tuple, Set
from config import Coord, Config

class WarehouseState:
    """
    Grid with shelves S#, stations P#, empties '.'.
    artery[r][c] == True → corridor cell (must be empty '.' or a station).
    NOTE: Coordinates are (row, col) with (0,0) at top-left.
    """
    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.grid = [['.' for _ in range(cols)] for _ in range(rows)]
        self.artery = [[False for _ in range(cols)] for _ in range(rows)]
        self.shelves: List[Coord] = []
        self.stations: List[Coord] = []

    def clone(self) -> "WarehouseState":
        t = WarehouseState(self.rows, self.cols)
        t.grid = [row[:] for row in self.grid]
        t.artery = [row[:] for row in self.artery]
        t.shelves = self.shelves[:]
        t.stations = self.stations[:]
        return t

    def empties_non_artery(self) -> List[Coord]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.grid[r][c] == '.' and not self.artery[r][c]]

    def empties_artery(self) -> List[Coord]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.grid[r][c] == '.' and self.artery[r][c]]

    def swap_cells(self, a: Coord, b: Coord):
        """
        Swap grid contents at a and b; rebuild shelves/stations lists.
        NOTE: This does not enforce artery invariants; call validate_state(...) after
        high-level edits if you need strict checking.
        """
        (ra, ca), (rb, cb) = a, b
        self.grid[ra][ca], self.grid[rb][cb] = self.grid[rb][cb], self.grid[ra][ca]
        # rebuild entity lists
        self.shelves.clear(); self.stations.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                v = self.grid[r][c]
                if isinstance(v, str) and v.startswith('S'):
                    self.shelves.append((r, c))
                elif isinstance(v, str) and v.startswith('P'):
                    self.stations.append((r, c))


def artery_mask(rows: int, cols: int) -> List[List[bool]]:
    """Connected corridors: all odd rows + every 3rd column (c % 3 == 1)."""
    A = [[False for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if (r % 2 == 1) or (c % 3 == 1):
                A[r][c] = True
    return A


def all_artery_cells(A: List[List[bool]]) -> List[Coord]:
    rows, cols = len(A), len(A[0])
    return [(r, c) for r in range(rows) for c in range(cols) if A[r][c]]


def all_non_artery_cells(A: List[List[bool]]) -> List[Coord]:
    rows, cols = len(A), len(A[0])
    return [(r, c) for r in range(rows) for c in range(cols) if not A[r][c]]


def _ensure_unique_within_domain(name: str, idxs: List[int], domain_size: int):
    if not idxs:
        return
    if any(i < 0 or i >= domain_size for i in idxs):
        raise ValueError(f"{name}: index out of range (domain size={domain_size}), got {idxs}")
    if len(set(idxs)) != len(idxs):
        raise ValueError(f"{name}: indices must be unique, got {idxs}")


def build_state_from_choices(cfg: Config, station_idxs: List[int], empty_idxs: List[int]) -> WarehouseState:
    """
    Decode a compact layout specification into a full grid:
    - Stations placed on artery cells by index (station_idxs into row-major artery list).
    - On non-artery cells, keep 'empty_idxs' empty; all others become shelves.
    Enforces:
      * stations on artery cells
      * no shelves on artery cells
    Raises ValueError on malformed inputs.
    """
    A = artery_mask(cfg.rows, cfg.cols)
    artery_cells = all_artery_cells(A)          # deterministic row-major order
    non_artery_cells = all_non_artery_cells(A)

    _ensure_unique_within_domain("station_idxs", station_idxs, len(artery_cells))
    _ensure_unique_within_domain("empty_idxs", empty_idxs, len(non_artery_cells))

    st = WarehouseState(cfg.rows, cfg.cols)
    st.artery = A

    # place stations on artery
    for j, idx in enumerate(station_idxs):
        r, c = artery_cells[idx]
        if st.grid[r][c] != '.':
            raise ValueError(f"station collision at {(r, c)}")
        st.grid[r][c] = f'P{j}'
        st.stations.append((r, c))

    # mark non-artery empties
    keep_empty: Set[Coord] = {non_artery_cells[i] for i in empty_idxs}

    # fill shelves on remaining non-artery cells
    sid = 0
    for r, c in non_artery_cells:
        if (r, c) not in keep_empty:
            st.grid[r][c] = f'S{sid}'
            st.shelves.append((r, c))
            sid += 1

    # Final invariant checks: no shelves on arteries
    for r in range(cfg.rows):
        for c in range(cfg.cols):
            if st.artery[r][c]:
                v = st.grid[r][c]
                if isinstance(v, str) and v.startswith('S'):
                    raise ValueError(f"invalid state: shelf on artery at {(r, c)}")

    return st


def validate_state(state: WarehouseState, strict: bool = True) -> None:
    """
    Sanity-check invariants. Raises ValueError on problems (if strict).
    - No shelves on artery cells.
    - Stations must be on artery cells.
    - All entity coordinates inside bounds.
    - No duplicate station/shelf positions.
    """
    R, C = state.rows, state.cols
    seen: Set[Coord] = set()

    for (r, c) in state.shelves:
        if not (0 <= r < R and 0 <= c < C):
            raise ValueError(f"shelf out of bounds at {(r, c)}")
        if state.artery[r][c]:
            raise ValueError(f"shelf on artery at {(r, c)}")
        if (r, c) in seen:
            raise ValueError(f"duplicate entity at {(r, c)}")
        seen.add((r, c))

    for (r, c) in state.stations:
        if not (0 <= r < R and 0 <= c < C):
            raise ValueError(f"station out of bounds at {(r, c)}")
        if not state.artery[r][c]:
            raise ValueError(f"station not on artery at {(r, c)}")
        if (r, c) in seen:
            raise ValueError(f"duplicate entity at {(r, c)}")
        seen.add((r, c))

    if strict:
        if len(state.stations) == 0:
            raise ValueError("no stations placed")
        if len(state.shelves) == 0:
            raise ValueError("no shelves placed")
