# warehouse_starter/viz.py
from __future__ import annotations
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from env import WarehouseState

def draw_layout(state: WarehouseState, path: str):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-0.5, state.cols-0.5); ax.set_ylim(-0.5, state.rows-0.5)
    ax.set_xticks(range(state.cols)); ax.set_yticks(range(state.rows)); ax.grid(True, linestyle=':')
    # shelves
    for (r,c) in state.shelves:
        rect = patches.Rectangle((c-0.5, state.rows-1-r-0.5), 1,1, linewidth=0, facecolor='0.6')
        ax.add_patch(rect)
    # empty corridors (outline)
    for r in range(state.rows):
        for c in range(state.cols):
            if state.artery[r][c] and state.grid[r][c]=='.':
                rect = patches.Rectangle((c-0.5, state.rows-1-r-0.5), 1,1, linewidth=0.3, edgecolor='0.8', facecolor='none')
                ax.add_patch(rect)
    # stations
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
    for j,(r,c) in enumerate(state.stations):
        circ = patches.Circle((c, state.rows-1-r), 0.33, facecolor=colors[j%len(colors)])
        ax.add_patch(circ); ax.text(c, state.rows-1-r, f'P{j}', ha='center', va='center', color='white', fontsize=9)
    ax.set_title('Warehouse Layout (gray=shelves, circles=stations)')
    plt.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True); plt.savefig(path, dpi=150); plt.close()

def draw_heatmap(heat: List[List[int]], path: str):
    import numpy as np
    arr = np.array(heat)
    plt.figure(figsize=(6,6))
    plt.imshow(arr[::-1,:], interpolation='nearest')
    plt.colorbar(label='Traversals'); plt.title('Traffic Heatmap')
    plt.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True); plt.savefig(path, dpi=150); plt.close()

def bar_station(per_station: Dict[str, List[int]], path: str):
    names = sorted(per_station.keys())
    vals = [sum(per_station[k])/max(1,len(per_station[k])) for k in names]
    plt.figure(figsize=(5,3.5)); plt.bar(names, vals)
    plt.xlabel('Station'); plt.ylabel('Avg fulfillment time'); plt.title('Per-Station Averages')
    plt.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True); plt.savefig(path, dpi=150); plt.close()
