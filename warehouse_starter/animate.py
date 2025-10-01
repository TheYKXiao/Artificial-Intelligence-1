# warehouse_starter/animate.py
from __future__ import annotations
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from config import Coord
from env import WarehouseState

def animate_paths(state: WarehouseState,
                  paths: List[List[Coord]],
                  picks_per_order: List[List[Tuple[int, Coord]]],
                  out_gif: str = "out/traffic.gif",
                  sec_per_step: float = 1.0,
                  stagger: int = 5,
                  dot_size: int = 120,
                  show_live: bool = False):
    if not paths:
        print("[anim] No paths to animate; skipping GIF.")
        return
    base_T = max((len(p) for p in paths), default=0)
    if base_T == 0:
        print("[anim] All orders infeasible; skipping GIF.")
        return

    # Normalize path lengths to base_T
    norm_paths = [p + [p[-1]] * (base_T - len(p)) if p else [(None, None)] * base_T for p in paths]
    n = len(norm_paths)
    T = base_T + stagger * (n - 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, state.cols - 0.5)
    ax.set_ylim(-0.5, state.rows - 0.5)
    ax.set_xticks(range(state.cols))
    ax.set_yticks(range(state.rows))
    ax.grid(True, linestyle=':')
    ax.set_aspect('equal', adjustable='box')

    # Background
    for (r, c) in state.shelves:
        rect = patches.Rectangle((c - 0.5, state.rows - 1 - r - 0.5), 1, 1, linewidth=0, facecolor='0.90', zorder=1)
        ax.add_patch(rect)
    for r in range(state.rows):
        for c in range(state.cols):
            if state.artery[r][c] and state.grid[r][c] == '.':
                rect = patches.Rectangle((c - 0.5, state.rows - 1 - r - 0.5), 1, 1,
                                         linewidth=0.2, edgecolor='0.75', facecolor='none', zorder=1)
                ax.add_patch(rect)

    # Color cycle
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan']

    # Draw stations
    for j, (r, c) in enumerate(state.stations):
        col = colors[j % len(colors)]
        circ = patches.Circle((c, state.rows - 1 - r), 0.22, facecolor=col, zorder=2)
        ax.add_patch(circ)
        ax.text(c, state.rows - 1 - r, f'P{j}', ha='center', va='center', color='white', fontsize=8, zorder=3)

    trails = []
    heads = []
    item_boxes: List[List[patches.Rectangle]] = []
    item_arrivals: List[List[int]] = []
    order_colors: List[str] = []  # color per order

    # One trail/head and a set of boxes per order
    for i in range(n):
        col = colors[i % len(colors)]        # <-- order color
        order_colors.append(col)

        ln, = ax.plot([], [], '-', linewidth=1.4, alpha=0.55, color=col, zorder=4)
        hd, = ax.plot([], [], 'o', markersize=max(4, int(dot_size ** 0.5)),
                      markerfacecolor=col, markeredgecolor='k', markeredgewidth=0.4, zorder=6)
        trails.append(ln)
        heads.append(hd)

        boxes_for_order: List[patches.Rectangle] = []
        arrivals_for_order: List[int] = []
        picks_for_i = picks_per_order[i] if i < len(picks_per_order) else []
        for (arr_k, item) in picks_for_i:
            rr, cc = item
            # Box matches the robot's color
            box = patches.Rectangle((cc - 0.35, state.rows - 1 - rr - 0.35), 0.7, 0.7,
                                    linewidth=1.0, edgecolor='k', facecolor=col,
                                    alpha=0.0, zorder=5)
            ax.add_patch(box)
            boxes_for_order.append(box)
            arrivals_for_order.append(arr_k)

        item_boxes.append(boxes_for_order)
        item_arrivals.append(arrivals_for_order)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left', zorder=7)
    ax.set_title('Traffic Animation')

    def init():
        for ln, hd in zip(trails, heads):
            ln.set_data([], [])
            hd.set_data([], [])
        time_text.set_text("")
        return trails + heads + [time_text]

    def update(t):
        for i, p in enumerate(norm_paths):
            k = t - i * stagger
            col = order_colors[i]  # color for this order

            # Trail + head
            if k < 0:
                trails[i].set_data([], [])
                heads[i].set_data([], [])
            else:
                k = min(k, base_T - 1)
                r, c = p[k]
                if r is None or c is None:
                    trails[i].set_data([], [])
                    heads[i].set_data([], [])
                else:
                    xs, ys = [], []
                    for j in range(k + 1):
                        rr, cc = p[j]
                        if rr is None or cc is None:
                            continue
                        xs.append(cc)
                        ys.append(state.rows - 1 - rr)
                    trails[i].set_data(xs, ys)
                    heads[i].set_data([c], [state.rows - 1 - r])

            # Item boxes (match color with robot)
            for box, arr in zip(item_boxes[i], item_arrivals[i]):
                if k < 0:
                    # order not started
                    box.set_alpha(0.0)
                    box.set_hatch(None)
                    box.set_facecolor(col)
                elif 0 <= k < arr:
                    # waiting – show in the order color
                    box.set_facecolor(col)
                    box.set_alpha(0.95)
                    box.set_hatch(None)
                elif k == arr:
                    # at pickup – emphasize (hatch) but keep the same color for linkage
                    box.set_facecolor(col)
                    box.set_alpha(1.0)
                    box.set_hatch('xx')
                else:
                    # picked – hide
                    box.set_alpha(0.0)
                    box.set_hatch(None)

        time_text.set_text(f"t = {t * sec_per_step:.1f} s")
        artists = trails + heads + [time_text]
        for lst in item_boxes:
            artists += lst
        return artists

    interval_ms = max(1, int(round(sec_per_step * 1000)))
    ani = animation.FuncAnimation(fig, update, init_func=init, frames=T, blit=True, interval=interval_ms)

    out_dir = os.path.dirname(out_gif) or '.'
    os.makedirs(out_dir, exist_ok=True)

    fps = max(1, int(round(1.0 / max(1e-6, sec_per_step))))
    ani.save(out_gif, writer=animation.PillowWriter(fps=fps))
    if show_live:
        plt.show()
    plt.close(fig)
