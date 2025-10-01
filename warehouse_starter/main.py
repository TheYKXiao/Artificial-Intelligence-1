# warehouse_starter/main.py
from __future__ import annotations
import argparse, json, os, random
from typing import List, Tuple
from config import Config
from env import build_state_from_choices, artery_mask, all_artery_cells, all_non_artery_cells
from sim import evaluate, simulate_with_paths
from viz import draw_layout, draw_heatmap, bar_station
from animate import animate_paths

# Try to load student algorithm modules; it's OK if they aren't ready yet.
try:
    from algos.hill import hill_climb
except Exception:
    hill_climb = None  # type: ignore

try:
    from algos.sa import simulated_annealing
except Exception:
    simulated_annealing = None  # type: ignore

try:
    from algos.ga import genetic_algorithm
except Exception:
    genetic_algorithm = None  # type: ignore


def random_baseline_state(cfg: Config, seed: int):
    """Creates a random but feasible layout (stations on arteries, sparse empties)."""
    rng = random.Random(seed)
    A = artery_mask(cfg.rows, cfg.cols)
    artery_cells = all_artery_cells(A)
    non_artery_cells = all_non_artery_cells(A)

    station_idxs = sorted(rng.sample(range(len(artery_cells)), cfg.n_stations))
    E = cfg.target_non_artery_empties
    empty_idxs = sorted(rng.sample(range(len(non_artery_cells)), E))
    return build_state_from_choices(cfg, station_idxs, empty_idxs)


def parse_args():
    p = argparse.ArgumentParser(description='Warehouse Layout – Starter (environment + animation).')
    p.add_argument('--algo', choices=['none','hc','sa','ga','all'], default='none',
                   help='none = baseline/visualize. Implement hc/sa/ga in algos/*.py.')
    # Search params (students can change these)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--restarts', type=int, default=4)
    p.add_argument('--T0', type=float, default=2.0)
    p.add_argument('--alpha', type=float, default=0.995)
    p.add_argument('--pop', type=int, default=30)
    p.add_argument('--gens', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)

    # Outputs (you can override paths; folders are created automatically)
    p.add_argument('--out-layout', type=str, default='out/layout.png')
    p.add_argument('--out-heat', type=str, default='out/heat.png')
    p.add_argument('--out-bars', type=str, default='out/station_bars.png')
    p.add_argument('--out-json', type=str, default='out/summary.json')

    # Animation
    p.add_argument('--animate', action='store_true')
    p.add_argument('--out-gif', type=str, default='out/traffic.gif')
    p.add_argument('--anim-orders', type=int, default=25)
    p.add_argument('--sec-per-step', type=float, default=1.0)
    p.add_argument('--stagger', type=int, default=5)
    p.add_argument('--dot-size', type=int, default=120)
    p.add_argument('--show-live', action='store_true')

    # Optional one-way aisle toggles
    p.add_argument('--ge-one-way', action='store_true')
    p.add_argument('--ge-row-dir', type=str, default='off', choices=['off','even_right','even_left'])
    p.add_argument('--ge-col-dir', type=str, default='off', choices=['off','up_only','down_only','alt_up_down'])
    return p.parse_args()


def _ensure_parent(path: str):
    """Create the parent directory for a file path if needed."""
    parent = os.path.dirname(path) or '.'
    os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    cfg = Config(seed=args.seed,
                 ge_one_way=args.ge_one_way, ge_row_dir=args.ge_row_dir, ge_col_dir=args.ge_col_dir)

    # Make sure all output directories exist
    for f in (args.out_layout, args.out_heat, args.out_bars, args.out_json, args.out_gif):
        _ensure_parent(f)

    chosen_state = None
    chosen_score = float('inf')

    if args.algo == 'none':
        chosen_state = random_baseline_state(cfg, cfg.seed)
        chosen_score, *_ = evaluate(chosen_state, cfg, cfg.seed)
        print("[starter] Using random baseline layout. Implement HC/SA/GA in algos/*.py to improve it.")
    else:
        # HC
        if args.algo in ('hc','all'):
            if hill_climb is None:
                print("[warn] Hill Climbing not implemented (algos/hill.py). Skipping.")
            else:
                try:
                    s, sc, _ = hill_climb(cfg, steps=args.steps, restarts=args.restarts, seed=args.seed)
                    print(f"HC best score = {sc:.3f}")
                    chosen_state, chosen_score = s, sc
                except NotImplementedError as e:
                    print("[warn] Hill Climbing raised NotImplementedError:", e)

        # SA
        if args.algo in ('sa','all'):
            if simulated_annealing is None:
                print("[warn] Simulated Annealing not implemented (algos/sa.py). Skipping.")
            else:
                try:
                    s, sc, _ = simulated_annealing(cfg, steps=args.steps, T0=args.T0, alpha=args.alpha, seed=args.seed)
                    print(f"SA best score = {sc:.3f}")
                    if chosen_state is None or sc < chosen_score:
                        chosen_state, chosen_score = s, sc
                except NotImplementedError as e:
                    print("[warn] Simulated Annealing raised NotImplementedError:", e)

        # GA
        if args.algo in ('ga','all'):
            if genetic_algorithm is None:
                print("[warn] Genetic Algorithm not implemented (algos/ga.py). Skipping.")
            else:
                try:
                    s, sc = genetic_algorithm(cfg, pop_size=args.pop, generations=args.gens, seed=args.seed)
                    print(f"GA best score = {sc:.3f}")
                    if chosen_state is None or sc < chosen_score:
                        chosen_state, chosen_score = s, sc
                except NotImplementedError as e:
                    print("[warn] Genetic Algorithm raised NotImplementedError:", e)

        # If none ran successfully, fall back
        if chosen_state is None:
            chosen_state = random_baseline_state(cfg, cfg.seed)
            chosen_score, *_ = evaluate(chosen_state, cfg, cfg.seed)
            print("[starter] Falling back to random baseline layout.")

    # Final evaluation + outputs
    score, metrics, per_station, heat, _ = evaluate(chosen_state, cfg, cfg.seed)
    # You may uncomment this once students implement the objective:
    # print(f"Chosen solution score = {score:.3f} (J1={metrics.get('J1')}, J2={metrics.get('J2')}, J3={metrics.get('J3')}, penalty={metrics.get('penalty')})")

    draw_layout(chosen_state, args.out_layout)
    draw_heatmap(heat, args.out_heat)
    bar_station(per_station, args.out_bars)

    with open(args.out_json, 'w') as f:
        json.dump({
            'score': score,
            'metrics': metrics,
            'rows': cfg.rows, 'cols': cfg.cols, 'n_stations': cfg.n_stations,
            'orders': cfg.orders, 'items_per_order': cfg.items_per_order, 'zipf_alpha': cfg.zipf_alpha,
            'weights': {'w1': cfg.w1, 'w2': cfg.w2, 'w3': cfg.w3},
            'seed': cfg.seed,
            'GE': {'one_way': cfg.ge_one_way, 'row_dir': cfg.ge_row_dir, 'col_dir': cfg.ge_col_dir}
        }, f, indent=2)

    if args.animate:
        _, _, _, _, paths, picks = simulate_with_paths(chosen_state, cfg, cfg.seed, limit_orders=args.anim_orders)
        animate_paths(chosen_state, paths, picks,
                      out_gif=args.out_gif,
                      sec_per_step=args.sec_per_step,
                      stagger=args.stagger,
                      dot_size=args.dot_size,
                      show_live=args.show_live)
        print(f"Saved GIF → {args.out_gif}")

    print(f"Saved layout → {args.out_layout}")
    print(f"Saved heatmap → {args.out_heat}")
    print(f"Saved station bars → {args.out_bars}")
    print(f"Saved JSON → {args.out_json}")


if __name__ == '__main__':
    main()
