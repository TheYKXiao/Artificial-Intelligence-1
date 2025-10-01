# Warehouse Layout Optimization — Assignment README

This folder is an assignment starter for a warehouse layout optimization task.
The code simulates orders, computes routing costs, produces static plots, and can
render a small GIF animation. 

Your job is to implement optimization algorithms and design an objective function that improves performance compared to the random baseline.


## Quick checklist (what to hand in)
- Your implemented algorithm files (place in `algos/` or keep the provided structure).
- You also need to define your own objective function inside sim.py (the placeholder is marked with a TODO comment). Your score should combine distance, congestion, fairness, and penalties but the exact formula is up to you.
- A short report (1–2 pages) describing your methods and results.
- The outputs: `layout.png`, `heat.png`, `station_bars.png`, `summary.json`.
- (Optional) `traffic.gif` if you generate an animation.


## Submission
- Place your solution, code and pdf in a single zip folder with your name and x500 (e.g., `joe_999.zip`) and upload
to canvas.
- Try to Keep directory structure intact (If you make a modification, please write a update this readme file)
- Make sure summary.json is in the folder you zip

## Project Structure

```
warehouse_starter/
│
├── main.py          # Entry point: presets, argument parsing, orchestration
├── config.py        # Configuration dataclass (grid size, orders, weights, seed, etc.)
├── env.py           # Environment: grid, arteries, shelves/stations placement helpers
├── sim.py           # Simulation & objective scoring (J1/J2/J3 + penalties); optional path 
├── viz.py           # Plots: layout.png, heat.png, station_bars.png
├── animate.py       # GIF animation: step-by-step robot movement and pickups
│
└── algos/           # Your algorithms (start here, but feel free to expand/replace)
    ├── hill.py      # implement hill_climb(...)
    ├── sa.py        # implement simulated_annealing(...)
    └── ga.py        # implement genetic_algorithm(...)
```


## Setup (local)
1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (repository includes `requirements.txt`):

```bash
pip install -r requirements.txt
```

If you don't want to use a virtualenv, installing with `pip` globally also works.



## Exact commands students should use
Run these from inside the `warehouse_starter/` directory.

1) Quick baseline run (produces static outputs):

```bash
python main.py --algo none 
```

Files produced (defaults):
- `out/layout.png`
- `out/heat.png`
- `out/station_bars.png`
- `out/summary.json`

2) Run a single algorithm (example: hill climbing)

```bash
python main.py --algo hc --steps 4000 --restarts 4 
```

3) Run simulated annealing with typical params:

```bash
python main.py --algo sa --steps 4000 --T0 2.0 --alpha 0.995 
```

4) Run genetic algorithm (example):

```bash
python main.py --algo ga --pop 30 --gens 100 
```

5) Generate an animation GIF (small number of orders recommended):

```bash
python main.py --algo none --animate --anim-orders 25 --out-gif out/traffic.gif
```

6) Run all algos and pick the best automatically:

```bash
python main.py --algo all 
```

Notes about flags
- `--seed` controls randomness; use it to make runs reproducible.
- Output paths are controlled by `--out-layout`, `--out-heat`, `--out-bars`, and
  `--out-json` (see `main.py` for defaults).
- Presets may be available in this repository — check the `main.py` parsing if
  your course version exposes `--preset` flags.

## What `summary.json` contains (important for grading)
After a run `main.py` writes `out/summary.json` containing at minimum:
- `score` — final objective value (lower is better).
- `J` — breakdown into components (J1, J2, J3, penalty).
- basic run configuration (rows, cols, n_stations, orders, items_per_order, zipf_alpha).
- algorithm parameters and the `seed`.

The grader will read `out/summary.json` to get your best score and to confirm
which algorithm produced it. Make sure `summary.json` is present in your submitted
folder.

## Testing & validation 
1. The submission contains `summary.json` and the static images listed above.
2. `summary.json['score']` is lower than the baseline (or compared to peers as required).
3. The code runs without modification from the `warehouse_starter/` directory using
   the exact commands above.

To self-check locally, run the baseline and your algorithm and compare:

```bash
python main.py --algo none --seed 1
python main.py --algo hc --steps 2000 --restarts 2 --seed 1
```

## Tips 
- Start small: verify `--algo none` baseline and inspect the images.
- Add unit tests for any helper you write (small fast tests around neighbor moves and
  state validity are very helpful).
- Log progress: keep a score curve output or save intermediate `summary.json` files.
- Keep changes modular: if you replace the interfaces, update `main.py` accordingly
  and document how to run your version.

## Troubleshooting
- If imports fail, ensure you run `python` from inside the `warehouse_starter/`
  directory or add the folder to `PYTHONPATH`.
- If GIF saving fails, install `pillow`:

```bash
pip install pillow
```


## Submission checklist (final)
-  Code implementing algorithms (in `algos/`), or clear instructions where they live
-  Objective filled in sim.py (TODO removed)
-  `out/layout.png`, `out/heat.png`, `out/station_bars.png`
-  `out/summary.json` (must contain `score` and `J` fields)
-  Report (PDF, 1–2 pages)
-  (Optional) `out/traffic.gif` for animation
